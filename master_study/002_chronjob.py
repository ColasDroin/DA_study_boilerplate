# ==================================================================================================
# --- Imports
# ==================================================================================================
import tree_maker
import os
import psutil
from pathlib import Path


# ==================================================================================================
# --- Class for job submission
# ==================================================================================================
class ClusterSubmission:
    def __init__(self, run_on="local_pc"):
        if run_on in ["local_pc", "htc", "slurm", "htc_docker", "slurm_docker"]:
            self.run_on = run_on
        else:
            raise ("Error: Submission mode specified is not yet implemented")

        # Dictionnary of instructions for submission files
        self.dic_submission = {
            "local_pc": {
                "head": "# Running on local pc\n",
                "body": lambda path_node: f"bash {path_node}/run.sh &\n",
                "tail": f"#{self.run_on}\n",
                "submit_command": lambda filename: f"bash {filename}",
            },
            "slurm": {
                "head": "# Running on SLURM \n",
                "body": lambda path_node: (
                    "sbatch --ntasks=2 --partition=slurm_hpc_acc --output=output.txt"
                    f" {path_node}/run.sh\n"
                ),
                "tail": f"#{self.run_on}\n",
                "submit_command": lambda filename: f"sbatch {filename}",
            },
            # ! This is not being used right now, to be fixed
            "slurm_docker": {
                "head": (
                    "# This is a SLURM submission file using Docker\n"
                    + "#SBATCH --partition=slurm_hpc_acc\n"
                    + "#SBATCH --output=output.txt\n"
                    + "#SBATCH --error=error.txt\n"
                    + "#SBATCH --ntasks=2\n"
                ),
                "body": lambda path_node: (
                    f"singularity exec ../da-study-docker_latest.sif {path_node}/run.sh\n"
                ),
                "tail": f"#{self.run_on}\n",
                "submit_command": lambda filename: f"sbatch {filename}",
            },
            "htc": {
                "head": (
                    "# This is a HTCondor submission file\n"
                    + "error  = error.txt\n"
                    + "output = output.txt\n"
                    + "log  = log.txt\n"
                ),
                "body": (
                    lambda path_node: f"initialdir = {path_node}\n"
                    + f"executable = {path_node}/run.sh\nqueue\n"
                ),
                "tail": f"#{self.run_on}\n",
                "submit_command": lambda filename: f"condor_submit {filename}",
            },
            "htc_docker": {
                "head": (
                    "# This is a HTCondor submission file using Docker\n"
                    + "error  = error.txt\n"
                    + "output = output.txt\n"
                    + "log  = log.txt\n"
                    + "universe = vanilla\n"
                    + "+SingularityImage ="
                    ' "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cdroin/da-study-docker:latest"\n'
                ),
                "body": (
                    lambda path_node: f"initialdir = {path_node}\n"
                    + f"executable = {path_node}/run.sh\nqueue\n"
                ),
                "tail": f"#{self.run_on}\n",
                "submit_command": lambda filename: f"condor_submit {filename}",
            },
        }

    def _get_state_jobs(self, verbose=True):
        running_jobs = self.running_jobs()
        queuing_jobs = self.queuing_jobs()
        if verbose:
            print(f"running: {running_jobs}")
            print(f"queuing: {queuing_jobs}")
        return running_jobs, queuing_jobs

    @staticmethod
    def _test_node(node, path_node, running_jobs, queuing_jobs):
        # Test if node is running, queuing or completed
        if node.has_been("completed"):
            print(f"{path_node} is completed.")
        elif path_node in running_jobs:
            print(f"{path_node} is running.")
        elif node in queuing_jobs:
            print(f"{path_node} is queuing.")
        else:
            return True
        return False

    def _write_sub_files_slurm(self, filename, running_jobs, queuing_jobs, list_of_nodes):
        l_filenames = []
        for idx_node, node in enumerate(list_of_nodes):
            # Get path node
            path_node = node.get_abs_path()

            # Test if node is running, queuing or completed
            if self._test_node(node, path_node, running_jobs, queuing_jobs):
                filename_node = f"{filename.split('.sub')[0]}_{idx_node}.sub"

                # Write the submission files
                with open(filename_node, "w") as fid:
                    # Head
                    fid.write("#!/bin/bash\n")
                    fid.write("# This is a SLURM submission file using Docker\n")
                    fid.write("#SBATCH --partition=slurm_hpc_acc\n")
                    fid.write("#SBATCH --output=output.txt\n")
                    fid.write("#SBATCH --error=error.txt\n")
                    fid.write("#SBATCH --ntasks=2\n")

                    # Careful, I implemented a fix for path due to the temporary home recovery folder
                    to_replace = "/storage-hpc/gpfs_data/HPC/home_recovery"
                    replacement = "/home/HPC"
                    fixed_path = path_node.replace(to_replace, replacement)
                    # update path for sed
                    to_replace = to_replace.replace("/", "\/")
                    replacement = replacement.replace("/", "\/")
                    # Mutate path in run.sh and other potentially problematic files
                    fid.write(f"sed -i 's/{to_replace}/{replacement}/' {fixed_path}/run.sh\n")
                    fid.write(f"sed -i 's/{to_replace}/{replacement}/' {fixed_path}/config.yaml\n")
                    # Final submission line
                    fid.write(
                        f"singularity exec ../da-study-docker_latest.sif {fixed_path}/run.sh\n"
                    )

                    # Tail
                    fid.write(f"#{self.run_on}\n")
                l_filenames.append(filename_node)
        return l_filenames

    def _write_sub_file(
        self, filename, running_jobs, queuing_jobs, list_of_nodes, write_htc_job_flavour=False
    ):
        # Get submission instructions
        str_head = self.dic_submission[self.run_on]["head"]
        str_body = self.dic_submission[self.run_on]["body"]
        str_tail = self.dic_submission[self.run_on]["tail"]

        # Write the submission file
        with open(filename, "w") as fid:
            fid.write(str_head)
            for node in list_of_nodes:
                # Get path node
                path_node = node.get_abs_path()
                # Test if node is running, queuing or completed
                if self._test_node(node, path_node, running_jobs, queuing_jobs):
                    # Write instruction for submission
                    fid.write(str_body(path_node))

                    # if user has defined a htc_job_flavor in config.yaml otherwise default is "espresso"
                    if (
                        "htc_job_flavor"
                        in list_of_nodes[0].root.parameters["generations"][
                            str(list_of_nodes[0].depth)
                        ]
                    ) and write_htc_job_flavour:
                        htc_job_flavor = list_of_nodes[0].root.parameters["generations"][
                            str(list_of_nodes[0].depth)
                        ]["htc_job_flavor"]
                        fid.write(f'+JobFlavour  = "{htc_job_flavor}"\n')

            # Tail instruction
            fid.write(str_tail)

    def _write_sub_files(self, filename, running_jobs, queuing_jobs, list_of_nodes):
        # Slurm docker is a peculiar case as one submission file must be created per job
        if self.run_on == "slurm_docker":
            return self._write_sub_files_slurm(filename, running_jobs, queuing_jobs, list_of_nodes)

        # htcondor, local_pc, etc.
        else:
            self._write_sub_file(
                filename,
                running_jobs,
                queuing_jobs,
                list_of_nodes,
                write_htc_job_flavour=True if self.run_on in ["htc", "htc_docker"] else False,
            )
            return [filename]

    def write_sub_files(self, list_of_nodes, filename="file.sub"):
        running_jobs, queuing_jobs = self._get_state_jobs(verbose=True)
        l_filenames = self._write_sub_files(filename, running_jobs, queuing_jobs, list_of_nodes)
        return l_filenames

    def submit(self, l_filenames):
        # Check that the submission file(s) is/are appropriate for the submission mode
        if len(l_filenames) > 1 and self.run_on != "slurm_docker":
            raise (
                "Error: Multiple submission files should not be implemented for this submission"
                " mode"
            )
        if len(l_filenames) == 0:
            print("No submission file to submit")

        # Submit
        for filename in l_filenames:
            if self.run_on in self.dic_submission:
                os.system(self.dic_submission[self.run_on]["submit_command"](filename))
            else:
                raise (f"Error: Submission mode {self.run_on} is not yet implemented")

    def running_jobs(self):
        my_list = []
        if self.run_on == "local_pc":
            # Warning, does not work at the moment in lxplus
            for ps in psutil.pids():
                aux = psutil.Process(ps).cmdline()
                if len(aux) > 1:
                    if "run.sh" in aux[-1]:
                        my_list.append(str(Path(psutil.Process(ps).cmdline()[-1]).parent))

            return my_list
        else:
            print("Running jobs are not implemented yet for this submission mode")
            return []

    def queuing_jobs(self, verbose=False):
        if self.run_on == "local_pc":
            # Always empty return as there is no queuing in local pc
            return []
        else:
            if verbose:
                print("Queuing jobs are not implemented yet for this submission mode")
            return []


# ==================================================================================================
# --- Main submission function
# ==================================================================================================
def submit_jobs_generation(root, generation=1):
    # Submit all the pending jobs of a given generation
    dic_int_to_str = {1: "first", 2: "second", 3: "third"}
    run_on = ClusterSubmission(root.parameters["generations"][f"{generation}"]["run_on"])
    path_file = f"submission_files/{dic_int_to_str[generation]}_generation.sub"
    l_filenames = run_on.write_sub_files(root.generation(generation), path_file)
    run_on.submit(l_filenames)


def submit_jobs(study_name, print_uncompleted_jobs=False):
    # Add suffix to the root node path to handle scans that are not in the root directory
    fix = "/scans/" + study_name
    root = tree_maker.tree_from_json(fix[1:] + "/tree_maker_" + study_name + ".json")
    root.add_suffix(suffix=fix)

    # Check that the study is not done yet
    if root.has_been("completed"):
        print("All descendants of root are completed!")
    else:
        # Do generation 1
        submit_jobs_generation(root, generation=1)

        # Do generation 2 is generation 1 is completed
        if all([node.has_been("completed") for node in root.generation(1)]):
            submit_jobs_generation(root, generation=2)

        # We assume there's no generation 3
        if all([descendant.has_been("completed") for descendant in root.descendants]):
            root.tag_as("completed")
            print("All descendants of root are completed!")

        # Print remaining jobs
        if print_uncompleted_jobs:
            for descendant in root.descendants:
                if descendant.has_not_been("completed"):
                    print("To be completed: " + descendant.get_abs_path())


# ==================================================================================================
# --- Submission
# ==================================================================================================
# Load the tree from a yaml and submit the jobs that haven't been completed yet
if __name__ == "__main__":
    # Define study
    study_name = "example_HL_tunescan"

    # Submit jobs
    submit_jobs(study_name)
