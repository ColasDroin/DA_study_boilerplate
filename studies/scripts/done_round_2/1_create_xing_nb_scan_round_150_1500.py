# ==================================================================================================
# --- Imports
# ==================================================================================================
import copy
import itertools
import os
import time

import numpy as np
import yaml
from generate_run_file import (
    generate_run_sh,
    generate_run_sh_htc,
)
from tree_maker import initialize

# ==================================================================================================
# --- Initial particle distribution parameters (generation 1)
#
# Below, the user defines the parameters for the initial particles distribution.
# Path for the particle distribution configuration:
# master_study/master_jobs/1_build_distr_and_collider/config.yaml [field config_particles]
# ==================================================================================================

# Define dictionary for the initial particle distribution
d_config_particles = {}

# Radius of the initial particle distribution
d_config_particles["r_min"] = 2
d_config_particles["r_max"] = 10
d_config_particles["n_angles"] = 11

# Number of angles for the initial particle distribution
d_config_particles["n_r"] = 1280 // d_config_particles["n_angles"]

# Number of split for parallelization
d_config_particles["n_split"] = 5

# ==================================================================================================
# --- Optics collider parameters (generation 1)
#
# Below, the user defines the optics collider parameters. These parameters cannot be scanned.
# Path for the collider configuration:
# master_study/master_jobs/1_build_distr_and_collider/config.yaml [field config_mad]
# ==================================================================================================

### Mad configuration

# Define dictionary for the Mad configuration
d_config_mad = {"beam_config": {"lhcb1": {}, "lhcb2": {}}, "links": {}}

# Optic file path (version, and round or flat)

### For v1.6 optics
d_config_mad["links"]["acc-models-lhc"] = "../../../../external_dependencies/acc-models-lhc"
d_config_mad["optics_file"] = "acc-models-lhc/strengths/flat/opt_flatvh_75_180_1500_thin.madx"
d_config_mad["ver_hllhc_optics"] = 1.6


# Beam energy (for both beams)
beam_energy_tot = 7000
d_config_mad["beam_config"]["lhcb1"]["beam_energy_tot"] = beam_energy_tot
d_config_mad["beam_config"]["lhcb2"]["beam_energy_tot"] = beam_energy_tot


# ==================================================================================================
# --- Base collider parameters (generation 2)
#
# Below, the user defines the standard collider parameters. Some of the values defined here are
# later updated according to the grid-search being done.
# Path for the collider config:
# master_study/master_jobs/2_configure_and_track/config.yaml [field config_collider]
# ==================================================================================================

### Tune and chroma configuration

# Define dictionnary for tune and chroma
d_config_tune_and_chroma = {
    "qx": {},
    "qy": {},
    "dqx": {},
    "dqy": {},
}
for beam in ["lhcb1", "lhcb2"]:
    d_config_tune_and_chroma["qx"][beam] = 62.315
    d_config_tune_and_chroma["qy"][beam] = 60.320
    d_config_tune_and_chroma["dqx"][beam] = 5.0
    d_config_tune_and_chroma["dqy"][beam] = 5.0

# Value to be added to linear coupling knobs
d_config_tune_and_chroma["delta_cmr"] = 0.001  # type: ignore
d_config_tune_and_chroma["delta_cmi"] = 0.0  # type: ignore

### Knobs configuration

# Define dictionary for the knobs settings
d_config_knobs = {}

# Knobs at IPs
d_config_knobs["on_x1"] = np.nan  # scanned
d_config_knobs["on_sep1"] = 0
d_config_knobs["on_x2"] = -170
d_config_knobs["on_sep2"] = 0.138
d_config_knobs["on_x5"] = np.nan  # scanned
d_config_knobs["on_sep5"] = 0
d_config_knobs["on_x8h"] = 0.0
d_config_knobs["on_x8v"] = 170

# Crab cavities
d_config_knobs["on_crab1"] = -190
d_config_knobs["on_crab5"] = -190

# Octupoles
d_config_knobs["i_oct_b1"] = -60.0
d_config_knobs["i_oct_b2"] = -60.0

### leveling configuration

# Leveling in IP 1/5
d_config_leveling_ip1_5 = {"constraints": {}, "skip_leveling": True}
d_config_leveling_ip1_5["luminosity"] = 5.0e34  # type: ignore
d_config_leveling_ip1_5["constraints"]["max_intensity"] = 2.3e11
d_config_leveling_ip1_5["constraints"]["max_PU"] = 160


# Define dictionary for the leveling settings
d_config_leveling = {
    "ip2": {},
    "ip8": {},
}

# Luminosity and particles


# Leveling parameters (ignored if skip_leveling is True)
d_config_leveling["ip2"]["separation_in_sigmas"] = 5
d_config_leveling["ip8"]["luminosity"] = 2.0e33

### Beam beam configuration

# Define dictionary for the beam beam settings
d_config_beambeam = {"mask_with_filling_pattern": {}}

# Beam settings
d_config_beambeam["num_particles_per_bunch"] = np.nan  # ! scanned
d_config_beambeam["nemitt_x"] = 2.5e-6  # type: ignore
d_config_beambeam["nemitt_y"] = 2.5e-6  # type: ignore

# Filling scheme (in json format)
# The scheme should consist of a json file containing two lists of booleans (one for each beam),
# representing each bucket of the LHC.
# Alternatively, one can get a fill directly from LPC from, e.g.:
# https://lpc.web.cern.ch/cgi-bin/fillTable.py?year=2023
# In this page, get the fill number of your fill of interest, and use it to replace the XXXX in the
# URL below before downloading:
# https://lpc.web.cern.ch/cgi-bin/schemeInfo.py?fill=XXXX&fmt=json
filling_scheme_path = os.path.abspath(
    "../filling_scheme/25ns_2760b_2748_2492_2574_288bpi_13inj_800ns_bs200ns_converted.json"
)
# Add to config file
d_config_beambeam["mask_with_filling_pattern"]["pattern_fname"] = filling_scheme_path

# Initialize bunch number
# If set to None, it will be set automatically to the worst bunch when running 2nd generation
d_config_beambeam["mask_with_filling_pattern"]["i_bunch_b1"] = None
d_config_beambeam["mask_with_filling_pattern"]["i_bunch_b2"] = None

# ==================================================================================================
# --- Generate dictionnary to encapsulate all base collider parameters (generation 2)
# ==================================================================================================
d_config_collider = {}

# Add tunes and chromas
d_config_collider["config_knobs_and_tuning"] = d_config_tune_and_chroma

# Add knobs
d_config_collider["config_knobs_and_tuning"]["knob_settings"] = d_config_knobs

# Add luminosity configuration
d_config_collider["config_lumi_leveling_ip1_5"] = d_config_leveling_ip1_5
d_config_collider["config_lumi_leveling"] = d_config_leveling

# Add beam beam configuration
d_config_collider["config_beambeam"] = d_config_beambeam

# ==================================================================================================
# --- Tracking parameters (generation 2)
#
# Below, the user defines the parameters for the tracking.
# ==================================================================================================
d_config_simulation = {}

# Number of turns to track
d_config_simulation["n_turns"] = 1000000

# Initial off-momentum
d_config_simulation["delta_max"] = 27.0e-5

# Beam to track (lhcb1 or lhcb2)
d_config_simulation["beam"] = "lhcb1"

# ==================================================================================================
# --- Dump collider and collider configuration
#
# Below, the user chooses if the gen 2 collider must be dumped, along with the corresponding
# configuration.
# ==================================================================================================
dump_collider = False
dump_config_in_collider = False

# ==================================================================================================
# --- Machine parameters being scanned (generation 2)
#
# Below, the user defines the grid for the machine parameters that must be scanned to find the
# optimal DA (e.g. tune, chroma, etc).
# ==================================================================================================
# Scan tune with step of 0.001 (need to round to correct for numpy numerical instabilities)
array_xing = np.linspace(120, 270, 16, endpoint=True)
array_nb = np.linspace(0.8, 1.6, 17, endpoint=True) * 1e11

# ==================================================================================================
# --- Make tree for the simulations (generation 1)
#
# The tree is built as a hierarchy of dictionnaries. We add a first generation (named as the
# study being done) to the root. This first generation is used set the initial particle
# distribution, and build a collider with only the optics set.
# ==================================================================================================

# Build empty tree: first generation (later added to the root), and second generation
children = {"base_collider": {"config_particles": {}, "config_mad": {}, "children": {}}}

# Add particles distribution parameters to the first generation
children["base_collider"]["config_particles"] = d_config_particles

# Add base machine parameters to the first generation
children["base_collider"]["config_mad"] = d_config_mad


# ==================================================================================================
# --- Complete tree for the simulations (generation 2)
#
# We now set a second generation for the tree. This second generation contains the tracking
# parameters, as well as a default set of parameters for the colliders (defined above), that we
# mutate according to the parameters we want to scan.
# ! Caution when mutating the dictionnary in this function, you have to pass a deepcopy to children,
# ! otherwise the dictionnary will be mutated for all the children.
# ==================================================================================================
track_array = np.arange(d_config_particles["n_split"])
for idx_job, (track, nb, xing) in enumerate(itertools.product(track_array, array_nb, array_xing)):
    # Mutate the appropriate collider parameters
    for beam in ["lhcb1", "lhcb2"]:
        d_config_collider["config_knobs_and_tuning"]["knob_settings"]["on_x1"] = float(xing)
        d_config_collider["config_knobs_and_tuning"]["knob_settings"]["on_x5"] = float(xing)
        d_config_collider["config_beambeam"]["num_particles_per_bunch"] = float(nb)
        if xing < 190:
            d_config_collider["config_knobs_and_tuning"]["knob_settings"]["on_crab1"] = -1 * float(
                xing
            )
            d_config_collider["config_knobs_and_tuning"]["knob_settings"]["on_crab5"] = -1 * float(
                xing
            )

    # Complete the dictionnary for the tracking
    d_config_simulation["particle_file"] = f"../particles/{track:02}.parquet"
    d_config_simulation["collider_file"] = "../collider.json.zip"

    # Add a child to the second generation, with all the parameters for the collider and tracking
    children["base_collider"]["children"][f"xtrack_{idx_job:04}"] = {
        "config_simulation": copy.deepcopy(d_config_simulation),
        "config_collider": copy.deepcopy(d_config_collider),
        "log_file": "tree_maker.log",
        "dump_collider": dump_collider,
        "dump_config_in_collider": dump_config_in_collider,
    }

# ==================================================================================================
# --- Simulation configuration
# ==================================================================================================
# Load the tree_maker simulation configuration
config = yaml.safe_load(open("config.yaml"))

# # Set the root children to the ones defined above
config["root"]["children"] = children

# Set miniconda environment path in the config
config["root"]["setup_env_script"] = os.getcwd() + "/../../source_python.sh"


# Recursively define the context for the simulations
def set_context(children, idx_gen, config):
    for child in children.values():
        child["context"] = config["root"]["generations"][idx_gen]["context"]
        if "children" in child.keys():
            set_context(child["children"], idx_gen + 1, config)


set_context(children, 1, config)
# ==================================================================================================
# --- Build tree and write it to the filesystem
# ==================================================================================================
# Define study name
study_name = "xing_nb_scan_round_150_1500"

# Creade folder that will contain the tree
if not os.path.exists(f"../scans/{study_name}"):
    os.makedirs(f"../scans/{study_name}")
else:
    answer = input(
        "The folder already exists. You might delete an existing study. "
        "Do you want to overwrite it? [y/n]"
    )
    if answer != "y":
        print("Aborting...")
        exit()

# Move to the folder that will contain the tree
os.chdir(f"../scans/{study_name}")

# Clean the id_job file
id_job_file_path = "id_job.yaml"
if os.path.isfile(id_job_file_path):
    os.remove(id_job_file_path)

# Create tree object
start_time = time.time()
root = initialize(config)
print("Done with the tree creation.")
print(f"--- {time.time() - start_time} seconds ---")

# Check if htcondor is the configuration
if "htc" in config["root"]["generations"][2]["run_on"]:
    generate_run = generate_run_sh_htc
else:
    generate_run = generate_run_sh

# From python objects we move the nodes to the filesystem.
start_time = time.time()
root.make_folders(generate_run)
print("The tree folders are ready.")
print(f"--- {time.time() - start_time} seconds ---")
