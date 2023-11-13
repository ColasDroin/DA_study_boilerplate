"""This script is used to configure the collider and track the particles. Functions in this script
are called sequentially, in the order in which they are defined. Modularity has been favored over 
simple scripting for reproducibility, to allow rebuilding the collider from a different program 
(e.g. dahsboard)."""

# ==================================================================================================
# --- Imports
# ==================================================================================================
import json
import logging
import os
import pickle
import time
from datetime import datetime

import numpy as np
import pandas as pd
import ruamel.yaml
import tree_maker
import xmask as xm
import xtrack as xt

# Initialize yaml reader
ryaml = ruamel.yaml.YAML()


# ==================================================================================================
# --- Function for tree_maker tagging
# ==================================================================================================
def tree_maker_tagging(config, tag="started"):
    # Start tree_maker logging if log_file is present in config
    if tree_maker is not None and "log_file" in config:
        tree_maker.tag_json.tag_it(config["log_file"], tag)
    else:
        logging.warning("tree_maker loging not available")


# ==================================================================================================
# --- Functions to read configuration files and generate configuration files for orbit correction
# ==================================================================================================
def read_configuration(config_path="config.yaml"):
    # Read configuration for simulations
    with open(config_path, "r") as fid:
        config = ryaml.load(fid)

    return config


def define_knob_separation(collider, path_dic_elements):

    # load dictionnary from pickle
    with open(path_dic_elements, "rb") as fid:
        dic_elements = pickle.load(fid)

    # define knob separation
    print(dic_elements)
    pass

    return collider


# ==================================================================================================
# --- Main function for collider configuration
# ==================================================================================================
def configure_collider(config):
    config_sim = config["config_simulation"]

    # Rebuild collider
    collider = xt.Multiline.from_json(config_sim["collider_file"])

    # Build trackers
    collider.build_trackers()

    # Build knob
    collider = define_knob_separation(collider, config_sim["elements_file"])

    return collider


# ==================================================================================================
# --- Main function for collider configuration and tracking
# ==================================================================================================
def configure_and_track(config_path="config.yaml"):
    # Get configuration
    config = read_configuration(config_path)

    # Tag start of the job
    tree_maker_tagging(config, tag="started")

    # Configure collider (not saved, since it may trigger overload of afs)
    collider = configure_collider(config)

    # Dump collider
    collider.to_json("collider.json")

    # Remote the correction folder, and potential C files remaining
    try:
        os.system("rm -rf correction")
        os.system("rm -f *.cc")
    except:
        pass

    # Tag end of the job
    tree_maker_tagging(config, tag="completed")


# ==================================================================================================
# --- Script for execution
# ==================================================================================================

if __name__ == "__main__":
    configure_and_track()


def track(collider, particles, config_sim, config_bb=None, save_input_particles=False):
    # Get beam being tracked
    beam = config_sim["beam"]

    # Optimize line for tracking # ! Commented out as it prevents changing the bb
    # collider[beam].optimize_for_tracking()

    # Save initial coordinates if requested
    if save_input_particles:
        pd.DataFrame(particles.to_dict()).to_parquet("input_particles.parquet")

    # Track (update bb in several steps)
    num_turns = config_sim["n_turns"]
    a = time.time()

    # Define steps for separation update
    n_steps = 20
    initial_sep_1 = collider.vars["on_sep1"]._value
    initial_sep_5 = collider.vars["on_sep5"]._value
    num_turns_step = int(num_turns / (n_steps + 1))
    sep_1_step = initial_sep_1 / n_steps
    sep_5_step = initial_sep_5 / n_steps

    # Function to compute footprint
    def return_footprint(collider, emittance, beam="lhcb1", n_turns=2000):
        fp_polar_xm = collider[beam].get_footprint(
            nemitt_x=emittance,
            nemitt_y=emittance,
            n_turns=n_turns,
            linear_rescale_on_knobs=[xt.LinearRescale(knob_name="beambeam_scale", v0=0.0, dv=0.05)],
            freeze_longitudinal=True,
        )

        qx = fp_polar_xm.qx
        qy = fp_polar_xm.qy

        return qx, qy

    time_simulated = 0
    time_reconfigured = 0


    dic_set_attr = {"bb_lr": set_attr_lr, "bb_ho": set_attr_ho}
    dic_elements_names = {
        beam_temp: {
            type_bb: [x for x in collider[beam_temp].element_names if type_bb in x]
            for type_bb in ["bb_lr", "bb_ho"]
        }
        for beam_temp in ["lhcb1", "lhcb2"]
    }

    # Function to build a multisetter for each attribute
    def build_multisetters(set_attr, beam_temp, l_elements):
        dic_setters = {}
        for attr in set_attr:
            dic_setters[attr] = xt.MultiSetter(
                collider[beam_temp],
                l_elements,
                field=attr,
                index=(
                    0
                    if (
                        isinstance(getattr(collider[beam_temp][l_elements[0]], attr), list)
                        or isinstance(getattr(collider[beam_temp][l_elements[0]], attr), np.ndarray)
                    )
                    else None
                ),
            )
        return dic_setters

    time_start = time.time()
    factor = int(20 / 5)
    dic_meta_setters = {}
    for i in range(n_steps + 1):
        # Update separation and reconfigure beambeam
        collider.vars["on_sep1"] = initial_sep_1 - i * sep_1_step
        collider.vars["on_sep5"] = initial_sep_5 - i * sep_5_step
        print(
            f"Updating on_sep1 to {collider.vars['on_sep1']._value} on_sep5 to"
            f" {collider.vars['on_sep5']._value}"
        )

        if config_bb is not None:
            t_before_reconfigure = time.time()
            if i == 0:
                collider = configure_beam_beam(collider, config_bb)
                dic_meta_setters = {
                    beam_temp: {
                        type_bb: build_multisetters(
                            set_attr_bb, beam_temp, dic_elements_names[beam_temp][type_bb]
                        )
                        for type_bb, set_attr_bb in dic_set_attr.items()
                    }
                    for beam_temp in dic_elements_names
                }
            else:
                print("Loading elements from dictionnary")
                with open(
                    f"../xtrack_0000_precompute_multiset/bb_elements_step_{i//factor}.pkl", "rb"
                ) as fid:
                    dic_elements = pickle.load(fid)
                try:
                    with open(
                        f"../xtrack_0000_precompute_multiset/bb_elements_step_{i//factor+1}.pkl",
                        "rb",
                    ) as fid:
                        dic_elements_2 = pickle.load(fid)
                except FileNotFoundError:
                    print("Last step, using same elements")
                    dic_elements_2 = dic_elements

                # Interpolate element values
                fraction = (i % factor) / factor
                for beam_temp in dic_elements:
                    for type_bb in dic_elements[beam_temp]:
                        for element in dic_elements[beam_temp][type_bb]:
                            for attr in dic_elements[beam_temp][type_bb][element]:
                                attr_val = dic_elements[beam_temp][type_bb][element][attr]
                                attr_val_2 = dic_elements_2[beam_temp][type_bb][element][attr]

                                if isinstance(attr_val, list) or isinstance(attr_val, np.ndarray):
                                    for j, sub_attr in enumerate(attr_val):
                                        attr_val[j] = (
                                            attr_val[j] * (1 - fraction) + attr_val_2[j] * fraction
                                        )

                                else:
                                    # Get type of attribute
                                    attr_type = type(attr_val)
                                    # Interpolate
                                    attr_val = attr_val * (1 - fraction) + attr_val_2 * fraction
                                    # Cast back to original type
                                    attr_val = attr_type(attr_val)

                                # Update value
                                dic_elements[beam_temp][type_bb][element][attr] = attr_val

                # Dump bb elements in a pickle
                print("Dumping elements in a pickle. i=", i)
                with open(f"bb_elements_step_{i}.pkl", "wb") as fid:
                    pickle.dump(dic_elements, fid)

                # Reconfigure beambeam
                for beam_temp in dic_meta_setters:
                    for type_bb in dic_meta_setters[beam_temp]:
                        for attr in dic_set_attr[type_bb]:
                            dic_meta_setters[beam_temp][type_bb][attr].set_values(
                                [
                                    dic_elements[beam_temp][type_bb][element][attr]
                                    for element in dic_elements_names[beam_temp][type_bb]
                                ]
                            )

            # collider.to_json(f"collider_step_{i}.json")
            t_after_reconfigure = time.time()
            time_reconfigured += t_after_reconfigure - t_before_reconfigure
        else:
            raise ValueError("Beam-beam configuration is required for dynamic tracking.")

        # Track until next checkpoint
        t_before_tracking = time.time()
        collider[beam].track(particles, turn_by_turn_monitor=False, num_turns=num_turns_step)
        t_after_tracking = time.time()
        time_simulated += t_after_tracking - t_before_tracking
    time_end = time.time()
    b = time.time()

    print(f"Total time simulation: {time_end - time_start} s")
    print(f"Total time reconfiguration: {time_reconfigured} s")
    print(f"Average time per reconfiguration: {time_reconfigured / (n_steps + 1)} s")
    print(f"Total time tracking: {time_simulated} s")
    print(f"Average time tracking per turn: {time_simulated / num_turns} s")

    # print(f"Elapsed time: {b-a} s")
    print(f"Elapsed time per particle per turn: {(b-a)/particles._capacity/num_turns*1e6} us")

    return particles
