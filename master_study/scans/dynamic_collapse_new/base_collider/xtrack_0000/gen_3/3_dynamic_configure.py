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
from misc import set_attr_ho, set_attr_lr

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
        config_tree_maker = ryaml.load(fid)

    # Also read configuration from previous generation
    with open("../" + config_path, "r") as fid:
        config = ryaml.load(fid)

    return config, config_tree_maker


# ==================================================================================================
# --- Function to configure beam-beam
# ==================================================================================================
def configure_beam_beam(collider, config_bb):
    print(f"Configuring beam-beam with configure_beambeam_interactions (f{datetime.now()})")
    collider.configure_beambeam_interactions(
        num_particles=config_bb["num_particles_per_bunch"],
        nemitt_x=config_bb["nemitt_x"],
        nemitt_y=config_bb["nemitt_y"],
    )

    print(f"Configuring beam-beam with apply_filing_pattern (f{datetime.now()})")
    # Configure filling scheme mask and bunch numbers
    if "mask_with_filling_pattern" in config_bb:
        # Initialize filling pattern with empty values
        filling_pattern_cw = None
        filling_pattern_acw = None

        # Initialize bunch numbers with empty values
        i_bunch_cw = None
        i_bunch_acw = None

        if "pattern_fname" in config_bb["mask_with_filling_pattern"]:
            # Fill values if possible
            if config_bb["mask_with_filling_pattern"]["pattern_fname"] is not None:
                fname = config_bb["mask_with_filling_pattern"]["pattern_fname"]
                with open(fname, "r") as fid:
                    filling = json.load(fid)
                filling_pattern_cw = filling["beam1"]
                filling_pattern_acw = filling["beam2"]

                # Only track bunch number if a filling pattern has been provided
                if "i_bunch_b1" in config_bb["mask_with_filling_pattern"]:
                    i_bunch_cw = config_bb["mask_with_filling_pattern"]["i_bunch_b1"]
                if "i_bunch_b2" in config_bb["mask_with_filling_pattern"]:
                    i_bunch_acw = config_bb["mask_with_filling_pattern"]["i_bunch_b2"]

                # Note that a bunch number must be provided if a filling pattern is provided
                # Apply filling pattern
                collider.apply_filling_pattern(
                    filling_pattern_cw=filling_pattern_cw,
                    filling_pattern_acw=filling_pattern_acw,
                    i_bunch_cw=i_bunch_cw,
                    i_bunch_acw=i_bunch_acw,
                )

    print(f"Done configuring (f{datetime.now()})")
    return collider


# ==================================================================================================
# --- Main function for collider configuration
# ==================================================================================================
def get_dynamic_configure_collider(config):

    # Get configurations
    config_sim = config["config_simulation"]

    # Rebuild collider
    collider = xt.Multiline.from_json(config_sim["collider_file"])
    collider.build_trackers()

    config_beambeam = config["config_collider"]["config_beambeam"]

    # Get dynamic configuration
    dd_elements = save_dynamic_configuration(collider, config_beambeam)

    return dd_elements


# ==================================================================================================
# --- Function to do the tracking
# ==================================================================================================
def save_dynamic_configuration(collider, config_bb):

    # Define steps for separation update
    n_steps = 25
    initial_sep_1 = collider.vars["on_sep1"]._value
    initial_sep_5 = collider.vars["on_sep5"]._value
    sep_1_step = initial_sep_1 / n_steps
    sep_5_step = initial_sep_5 / n_steps

    dic_set_attr = {"bb_lr": set_attr_lr, "bb_ho": set_attr_ho}
    dic_elements_names = {
        beam_temp: {
            type_bb: [x for x in collider[beam_temp].element_names if type_bb in x]
            for type_bb in ["bb_lr", "bb_ho"]
        }
        for beam_temp in ["lhcb1", "lhcb2"]
    }
    time_reconfigured = 0
    time_start = time.time()
    dd_elements = {}
    for i in range(n_steps + 1):
        # Update separation and reconfigure beambeam
        collider.vars["on_sep1"] = initial_sep_1 - i * sep_1_step
        collider.vars["on_sep5"] = initial_sep_5 - i * sep_5_step
        print(
            f"Updating on_sep1 to {collider.vars['on_sep1']._value} on_sep5 to"
            f" {collider.vars['on_sep5']._value}"
        )

        t_before_reconfigure = time.time()
        collider = configure_beam_beam(collider, config_bb)
        print("Dumping elements in dictionnary")
        dic_elements = {
            beam_temp: {
                type_bb: {
                    el: {
                        attr: getattr(collider[beam_temp][el], attr)
                        for attr in dic_set_attr[type_bb]
                    }
                    for el in dic_elements_names[beam_temp][type_bb]
                }
                for type_bb in ["bb_lr", "bb_ho"]
            }
            for beam_temp in ["lhcb1", "lhcb2"]
        }

        dd_elements[collider.vars["on_sep1"]._value] = dic_elements

        t_after_reconfigure = time.time()
        time_reconfigured += t_after_reconfigure - t_before_reconfigure

    time_end = time.time()

    print(f"Total time simulation: {time_end - time_start} s")
    print(f"Total time reconfiguration: {time_reconfigured} s")
    print(f"Average time per reconfiguration: {time_reconfigured / (n_steps + 1)} s")

    return dd_elements


# ==================================================================================================
# --- Main function for collider configuration and tracking
# ==================================================================================================
def configure_and_track(config_path="config.yaml"):
    # Get configuration
    config, config_tree_maker = read_configuration(config_path)

    # Tag start of the job
    tree_maker_tagging(config_tree_maker, tag="started")

    # Configure collider (not saved, since it may trigger overload of afs)
    dd_elements = get_dynamic_configure_collider(config)

    # Save output
    with open("dynamic_configure.pkl", "wb") as fid:
        pickle.dump(dd_elements, fid)

    # Remote the correction folder, and potential C files remaining
    try:
        os.system("rm -rf correction")
        os.system("rm -f *.cc")
    except:
        pass

    # Tag end of the job
    tree_maker_tagging(config_tree_maker, tag="completed")


# ==================================================================================================
# --- Script for execution
# ==================================================================================================

if __name__ == "__main__":
    configure_and_track()
