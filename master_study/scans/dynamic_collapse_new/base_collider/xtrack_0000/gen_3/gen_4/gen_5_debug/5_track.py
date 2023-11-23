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
import time
from datetime import datetime

import dill as pickle
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


# ==================================================================================================
# --- Main function for collider configuration
# ==================================================================================================
def configure_collider(config):

    # Get configurations
    config_sim = config["config_simulation"]
    config_bb = config["config_collider"]["config_beambeam"]

    # Rebuild collider
    collider = xt.Multiline.from_json(config_sim["collider_file"])

    # Build trackers
    collider.build_trackers()

    return collider, config_sim, config_bb


# ==================================================================================================
# --- Function to prepare particles distribution for tracking
# ==================================================================================================
def prepare_particle_distribution(config_sim, collider, config_bb):
    beam = config_sim["beam"]

    particle_df = pd.read_parquet(config_sim["particle_file"])

    r_vect = particle_df["normalized amplitude in xy-plane"].values
    theta_vect = particle_df["angle in xy-plane [deg]"].values * np.pi / 180  # [rad]

    A1_in_sigma = r_vect * np.cos(theta_vect)
    A2_in_sigma = r_vect * np.sin(theta_vect)

    particles = collider[beam].build_particles(
        x_norm=A1_in_sigma,
        y_norm=A2_in_sigma,
        delta=config_sim["delta_max"],
        scale_with_transverse_norm_emitt=(config_bb["nemitt_x"], config_bb["nemitt_y"]),
    )
    particles.particle_id = particle_df.particle_id.values

    return particles


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
# --- Function to do the tracking
# ==================================================================================================
def track(collider, particles, config_sim, config_bb=None, save_input_particles=False):
    # Get beam being tracked
    beam_track = config_sim["beam"]

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
    initial_sep_1 = collider.vars["on_sep1"]._value/20
    initial_sep_5 = collider.vars["on_sep5"]._value/20
    num_turns_step = int(num_turns / (n_steps + 1))
    sep_1_step = initial_sep_1 / n_steps
    sep_5_step = initial_sep_5 / n_steps

    for i in range(n_steps + 1):
        # Update separation and reconfigure beambeam
        collider.vars["on_sep1"] = initial_sep_1 - i * sep_1_step
        collider.vars["on_sep5"] = initial_sep_5 - i * sep_5_step
        print(
            f"Updating on_sep1 to {collider.vars['on_sep1']._value} on_sep5 to"
            f" {collider.vars['on_sep5']._value}"
        )

        if config_bb is not None:
            collider = configure_beam_beam(collider, config_bb)

        collider[beam_track].track(particles, turn_by_turn_monitor=False, num_turns=num_turns_step)
    b = time.time()
    print(f"Elapsed time: {b-a} s")
    print(f"Elapsed time per particle per turn: {(b-a)/particles._capacity/num_turns*1e6} us")

    return particles


# ==================================================================================================
# --- Main function for collider configuration and tracking
# ==================================================================================================
def configure_and_track(config_path="config.yaml"):
    # Get configuration
    config = read_configuration(config_path)

    # Tag start of the job
    tree_maker_tagging(config, tag="started")

    # Configure collider (not saved, since it may trigger overload of afs)
    collider, config_sim, config_bb = configure_collider(config)

    # Prepare particle distribution
    particles = prepare_particle_distribution(config_sim, collider, config_bb)

    # Track
    particles = track(collider, particles, config_sim, config_bb)

    # Save output
    pd.DataFrame(particles.to_dict()).to_parquet("output_particles.parquet")

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
