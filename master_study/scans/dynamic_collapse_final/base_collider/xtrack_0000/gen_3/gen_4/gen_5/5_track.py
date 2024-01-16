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
import xdeps as xd
import xmask as xm
import xobjects as xo
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


def create_knob_sep(collider, d_element_attr_regression):
    # Create knob for beam-beam in collider
    for beam in d_element_attr_regression:
        for element in d_element_attr_regression[beam]:
            if "l1" or "r1" in element:
                sep = "on_sep1"
            elif "l5" or "r5" in element:
                sep = "on_sep5"
            else:
                continue
            for attr in d_element_attr_regression[beam][element]:
                collider[beam].functions[f"interp_{element}_{attr}"] = d_element_attr_regression[
                    beam
                ][element][attr]
                if isinstance(getattr(collider[beam][element], attr), list) or isinstance(
                    getattr(collider[beam][element], attr), np.ndarray
                ):
                    setattr(
                        collider[beam].element_refs[element],
                        attr[0],
                        collider[beam].functions[f"interp_{element}_{attr}"](collider.vars[sep]),
                    )
                else:
                    setattr(
                        collider[beam].element_refs[element],
                        attr,
                        collider[beam].functions[f"interp_{element}_{attr}"](collider.vars[sep]),
                    )

    return collider


# ==================================================================================================
# --- Main function for collider configuration
# ==================================================================================================
def configure_collider(config):

    # Get configurations
    config_sim = config["config_simulation"]
    config_bb = config["config_collider"]["config_beambeam"]

    # Rebuild collider
    collider = xt.Multiline.from_json(config_sim["collider_file"])

    # Load dictionnary of regressions
    with open(config_sim["regression_file"], "rb") as fid:
        d_element_attr_regression = pickle.load(fid)

    # Create knob sep
    collider = create_knob_sep(collider, d_element_attr_regression)

    # Build trackers on GPU
    context = xo.ContextPyopencl()
    # context = xo.ContextCpu()
    collider.build_trackers(_context=context)

    return collider, config_sim, config_bb, context


# ==================================================================================================
# --- Function to prepare particles distribution for tracking
# ==================================================================================================
def prepare_particle_distribution(config_sim, collider, config_bb, context):
    beam = config_sim["beam"]

    particle_df = pd.read_parquet(config_sim["particle_file"])

    r_vect = particle_df["normalized amplitude in xy-plane"].values
    theta_vect = particle_df["angle in xy-plane [deg]"].values * np.pi / 180  # [rad]

    A1_in_sigma = r_vect * np.cos(theta_vect)
    A2_in_sigma = r_vect * np.sin(theta_vect)

    particles = collider[beam].build_particles(
        _context=context,
        x_norm=A1_in_sigma,
        y_norm=A2_in_sigma,
        delta=config_sim["delta_max"],
        scale_with_transverse_norm_emitt=(config_bb["nemitt_x"], config_bb["nemitt_y"]),
    )
    particle_id = particle_df.particle_id.values.astype(np.int32, copy=True)

    return particles, particle_id


# ==================================================================================================
# --- Function to do the tracking
# ==================================================================================================
def track(collider, particles, config_sim, save_input_particles=False):
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
    initial_sep_1 = collider.vars["on_sep1"]._value / 50  # ! REMOVE /50
    initial_sep_5 = collider.vars["on_sep5"]._value / 50  # ! REMOVE /50

    # Define time-dependant closing
    collider.lhcb1.enable_time_dependent_vars = True
    time_separation = 10  # s # ! 90
    f_LHC = 11247.2428926  # Hz
    n_turns = int(round(f_LHC * time_separation))
    print("n_turns = ", n_turns)
    f_sep_1 = 0  # initial_sep_1 / time_separation # ! UNCOMMENT
    f_sep_5 = 0  # initial_sep_5 / time_separation # ! UNCOMMENT
    collider.vars["on_sep1"] = initial_sep_1 - collider.vars["t_turn_s"] * f_sep_1
    collider.vars["on_sep5"] = initial_sep_5 - collider.vars["t_turn_s"] * f_sep_5
    # Track
    print("t_turn_s = ", collider.lhcb1.vars["t_turn_s"]._value)
    collider[beam_track].track(particles, turn_by_turn_monitor=False, num_turns=n_turns)
    print("t_turn_s = ", collider.lhcb1.vars["t_turn_s"]._value)
    print(collider.vars["on_sep1"]._info())

    # Track for N more turns at the end
    # collider.vars["on_sep1"] = 0
    # collider.vars["on_sep5"] = 0
    collider.lhcb1.enable_time_dependent_vars = False
    # N = 50000
    # collider[beam_track].track(particles, turn_by_turn_monitor=False, num_turns=N)
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
    collider, config_sim, config_bb, context = configure_collider(config)

    # Prepare particle distribution
    particles, particle_id = prepare_particle_distribution(config_sim, collider, config_bb, context)

    # Track
    particles = track(collider, particles, config_sim)

    # Save output
    particles_dict = particles.to_dict()
    particles_dict["particle_id"] = particle_id
    pd.DataFrame(particles_dict).to_parquet("output_particles.parquet")

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
