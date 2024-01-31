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

import numpy as np
import pandas as pd
import ruamel.yaml
import tree_maker
import xobjects as xo
import xpart as xp
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
    config_sim = config["config_simulation"]

    # Read config from previous generation
    with open("../" + config_path, "r") as fid:
        config_bb = ryaml.load(fid)["config_collider"]["config_beambeam"]

    return config, config_sim, config_bb


# ==================================================================================================
# --- Main function for collider configuration
# ==================================================================================================
def configure_collider(config_sim):

    # Rebuild collider
    collider = xt.Multiline.from_json(config_sim["collider_file"])

    # Build trackers on GPU
    context = xo.ContextPyopencl()
    # context = xo.ContextCpu()
    collider.build_trackers(_context=context)

    return collider


# ==================================================================================================
# --- Function to prepare particles distribution for tracking
# ==================================================================================================
def prepare_particle_distribution(config_sim, collider, config_bb):
    n_part = 20000
    particles = xp.generate_matched_gaussian_bunch(
        num_particles=n_part,
        total_intensity_particles=config_bb["num_particles_per_bunch"],
        nemitt_x=config_bb["nemitt_x"],
        nemitt_y=config_bb["nemitt_y"],
        sigma_z=config_bb["sigma_z"],
        line=collider.lhcb1,
    )

    return particles


# ==================================================================================================
# --- Function to do the tracking
# ==================================================================================================
def track_sampled(
    collider,
    beam_track,
    particles,
    n_turns,
    freq,
    l_emittance_x=[],
    l_emittance_y=[],
    l_oct=[],
    l_n_turns=[],
    ll_particles_x=[],
    ll_particles_px=[],
    ll_particles_y=[],
    ll_particles_py=[],
):
    for i in range(n_turns // freq):
        collider[beam_track].track(particles, turn_by_turn_monitor=False, num_turns=freq)
        particles_state = particles.state.get()
        particles_x = particles.x.get()[particles_state > 0]
        particles_px = particles.px.get()[particles_state > 0]
        particles_y = particles.y.get()[particles_state > 0]
        particles_py = particles.py.get()[particles_state > 0]
        emittance_x = np.sqrt(
            np.mean(particles_x**2) * np.mean(particles_px**2)
            - np.mean(particles_x * particles_px) ** 2
        )
        emittance_y = np.sqrt(
            np.mean(particles_y**2) * np.mean(particles_py**2)
            - np.mean(particles_y * particles_py) ** 2
        )

        ll_particles_x.append(list(particles_x))
        ll_particles_px.append(list(particles_px))
        ll_particles_y.append(list(particles_y))
        ll_particles_py.append(list(particles_py))
        l_emittance_x.append(emittance_x)
        l_emittance_y.append(emittance_y)
        l_oct.append(collider.vars["i_oct_b1"]._value)
        if len(l_n_turns) == 0:
            l_n_turns.append(freq)
        else:
            l_n_turns.append(l_n_turns[-1] + freq)
    return (
        ll_particles_x,
        ll_particles_px,
        ll_particles_y,
        ll_particles_py,
        l_emittance_x,
        l_emittance_y,
        l_oct,
        l_n_turns,
    )


def track(collider, particles, config_sim, save_input_particles=True):
    # Get beam being tracked
    beam_track = config_sim["beam"]

    # Optimize line for tracking # ! Commented out as it prevents changing the knobs
    # collider[beam_track].optimize_for_tracking()

    # Save initial coordinates if requested
    if save_input_particles:
        pd.DataFrame(particles.to_dict()).to_parquet("input_particles.parquet")

    # Track (update bb in several steps)
    num_turns = config_sim["n_turns"]
    a = time.time()

    # Start to track for 5000 turns with zero octupoles
    print("Start to track initial 5000 turns")
    collider.vars["i_oct_b1"] = 0
    collider.vars["i_oct_b2"] = 0

    # Set initial values
    l_oct = [0]

    particles_x = particles.x.get()
    particles_px = particles.px.get()
    particles_y = particles.y.get()
    particles_py = particles.py.get()
    emittance_x = np.sqrt(
        np.mean(particles_x**2) * np.mean(particles_px**2)
        - np.mean(particles_x * particles_px) ** 2
    )
    emittance_y = np.sqrt(
        np.mean(particles_y**2) * np.mean(particles_py**2)
        - np.mean(particles_y * particles_py) ** 2
    )
    l_emittance_x = [emittance_x]
    l_emittance_y = [emittance_y]
    l_n_turns = [0]
    ll_particles_x = [list(particles_x)]
    ll_particles_px = [list(particles_px)]
    ll_particles_y = [list(particles_y)]
    ll_particles_py = [list(particles_py)]

    # Get emittance every 1000 turns
    n_turns_init = 20000
    freq_emittance = 1000
    (
        ll_particles_x,
        ll_particles_px,
        ll_particles_y,
        ll_particles_py,
        l_emittance_x,
        l_emittance_y,
        l_oct,
        l_n_turns,
    ) = track_sampled(
        collider,
        beam_track,
        particles,
        n_turns_init,
        freq_emittance,
        l_emittance_x=l_emittance_x,
        l_emittance_y=l_emittance_y,
        l_oct=l_oct,
        l_n_turns=l_n_turns,
        ll_particles_x=ll_particles_x,
        ll_particles_px=ll_particles_px,
        ll_particles_y=ll_particles_y,
        ll_particles_py=ll_particles_py,
    )

    # Reset number of turns
    print(f"t_turn_s after {n_turns_init} = ", collider.lhcb1.vars["t_turn_s"]._value)
    collider.lhcb1.vars["t_turn_s"] = 0
    print("t_turn_s after reset = ", collider.lhcb1.vars["t_turn_s"]._value)

    # Then progressively increase the octupoles
    target_oct = 50
    collider.lhcb1.enable_time_dependent_vars = True
    time_to_target = 80  # s
    f_LHC = 11247.2428926  # Hz
    n_turns = int(round(f_LHC * time_to_target))
    f_sep_1 = target_oct / time_to_target
    f_sep_5 = target_oct / time_to_target
    collider.vars["i_oct_b1"] = 0 + collider.vars["t_turn_s"] * f_sep_1
    collider.vars["i_oct_b2"] = 0 + collider.vars["t_turn_s"] * f_sep_5
    # Track
    print("Start to track raising octupoles")
    print("t_turn_s = ", collider.lhcb1.vars["t_turn_s"]._value)
    (
        ll_particles_x,
        ll_particles_px,
        ll_particles_y,
        ll_particles_py,
        l_emittance_x,
        l_emittance_y,
        l_oct,
        l_n_turns,
    ) = track_sampled(
        collider,
        beam_track,
        particles,
        n_turns,
        freq_emittance,
        l_emittance_x=l_emittance_x,
        l_emittance_y=l_emittance_y,
        l_oct=l_oct,
        l_n_turns=l_n_turns,
        ll_particles_x=ll_particles_x,
        ll_particles_px=ll_particles_px,
        ll_particles_y=ll_particles_y,
        ll_particles_py=ll_particles_py,
    )

    print("t_turn_s = ", collider.lhcb1.vars["t_turn_s"]._value)

    # Reset number of turns
    collider.vars["t_turn_s"] = 0
    print("t_turn_s after reset = ", collider.lhcb1.vars["t_turn_s"]._value)

    # Then progressively decrease the octupoles
    collider.vars["i_oct_b1"] = target_oct - collider.vars["t_turn_s"] * f_sep_1
    collider.vars["i_oct_b2"] = target_oct - collider.vars["t_turn_s"] * f_sep_5
    print("Start to track decreasing octupoles octupoles")
    (
        ll_particles_x,
        ll_particles_px,
        ll_particles_y,
        ll_particles_py,
        l_emittance_x,
        l_emittance_y,
        l_oct,
        l_n_turns,
    ) = track_sampled(
        collider,
        beam_track,
        particles,
        n_turns,
        freq_emittance,
        l_emittance_x=l_emittance_x,
        l_emittance_y=l_emittance_y,
        l_oct=l_oct,
        l_n_turns=l_n_turns,
        ll_particles_x=ll_particles_x,
        ll_particles_px=ll_particles_px,
        ll_particles_y=ll_particles_y,
        ll_particles_py=ll_particles_py,
    )

    print(f"Start to track last {n_turns_init} turns")
    (
        ll_particles_x,
        ll_particles_px,
        ll_particles_y,
        ll_particles_py,
        l_emittance_x,
        l_emittance_y,
        l_oct,
        l_n_turns,
    ) = track_sampled(
        collider,
        beam_track,
        particles,
        n_turns,
        freq_emittance,
        l_emittance_x=l_emittance_x,
        l_emittance_y=l_emittance_y,
        l_oct=l_oct,
        l_n_turns=l_n_turns,
        ll_particles_x=ll_particles_x,
        ll_particles_px=ll_particles_px,
        ll_particles_y=ll_particles_y,
        ll_particles_py=ll_particles_py,
    )

    b = time.time()
    print(f"Elapsed time: {b-a} s")
    print(f"Elapsed time per particle per turn: {(b-a)/particles._capacity/num_turns*1e6} us")

    # Create dataframe containing the emittance and octupoles
    df_emittance = pd.DataFrame(
        {
            "particles_x": ll_particles_x,
            "particles_px": ll_particles_px,
            "particles_y": ll_particles_y,
            "particles_py": ll_particles_py,
            "emittance_x": l_emittance_x,
            "emittance_y": l_emittance_y,
            "octupoles": l_oct,
            "n_turns": l_n_turns,
        }
    )

    return particles, df_emittance


# ==================================================================================================
# --- Main function for collider configuration and tracking
# ==================================================================================================
def configure_and_track(config_path="config.yaml"):
    # Get configuration
    config, config_sim, config_bb = read_configuration(config_path)

    # Tag start of the job
    tree_maker_tagging(config, tag="started")

    # Configure collider (not saved, since it may trigger overload of afs)
    collider = configure_collider(config_sim)

    # Prepare particle distribution
    particles = prepare_particle_distribution(config_sim, collider, config_bb)

    # Track
    particles, df_emittance = track(collider, particles, config_sim)

    # Save output
    pd.DataFrame(particles.to_dict()).to_parquet("output_particles.parquet")
    df_emittance.to_parquet("output_emittance.parquet")

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
