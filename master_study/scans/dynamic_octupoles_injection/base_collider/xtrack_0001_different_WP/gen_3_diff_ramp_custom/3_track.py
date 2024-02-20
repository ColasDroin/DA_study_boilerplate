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
    context = xo.ContextCupy(device="2")

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
def sample(
    collider,
    beam_track,
    particles,
    nemitt_x=None,
    nemitt_y=None,
):
    # Twiss to get normalized coordinates (temporarily disable time dependent variables)
    collider[beam_track].enable_time_dependent_vars = False
    tw = collider[beam_track].twiss()
    norm_coord = tw.get_normalized_coordinates(particles, nemitt_x=nemitt_x, nemitt_y=nemitt_y)
    collider[beam_track].enable_time_dependent_vars = True

    # Get (alive) particles coordinates
    particles_state = particles.state.get()
    particles_id = particles.particle_id.get()
    particles_x = particles.x.get()
    particles_px = particles.px.get()
    particles_y = particles.y.get()
    particles_py = particles.py.get()
    particles_zeta = particles.zeta.get()
    particles_pzeta = particles.pzeta.get()

    # Get normalized coordinates
    particles_id_norm = norm_coord.particle_id
    particles_x_norm = norm_coord.x_norm
    particles_px_norm = norm_coord.px_norm
    particles_y_norm = norm_coord.y_norm
    particles_py_norm = norm_coord.py_norm
    particles_zeta_norm = norm_coord.zeta_norm
    particles_pzeta_norm = norm_coord.pzeta_norm

    # Store everything in a dataframe
    df_particles = pd.DataFrame(
        {
            "particle_id": particles_id,
            "x": particles_x,
            "px": particles_px,
            "y": particles_y,
            "py": particles_py,
            "zeta": particles_zeta,
            "pzeta": particles_pzeta,
            "particle_id_norm": particles_id_norm,
            "x_norm": particles_x_norm,
            "px_norm": particles_px_norm,
            "y_norm": particles_y_norm,
            "py_norm": particles_py_norm,
            "zeta_norm": particles_zeta_norm,
            "pzeta_norm": particles_pzeta_norm,
            "state": particles_state,
        }
    )

    return df_particles


def track_sampled(
    collider,
    beam_track,
    particles,
    n_turns,
    freq,
    l_df_particles=[],
    l_n_turns=[],
    l_oct=[],
    nemitt_x=None,
    nemitt_y=None,
):
    for i in range(n_turns // freq):
        collider[beam_track].track(particles, turn_by_turn_monitor=False, num_turns=freq)
        l_df_particles.append(
            sample(
                collider,
                beam_track,
                particles,
                nemitt_x=nemitt_x,
                nemitt_y=nemitt_y,
            )
        )
        l_oct.append(collider.vars["i_oct_b1"]._value)
        if len(l_n_turns) == 0:
            l_n_turns.append(freq)
        else:
            l_n_turns.append(l_n_turns[-1] + freq)
    return l_df_particles, l_n_turns, l_oct


def track(collider, particles, config_sim, config_bb, save_input_particles=True):
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

    # Zero octupoles
    collider.vars["i_oct_b1"] = 0
    collider.vars["i_oct_b2"] = 0

    # Get initial particles distribution
    df_particles = sample(
        collider,
        beam_track,
        particles,
        nemitt_x=config_bb["nemitt_x"],
        nemitt_y=config_bb["nemitt_y"],
    )

    # Set initial values
    l_oct = [0]
    l_n_turns = [0]
    l_df_particles = [df_particles]
    collider.lhcb1.enable_time_dependent_vars = True

    # Sample every 1000 turns
    n_turns_init = 50000
    print(f"Start to track initial {n_turns_init} turns")
    freq_sampling = 1000
    l_df_particles, l_n_turns, l_oct = track_sampled(
        collider,
        beam_track,
        particles,
        n_turns_init,
        freq_sampling,
        l_df_particles=l_df_particles,
        l_n_turns=l_n_turns,
        l_oct=l_oct,
        nemitt_x=config_bb["nemitt_x"],
        nemitt_y=config_bb["nemitt_y"],
    )
    print(f"t_turn_s after {n_turns_init} = ", collider.lhcb1.vars["t_turn_s"]._value)

    # Then progressively increase/decrease the octupoles
    time_to_target = 88.18  # s
    f_LHC = 11247.2428926  # Hz
    n_turns = int(round(f_LHC * time_to_target))

    # Get t_turn_s_init
    def generate_smooth_ramp_signal():
        dt = 0.01  # time step
        t = 0.0

        # Initialize signals
        SI = 0.0
        dSI_dt = 0.0
        d2SI_dt2 = 0.1

        # List to store the signal values
        l_SI = [SI]
        l_dSI_dt = [dSI_dt]
        l_d2SI_dt2 = [d2SI_dt2]
        l_t = [t]

        # Integrate dIdt2 until ignal reaches 25:
        while SI < 25:
            SI += dSI_dt * dt
            dSI_dt += d2SI_dt2 * dt
            t += dt

            l_SI.append(SI)
            l_dSI_dt.append(dSI_dt)
            l_d2SI_dt2.append(d2SI_dt2)
            l_t.append(t)

        print("End of first integration:", SI, dSI_dt, d2SI_dt2, t)

        # Keep integrating reversing second dev until SI reaches 50, and back to 25:
        d2SI_dt2 = -0.1
        while SI > 25:
            SI += dSI_dt * dt
            dSI_dt += d2SI_dt2 * dt
            t += dt

            l_SI.append(SI)
            l_dSI_dt.append(dSI_dt)
            l_d2SI_dt2.append(d2SI_dt2)
            l_t.append(t)

        # Now reverse the trend one last time
        d2SI_dt2 = +0.1
        while SI > 0 + 1e-6:
            SI += dSI_dt * dt
            dSI_dt += d2SI_dt2 * dt
            t += dt

            l_SI.append(SI)
            l_dSI_dt.append(dSI_dt)
            l_d2SI_dt2.append(d2SI_dt2)
            l_t.append(t)

        return l_t, l_SI, l_dSI_dt, l_d2SI_dt2

    # Get the generated signal
    array_time, signal, first_derivative, second_derivative = generate_smooth_ramp_signal()
    collider.lhcb1.functions["fun_oct"] = xt.FunctionPieceWiseLinear(array_time, signal)
    collider.vars["i_oct_b1"] = collider.lhcb1.functions["fun_oct"](collider.lhcb1.vars["t_turn_s"])
    collider.vars["i_oct_b2"] = collider.lhcb1.functions["fun_oct"](collider.lhcb1.vars["t_turn_s"])

    # Track
    print("Start to track raising octupoles")
    print("Octupoles: ", collider.vars["i_oct_b1"]._value, collider.vars["i_oct_b2"]._value)
    print("t_turn_s = ", collider.lhcb1.vars["t_turn_s"]._value)
    l_df_particles, l_n_turns, l_oct = track_sampled(
        collider,
        beam_track,
        particles,
        n_turns,
        freq_sampling,
        l_df_particles=l_df_particles,
        l_n_turns=l_n_turns,
        l_oct=l_oct,
        nemitt_x=config_bb["nemitt_x"],
        nemitt_y=config_bb["nemitt_y"],
    )

    print("t_turn_s = ", collider.lhcb1.vars["t_turn_s"]._value)
    print("Octupoles: ", collider.vars["i_oct_b1"]._value, collider.vars["i_oct_b2"]._value)
    # Reset octupoles
    collider.vars["i_oct_b1"] = 0
    collider.vars["i_oct_b2"] = 0
    print(f"Start to track last {n_turns_init} turns")
    l_df_particles, l_n_turns, l_oct = track_sampled(
        collider,
        beam_track,
        particles,
        n_turns_init,
        freq_sampling,
        l_df_particles=l_df_particles,
        l_n_turns=l_n_turns,
        l_oct=l_oct,
        nemitt_x=config_bb["nemitt_x"],
        nemitt_y=config_bb["nemitt_y"],
    )
    b = time.time()
    print(f"Elapsed time: {b-a} s")
    print(f"Elapsed time per particle per turn: {(b-a)/particles._capacity/num_turns*1e6} us")

    # Create a dictionnary with all important observables
    d_observables = {
        "l_df_particles": l_df_particles,
        "l_n_turns": l_n_turns,
        "l_oct": l_oct,
    }

    return particles, d_observables


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
    particles, d_observables = track(collider, particles, config_sim, config_bb)

    # Save output
    pd.DataFrame(particles.to_dict()).to_parquet("output_particles.parquet")

    # Save observables as pickle
    with open("observables.pkl", "wb") as fid:
        pickle.dump(d_observables, fid)

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
