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

import dill as pickle
import numpy as np
import ruamel.yaml
import tree_maker
import xdeps as xd
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


def build_dic_element_values(dic_elements):
    l_xrange = []
    d_element_attr_vals = {"lhcb1": {}, "lhcb2": {}}
    for on_sep, d_beam in sorted(dic_elements.items()):
        l_xrange.append(on_sep)
        for beam_temp, d_bb in d_beam.items():
            for type_bb, d_elements in d_bb.items():
                for element, d_attr in d_elements.items():
                    if element not in d_element_attr_vals[beam_temp]:
                        d_element_attr_vals[beam_temp][element] = {}

                    for attr, val in d_attr.items():
                        if attr in d_element_attr_vals[beam_temp][element]:
                            d_element_attr_vals[beam_temp][element][attr].append(val)
                        else:
                            d_element_attr_vals[beam_temp][element][attr] = [val]
    return d_element_attr_vals, l_xrange


def linear_regression_bb_values(l_xrange, d_element_attr_vals):
    def make_closure_interp(extended_xrange, extended_y):
        return lambda x: xd.FunctionPieceWiseLinear(x=extended_xrange, y=extended_y)(x)

    extended_xrange = list(l_xrange) + [-x for x in l_xrange[::-1]]
    d_element_attr_regression = {"lhcb1": {}, "lhcb2": {}}
    for beam in d_element_attr_regression:
        d_element_attr_regression[beam] = {}
        for element in d_element_attr_vals[beam]:
            d_element_attr_regression[beam][element] = {}
            for attr in d_element_attr_vals[beam][element]:
                y = list(np.squeeze(np.array(d_element_attr_vals[beam][element][attr])))
                extended_y = y + y[::-1]
                d_element_attr_regression[beam][element][attr] = make_closure_interp(
                    extended_xrange, extended_y
                )

    return d_element_attr_regression


def interpolate_separation(dic_elements):

    # Get knob values as sep evolves
    d_element_attr_vals, l_xrange = build_dic_element_values(dic_elements)

    # Get dictionnary of regression
    d_element_attr_regression = linear_regression_bb_values(l_xrange, d_element_attr_vals)

    return d_element_attr_regression


# ==================================================================================================
# --- Main function for collider configuration and tracking
# ==================================================================================================
def configure_and_track(config_path="config.yaml"):
    # Get configuration
    config = read_configuration(config_path)

    # Tag start of the job
    tree_maker_tagging(config, tag="started")

    # load dictionnary of elements from pickle
    with open(config["elements_file"], "rb") as fid:
        dic_elements = pickle.load(fid)

    # Build dictionnary of regression
    d_element_attr_regression = interpolate_separation(dic_elements)

    # Dump dictionnary of regression
    with open("d_element_attr_regression.pkl", "wb") as fid:
        pickle.dump(d_element_attr_regression, fid)

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
