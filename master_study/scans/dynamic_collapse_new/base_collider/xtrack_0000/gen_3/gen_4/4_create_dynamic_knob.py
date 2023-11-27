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
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

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
    def interp(x, l_xrange, l_yrange):
        assert isinstance(x, float) or isinstance(x, int)
        for x1, x2, y1, y2 in zip(l_xrange[:-1], l_xrange[1:], l_yrange[:-1], l_yrange[1:]):
            if x1 <= x <= x2:
                return y1 + (y2 - y1) * (x - x1) / (x2 - x1)

    def make_linear_interp(model):
        return lambda x: interp(x, model["l_xrange"], model["attr"])

    d_element_attr_regression = {"lhcb1": {}, "lhcb2": {}}
    for beam in d_element_attr_regression:
        d_element_attr_regression[beam] = {}
        for element in d_element_attr_vals[beam]:
            d_element_attr_regression[beam][element] = {}
            for attr in d_element_attr_vals[beam][element]:
                model = {"l_xrange": l_xrange, "attr": d_element_attr_vals[beam][element][attr]}
                d_element_attr_regression[beam][element][attr]["fit"] = make_linear_interp(model)

    return d_element_attr_regression


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
                l_coef = d_element_attr_regression[beam][element][attr]["coeffs"]
                if isinstance(getattr(collider[beam][element], attr), list) or isinstance(
                    getattr(collider[beam][element], attr), np.ndarray
                ):
                    setattr(
                        collider[beam].element_refs[element],
                        attr[0],
                        sum([coef * collider.vars[sep] ** i for i, coef in enumerate(l_coef)]),
                    )
                else:
                    setattr(
                        collider[beam].element_refs[element],
                        attr,
                        sum([coef * collider.vars[sep] ** i for i, coef in enumerate(l_coef)]),
                    )

    return collider


def interpolate_separation(collider, dic_elements):

    # Get knob values as sep evolves
    d_element_attr_vals, l_xrange = build_dic_element_values(dic_elements)

    # Get dictionnary of regression
    d_element_attr_regression = linear_regression_bb_values(l_xrange, d_element_attr_vals)

    # Create knob for separation
    collider = create_knob_sep(collider, d_element_attr_regression)

    return collider, d_element_attr_regression


# ==================================================================================================
# --- Main function for collider configuration
# ==================================================================================================
def configure_collider(config):
    config_sim = config["config_simulation"]

    # Rebuild collider
    collider = xt.Multiline.from_json(config_sim["collider_file"])

    # Build trackers
    collider.build_trackers()

    # load dictionnary of elements from pickle
    with open(config_sim["elements_file"], "rb") as fid:
        dic_elements = pickle.load(fid)

    # Build knob
    collider, d_element_attr_regression = interpolate_separation(collider, dic_elements)

    return collider, d_element_attr_regression


# ==================================================================================================
# --- Main function for collider configuration and tracking
# ==================================================================================================
def configure_and_track(config_path="config.yaml"):
    # Get configuration
    config = read_configuration(config_path)

    # Tag start of the job
    tree_maker_tagging(config, tag="started")

    # Configure collider (not saved, since it may trigger overload of afs)
    collider, d_element_attr_regression = configure_collider(config)

    # Dump collider
    collider.to_json("collider.json")

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
