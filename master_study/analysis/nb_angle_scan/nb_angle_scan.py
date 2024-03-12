# %% [markdown]
# ### Import modules

# %%
# Standard imports
import json
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xtrack as xt
from scipy.ndimage import gaussian_filter
from skimage.measure import find_contours

# Local imports
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import analysis_functions

# Apply better style
analysis_functions.apply_heatmap_style()


# %% [markdown]
# ### 30 cm
#

# %%
# Define study
STUDY_NAME = "PU_function_all_optics"
COLLIDER = "collider_00"
PARQUET_PATH = f"../../scans/{STUDY_NAME}/da.parquet"
CONF_MAD_PATH = f"../../scans/{STUDY_NAME}/{COLLIDER}/config.yaml"
CONF_COLLIDER_PATH = f"../../scans/{STUDY_NAME}/{COLLIDER}/xtrack_0016/config.yaml"

# SAve study on EOS
SAVE_STUDY_EOS = False
# Load dataframe
df = pd.read_parquet(f"../../scans/{STUDY_NAME}/da.parquet")

# Round all numbers to 3 decimals
df = df.round(3)


df

# %%
# Keep only relevant collider
df = df.reset_index(level=1)
df = df[df["name base collider"] == COLLIDER]

# Reshape for plotting
df_to_plot = df.pivot(
    index="num_particles_per_bunch",
    columns="crossing_angle",
    values="normalized amplitude in xy-plane",
)
df_to_plot.index /= 1e11
df_to_plot


# %%
# Load configuration files
conf_mad = analysis_functions.load_config(CONF_MAD_PATH)["config_mad"]
conf_collider = analysis_functions.load_config(CONF_COLLIDER_PATH)["config_collider"]
analysis_functions.plot_heatmap(
    df_to_plot,
    STUDY_NAME + "_" + COLLIDER,
    link=None,
    plot_contours=True,
    conf_mad=conf_mad,
    conf_collider=conf_collider,
    type_crossing="flatvh",
    betx=0.3,
    bety=0.3,
    Nb=False,
    green_contour=5.5,
    extended_diagonal=False,
    symmetric=False,
    mask_lower_triangle=False,
    xlabel="Crossing angle [urad]",
    ylabel="Number of particles per bunch (1e11)",
    vmin=4.5,
    vmax=7.0,
    plot_diagonal_lines=False,
    xaxis_ticks_on_top=False,
    title="DA for Bunch charge vs crossing angle (2024 EoL 30cm optics)",
)


# %%
df_30_cm = df_to_plot.copy()

# %% [markdown]
# ### 26cm

# %%
# Define study
STUDY_NAME = "PU_function_all_optics"
COLLIDER = "collider_01"
PARQUET_PATH = f"../../scans/{STUDY_NAME}/da.parquet"
CONF_MAD_PATH = f"../../scans/{STUDY_NAME}/{COLLIDER}/config.yaml"
CONF_COLLIDER_PATH = f"../../scans/{STUDY_NAME}/{COLLIDER}/xtrack_0016/config.yaml"

# SAve study on EOS
SAVE_STUDY_EOS = False
# Load dataframe
df = pd.read_parquet(f"../../scans/{STUDY_NAME}/da.parquet")

# Round all numbers to 3 decimals
df = df.round(3)

# Keep only relevant collider
df = df.reset_index(level=1)
df = df[df["name base collider"] == COLLIDER]

# Reshape for plotting
df_to_plot = df.pivot(
    index="num_particles_per_bunch",
    columns="crossing_angle",
    values="normalized amplitude in xy-plane",
)
df_to_plot.index /= 1e11
# df_to_plot

# Load configuration files
conf_mad = analysis_functions.load_config(CONF_MAD_PATH)["config_mad"]
conf_collider = analysis_functions.load_config(CONF_COLLIDER_PATH)["config_collider"]
analysis_functions.plot_heatmap(
    df_to_plot,
    STUDY_NAME + "_" + COLLIDER,
    link=None,
    plot_contours=True,
    conf_mad=conf_mad,
    conf_collider=conf_collider,
    type_crossing="flatvh",
    betx=0.3,
    bety=0.3,
    Nb=False,
    green_contour=5.5,
    extended_diagonal=False,
    symmetric=False,
    mask_lower_triangle=False,
    xlabel="Crossing angle [urad]",
    ylabel="Number of particles per bunch (1e11)",
    vmin=4.5,
    vmax=7.0,
    plot_diagonal_lines=False,
    xaxis_ticks_on_top=False,
    title="DA for Bunch charge vs crossing angle (2024 EoL 26cm optics)",
)

# %%
df_26_cm = df_to_plot.copy()

# %% [markdown]
# ### 22 cm

# %%
# Define study
STUDY_NAME = "PU_function_all_optics"
COLLIDER = "collider_02"
PARQUET_PATH = f"../../scans/{STUDY_NAME}/da.parquet"
CONF_MAD_PATH = f"../../scans/{STUDY_NAME}/{COLLIDER}/config.yaml"
CONF_COLLIDER_PATH = f"../../scans/{STUDY_NAME}/{COLLIDER}/xtrack_0016/config.yaml"

# SAve study on EOS
SAVE_STUDY_EOS = False
# Load dataframe
df = pd.read_parquet(f"../../scans/{STUDY_NAME}/da.parquet")

# Round all numbers to 3 decimals
df = df.round(3)

# Keep only relevant collider
df = df.reset_index(level=1)
df = df[df["name base collider"] == COLLIDER]

# Reshape for plotting
df_to_plot = df.pivot(
    index="num_particles_per_bunch",
    columns="crossing_angle",
    values="normalized amplitude in xy-plane",
)
df_to_plot.index /= 1e11
# df_to_plot

# Load configuration files
conf_mad = analysis_functions.load_config(CONF_MAD_PATH)["config_mad"]
conf_collider = analysis_functions.load_config(CONF_COLLIDER_PATH)["config_collider"]
analysis_functions.plot_heatmap(
    df_to_plot,
    STUDY_NAME + "_" + COLLIDER,
    link=None,
    plot_contours=True,
    conf_mad=conf_mad,
    conf_collider=conf_collider,
    type_crossing="flatvh",
    betx=0.3,
    bety=0.3,
    Nb=False,
    green_contour=5.5,
    extended_diagonal=False,
    symmetric=False,
    mask_lower_triangle=False,
    xlabel="Crossing angle [urad]",
    ylabel="Number of particles per bunch (1e11)",
    vmin=4.5,
    vmax=7.0,
    plot_diagonal_lines=False,
    xaxis_ticks_on_top=False,
    title="DA for Bunch charge vs crossing angle (2024 EoL 22cm optics)",
)

# %%
df_22_cm = df_to_plot.copy()

# %% [markdown]
# ## Interpolate the data for each optics

# %%
import numpy as np
from scipy import interpolate

# %%
x = df_30_cm.index
y = df_30_cm.columns
array = np.ma.masked_invalid(df_30_cm.values.T)
xx, yy = np.meshgrid(x, y)
# get only the valid values
x1 = xx[~array.mask]
y1 = yy[~array.mask]
newarr = array[~array.mask]
interpolated_grid = interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy), method="cubic")
interpolate_df_30cm = pd.DataFrame(interpolated_grid.T).interpolate(axis=1)

# %%
conf_mad = analysis_functions.load_config(CONF_MAD_PATH)["config_mad"]
conf_collider = analysis_functions.load_config(CONF_COLLIDER_PATH)["config_collider"]
analysis_functions.plot_heatmap(
    interpolate_df_30cm,
    STUDY_NAME + "_" + COLLIDER + "_interpolated",
    link=None,
    plot_contours=True,
    conf_mad=conf_mad,
    conf_collider=conf_collider,
    type_crossing="flatvh",
    betx=0.3,
    bety=0.3,
    Nb=False,
    green_contour=5.5,
    extended_diagonal=False,
    symmetric=False,
    mask_lower_triangle=False,
    xlabel="Crossing angle [urad]",
    ylabel="Number of particles per bunch (1e11)",
    vmin=4.5,
    vmax=7.0,
    plot_diagonal_lines=False,
    xaxis_ticks_on_top=False,
    title="DA for Bunch charge vs crossing angle (2024 EoL 30cm optics)",
)

# %%
x = df_26_cm.index
y = df_26_cm.columns
array = np.ma.masked_invalid(df_26_cm.values.T)
xx, yy = np.meshgrid(x, y)
# get only the valid values
x1 = xx[~array.mask]
y1 = yy[~array.mask]
newarr = array[~array.mask]
interpolated_grid = interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy), method="cubic")
interpolate_df_26cm = pd.DataFrame(interpolated_grid.T).interpolate(axis=1)

conf_mad = analysis_functions.load_config(CONF_MAD_PATH)["config_mad"]
conf_collider = analysis_functions.load_config(CONF_COLLIDER_PATH)["config_collider"]
analysis_functions.plot_heatmap(
    interpolate_df_26cm,
    STUDY_NAME + "_" + COLLIDER + "_interpolated",
    link=None,
    plot_contours=True,
    conf_mad=conf_mad,
    conf_collider=conf_collider,
    type_crossing="flatvh",
    betx=0.3,
    bety=0.3,
    Nb=False,
    green_contour=5.5,
    extended_diagonal=False,
    symmetric=False,
    mask_lower_triangle=False,
    xlabel="Crossing angle [urad]",
    ylabel="Number of particles per bunch (1e11)",
    vmin=4.5,
    vmax=7.0,
    plot_diagonal_lines=False,
    xaxis_ticks_on_top=False,
    title="DA for Bunch charge vs crossing angle (2024 EoL 26cm optics)",
)

# %%
x = df_22_cm.index
y = df_22_cm.columns
array = np.ma.masked_invalid(df_22_cm.values.T)
xx, yy = np.meshgrid(x, y)
# get only the valid values
x1 = xx[~array.mask]
y1 = yy[~array.mask]
newarr = array[~array.mask]
interpolated_grid = interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy), method="cubic")
interpolate_df_22cm = pd.DataFrame(interpolated_grid.T).interpolate(axis=1)

conf_mad = analysis_functions.load_config(CONF_MAD_PATH)["config_mad"]
conf_collider = analysis_functions.load_config(CONF_COLLIDER_PATH)["config_collider"]
analysis_functions.plot_heatmap(
    interpolate_df_22cm,
    STUDY_NAME + "_" + COLLIDER + "_interpolated",
    link=None,
    plot_contours=True,
    conf_mad=conf_mad,
    conf_collider=conf_collider,
    type_crossing="flatvh",
    betx=0.3,
    bety=0.3,
    Nb=False,
    green_contour=5.5,
    extended_diagonal=False,
    symmetric=False,
    mask_lower_triangle=False,
    xlabel="Crossing angle [urad]",
    ylabel="Number of particles per bunch (1e11)",
    vmin=4.5,
    vmax=7.0,
    plot_diagonal_lines=False,
    xaxis_ticks_on_top=False,
    title="DA for Bunch charge vs crossing angle (2024 EoL 22cm optics)",
)

# %% [markdown]
# ### Find contours at 5.5 sigma

# %%
contours_30cm = find_contours(interpolate_df_30cm.values, level=5.5)

# %%
fig, ax = plt.subplots(1, 1)
img = ax.imshow(interpolate_df_30cm.values, origin="lower")
plt.colorbar(img)
for i in range(len(contours_30cm)):
    p = plt.Polygon(contours_30cm[i][:, [1, 0]], fill=False, color="w", closed=False)
    ax.add_artist(p)

plt.show()

# %%
# Smooth a lot to have a smooth contour
smooth_30cm = gaussian_filter(interpolate_df_30cm.values, sigma=1)
contours_30cm = find_contours(smooth_30cm, level=5.5)


# %%
fig, ax = plt.subplots(1, 1)
img = ax.imshow(interpolate_df_30cm.values, origin="lower")
plt.colorbar(img)
for i in range(len(contours_30cm)):
    p = plt.Polygon(contours_30cm[i][:, [1, 0]], fill=False, color="w", closed=False)
    ax.add_artist(p)
plt.show()

# %%
# Also get the same curve for 26cm and 22cm
smooth_26cm = gaussian_filter(interpolate_df_26cm.values, sigma=1)
contours_26cm = find_contours(smooth_26cm, level=5.5)

smooth_22cm = gaussian_filter(interpolate_df_22cm.values, sigma=1)
contours_22cm = find_contours(smooth_22cm, level=5.5)


# %%
# Apply better style
sns.set_theme(style="whitegrid")
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)
# sns.set(font='Adobe Devanagari')
sns.set_context("paper", font_scale=1, rc={"lines.linewidth": 0.5, "grid.linewidth": 0.3})


matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "STIXGeneral"
# Not italized latex
matplotlib.rcParams["mathtext.default"] = "regular"
matplotlib.rcParams["font.weight"] = "light"

# %%
# Extract curve
contour_for_extraction_30cm = contours_30cm[0][contours_30cm[0][:, 1].argsort()]
contour_for_extraction_26cm = contours_26cm[0][contours_26cm[0][:, 1].argsort()]
contour_for_extraction_22cm = contours_22cm[0][contours_22cm[0][:, 1].argsort()]

x_30cm = contour_for_extraction_30cm[:, 1]
y_30cm = contour_for_extraction_30cm[:, 0]

x_26cm = contour_for_extraction_26cm[:, 1]
y_26cm = contour_for_extraction_26cm[:, 0]

x_22cm = contour_for_extraction_22cm[:, 1]
y_22cm = contour_for_extraction_22cm[:, 0]

# Convert back to initial units
f1 = interpolate.interp1d(range(len(df_30_cm.index)), df_30_cm.index)
f2 = interpolate.interp1d(range(len(df_30_cm.columns)), df_30_cm.columns)

x_real_30cm = f2(x_30cm)
y_real_30cm = f1(y_30cm)

x_real_26cm = f2(x_26cm)
y_real_26cm = f1(y_26cm)

x_real_22cm = f2(x_22cm)
y_real_22cm = f1(y_22cm)

# Compute rescaled 26cm and 22cm
f3 = interpolate.interp1d(x_real_30cm, y_real_30cm, bounds_error=False, fill_value="extrapolate")
rescaled_y_22cm = f3(x_real_30cm * np.sqrt(22 / 30))
rescaled_y_26cm = f3(x_real_30cm * np.sqrt(26 / 30))

plt.plot(x_real_30cm, y_real_30cm, color="C0", label="30cm")

plt.plot(x_real_26cm, y_real_26cm, color="C2", label="26cm")
plt.plot(x_real_30cm, rescaled_y_26cm, "--", color="C2", label="26cm from scaling law")

plt.plot(x_real_22cm, y_real_22cm, color="C1", label="22cm")
plt.plot(x_real_30cm, rescaled_y_22cm, "--", color="C1", label="22cm from scaling law")
plt.xlabel("Crossing angle [urad]")
plt.ylabel("Bunch charge [1e11]")
plt.title("5.5 iso-DA curves")
plt.ylim(0.6, 1.3)
plt.legend()
plt.grid()
plt.savefig("iso_DA_5dot5.pdf")
plt.show()

# %% [markdown]
# ### Find contours at 5 sigma

# %%
# Smooth to have a smooth contour
contours_30cm_5sig = find_contours(smooth_30cm, level=5)
contours_26cm_5sig = find_contours(smooth_26cm, level=5)
contours_22cm_5sig = find_contours(smooth_22cm, level=5)

# Extract curve
contour_for_extraction_30cm_5sig = contours_30cm_5sig[0][contours_30cm_5sig[0][:, 1].argsort()]
contour_for_extraction_26cm_5sig = contours_26cm_5sig[0][contours_26cm_5sig[0][:, 1].argsort()]
contour_for_extraction_22cm_5sig = contours_22cm_5sig[0][contours_22cm_5sig[0][:, 1].argsort()]

x_30cm_5sig = contour_for_extraction_30cm_5sig[:, 1]
y_30cm_5sig = contour_for_extraction_30cm_5sig[:, 0]

x_26cm_5sig = contour_for_extraction_26cm_5sig[:, 1]
y_26cm_5sig = contour_for_extraction_26cm_5sig[:, 0]

x_22cm_5sig = contour_for_extraction_22cm_5sig[:, 1]
y_22cm_5sig = contour_for_extraction_22cm_5sig[:, 0]

# Convert back to initial units
f1 = interpolate.interp1d(range(len(df_30_cm.index)), df_30_cm.index)
f2 = interpolate.interp1d(range(len(df_30_cm.columns)), df_30_cm.columns)

x_real_30cm_5sig = f2(x_30cm_5sig)
y_real_30cm_5sig = f1(y_30cm_5sig)

x_real_26cm_5sig = f2(x_26cm_5sig)
y_real_26cm_5sig = f1(y_26cm_5sig)

x_real_22cm_5sig = f2(x_22cm_5sig)
y_real_22cm_5sig = f1(y_22cm_5sig)

# Compute rescaled 26cm and 22cm
f3 = interpolate.interp1d(
    x_real_30cm_5sig, y_real_30cm_5sig, bounds_error=False, fill_value="extrapolate"
)
rescaled_y_22cm_5sig = f3(x_real_30cm_5sig * np.sqrt(22 / 30))
rescaled_y_26cm_5sig = f3(x_real_30cm_5sig * np.sqrt(26 / 30))

plt.plot(x_real_30cm_5sig, y_real_30cm_5sig, color="C0", label="30cm")

plt.plot(x_real_26cm_5sig, y_real_26cm_5sig, color="C2", label="26cm")
plt.plot(x_real_30cm_5sig, rescaled_y_26cm_5sig, "--", color="C2", label="26cm from scaling law")

plt.plot(x_real_22cm_5sig, y_real_22cm_5sig, color="C1", label="22cm")
plt.plot(x_real_30cm_5sig, rescaled_y_22cm_5sig, "--", color="C1", label="22cm from scaling law")
plt.xlabel("Crossing angle [urad]")
plt.ylabel("Bunch charge [1e11]")
plt.title("5 iso-DA curves")
plt.ylim(0.6, 1.3)
plt.legend()
plt.grid()
plt.savefig("iso_DA_5.pdf")
plt.show()

# %% [markdown]
# ### Convert to PU function


# %%
# Extracted from master jobs
def compute_collision_from_scheme(config_bb):
    # Get the filling scheme path (in json or csv format)
    filling_scheme_path = config_bb["mask_with_filling_pattern"]["pattern_fname"]

    # Load the filling scheme
    if filling_scheme_path.endswith(".json"):
        with open(filling_scheme_path, "r") as fid:
            filling_scheme = json.load(fid)
    else:
        raise ValueError(
            f"Unknown filling scheme file format: {filling_scheme_path}. It you provided a csv"
            " file, it should have been automatically convert when running the script"
            " 001_make_folders.py. Something went wrong."
        )

    # Extract booleans beam arrays
    array_b1 = np.array(filling_scheme["beam1"])
    array_b2 = np.array(filling_scheme["beam2"])

    # Assert that the arrays have the required length, and do the convolution
    assert len(array_b1) == len(array_b2) == 3564
    n_collisions_ip1_and_5 = array_b1 @ array_b2
    n_collisions_ip2 = np.roll(array_b1, 891) @ array_b2
    n_collisions_ip8 = np.roll(array_b1, 2670) @ array_b2

    return n_collisions_ip1_and_5, n_collisions_ip2, n_collisions_ip8


def compute_lumi(
    bunch_charge, twiss_b1, twiss_b2, crab, nemitt_x, nemitt_y, sigma_z, num_colliding_bunches
):
    luminosity = xt.lumi.luminosity_from_twiss(
        n_colliding_bunches=num_colliding_bunches,
        num_particles_per_bunch=bunch_charge,
        ip_name="ip1",
        nemitt_x=nemitt_x,
        nemitt_y=nemitt_y,
        sigma_z=sigma_z,
        twiss_b1=twiss_b1,
        twiss_b2=twiss_b2,
        crab=crab,
    )
    return luminosity


def compute_PU(luminosity, num_colliding_bunches, T_rev0, cross_section=81e-27):
    return luminosity / num_colliding_bunches * cross_section * T_rev0


def compute_PU_from_lumi(collider, config_bb):
    twiss_b1 = collider.lhcb1.twiss()
    twiss_b2 = collider.lhcb2.twiss()
    crab = False
    nemitt_x = config_bb["nemitt_x"]
    nemitt_y = config_bb["nemitt_y"]
    sigma_z = config_bb["sigma_z"]
    n_collisions_ip1_and_5, n_collisions_ip2, n_collisions_ip8 = compute_collision_from_scheme(
        config_bb
    )
    bunch_charge = config_bb["num_particles_per_bunch"]
    T_rev0 = twiss_b1["T_rev0"]
    luminosity = compute_lumi(
        bunch_charge, twiss_b1, twiss_b2, crab, nemitt_x, nemitt_y, sigma_z, n_collisions_ip1_and_5
    )
    return compute_PU(luminosity, n_collisions_ip1_and_5, T_rev0)


# Extracted from master job
def configure_beam_beam(collider, config_bb):
    collider.configure_beambeam_interactions(
        num_particles=config_bb["num_particles_per_bunch"],
        nemitt_x=config_bb["nemitt_x"],
        nemitt_y=config_bb["nemitt_y"],
    )

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
    return collider


# %%
# Load collider and config for 30cm optics
collider_30cm_path = "/home/HPC/cdroin/example_DA_study_runIII_PU/master_study/scans/collider_30cm/base_collider/xtrack_0000/collider.json"
config_collider_30cm_path = "/home/HPC/cdroin/example_DA_study_runIII_PU/master_study/scans/collider_30cm/base_collider/xtrack_0000/config.yaml"
collider_30cm = xt.Multiline.from_json(collider_30cm_path)
config_30cm = analysis_functions.load_config(config_collider_30cm_path)["config_collider"][
    "config_beambeam"
]
collider_30cm.build_trackers()

# Same with 26cm and 22cm optics
collider_26cm_path = "/home/HPC/cdroin/example_DA_study_runIII_PU/master_study/scans/collider_26cm/base_collider/xtrack_0000/collider.json"
config_collider_26cm_path = "/home/HPC/cdroin/example_DA_study_runIII_PU/master_study/scans/collider_26cm/base_collider/xtrack_0000/config.yaml"
collider_26cm = xt.Multiline.from_json(collider_26cm_path)
config_26cm = analysis_functions.load_config(config_collider_26cm_path)["config_collider"][
    "config_beambeam"
]
collider_26cm.build_trackers()

collider_22cm_path = "/home/HPC/cdroin/example_DA_study_runIII_PU/master_study/scans/collider_22cm/base_collider/xtrack_0000/collider.json"
config_collider_22cm_path = "/home/HPC/cdroin/example_DA_study_runIII_PU/master_study/scans/collider_22cm/base_collider/xtrack_0000/config.yaml"
collider_22cm = xt.Multiline.from_json(collider_22cm_path)
config_22cm = analysis_functions.load_config(config_collider_22cm_path)["config_collider"][
    "config_beambeam"
]
collider_22cm.build_trackers()

# %%
# Set bunch charge and crossing angle for all points along the curve, recompute the beam-beam and compute the corresponding PU

# Start with 5.5 sigmas
ll_PU = []
for collider, array_xing, array_nb, config_bb in zip(
    [collider_30cm, collider_26cm, collider_22cm],
    [x_real_30cm, x_real_26cm, x_real_22cm],
    [y_real_30cm, y_real_26cm, y_real_22cm],
    [config_30cm, config_26cm, config_22cm],
):
    l_PU = []
    for xing, nb in zip(array_xing, array_nb):
        collider.vars["on_x1"] = float(xing)
        collider.vars["on_x5"] = float(xing)
        config_bb["num_particles_per_bunch"] = nb
        collider = configure_beam_beam(collider, config_bb)
        PU = compute_PU_from_lumi(collider, config_bb)
        l_PU.append(PU)
    ll_PU.append(l_PU)

# Convert ll_PU to numpy as save
np.save("PU_5dot5.npy", np.array(ll_PU))

# Same with 5 sigmas
ll_PU_5sig = []
for collider, array_xing, array_nb, config_bb in zip(
    [collider_30cm, collider_26cm, collider_22cm],
    [x_real_30cm_5sig, x_real_26cm_5sig, x_real_22cm_5sig],
    [y_real_30cm_5sig, y_real_26cm_5sig, y_real_22cm_5sig],
    [config_30cm, config_26cm, config_22cm],
):
    l_PU = []
    for xing, nb in zip(array_xing, array_nb):
        collider.vars["on_x1"] = float(xing)
        collider.vars["on_x5"] = float(xing)
        config_bb["num_particles_per_bunch"] = nb
        collider = configure_beam_beam(collider, config_bb)
        PU = compute_PU_from_lumi(collider, config_bb)
        l_PU.append(PU)
    ll_PU_5sig.append(l_PU)

# Convert ll_PU to numpy as save
np.save("PU_5.npy", np.array(ll_PU_5sig))
