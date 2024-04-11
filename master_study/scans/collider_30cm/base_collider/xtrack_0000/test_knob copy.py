# %%
import json

import numpy as np
import xtrack as xt
from matplotlib import pyplot as plt

my_beam = BEAM = "b2"  # this is the beam 4 in madx sense
collider = xt.Multiline.from_json("../collider/collider_43.json")


line = collider[f"lhc{my_beam}"]

# %%
tw = line.twiss()
tw.rows["ip.*"][["x", "px", "y", "py", "betx", "bety"]]

# %%
line.vars[f"i_wire_ip1.{my_beam}"] = 0.0
line.vars[f"d_wire_ip1.{my_beam}"] = 0.01

line.vars[f"i_wire_ip5.{my_beam}"] = 0.0
line.vars[f"d_wire_ip5.{my_beam}"] = 0.01

side = "r" if BEAM == "b2" else "l"
sign = 1 if BEAM == "b2" else -1
l_name_wire = [f"bbwc.t.4{side}1", f"bbwc.b.4{side}1", f"bbwc.e.4{side}5", f"bbwc.i.4{side}5"]
l_name_tct = [f"tctpv.4{side}1", f"tctpv.4{side}1", f"tctph.4{side}5", f"tctph.4{side}5"]
l_h_dist = sign * np.array([0.0, 0.0, 1.0, -1.0])
l_v_dist = sign * np.array([1.0, -1.0, 0.0, 0.0])

# Tw to get the position of the tct, but need to discard tracker afterwards to unfreeze the line
tw = line.twiss()
l_s_tct = [
    ((tw.rows[f"{name_tct}.{BEAM}_entry"].s + tw.rows[f"{name_tct}.{BEAM}_exit"].s) / 2)[0]
    for name_tct in l_name_tct
]
line.discard_tracker()
for name_wire, name_tct, h_dist, v_dist, s_tct in zip(
    l_name_wire, l_name_tct, l_h_dist, l_v_dist, l_s_tct
):
    line.insert_element(
        name=f"{name_wire}.{BEAM}",
        element=xt.Wire(
            L_phy=1,
            L_int=2,
            current=0.0,
            xma=h_dist,
            yma=v_dist,  # very far from the beam
        ),
        at_s=s_tct,
    )

# %%
tw = line.twiss()
tw.rows["ip.*"][["x", "px", "y", "py", "betx", "bety"]]

# %%
x_tct_ip1, x_tct_ip5 = 0.0, 0.0
y_tct_ip1, y_tct_ip5 = 0.0, 0.0
for co_wire, co in zip(
    [
        f"co_y_wire_ip1.{BEAM}",
        f"co_x_wire_ip1.{BEAM}",
        f"co_y_wire_ip5.{BEAM}",
        f"co_x_wire_ip5.{BEAM}",
    ],
    [y_tct_ip1, x_tct_ip1, y_tct_ip5, x_tct_ip5],
):
    line.vars[co_wire] = co
# %%
for name_wire in l_name_wire:
    # Check IP
    if "r1" in name_wire or "l1" in name_wire:
        ip = 1
        plane = "y"
    elif "r5" in name_wire or "l5" in name_wire:
        ip = 5
        plane = "x"
    else:
        raise ValueError("Invalid wire name")

    # Check sign
    if ".t." in name_wire or ".e." in name_wire:
        sign = 1
    elif ".b." in name_wire or ".i." in name_wire:
        sign = -1
    else:
        raise ValueError("Invalid wire name")

    # Assign knob
    line.element_refs[f"{name_wire}.{BEAM}"].current = line.vars[f"i_wire_ip{ip}.{BEAM}"]
    if plane == "y":
        line.element_refs[f"{name_wire}.{BEAM}"].yma = (
            sign * line.vars[f"d_wire_ip{ip}.{BEAM}"] + line.vars[f"co_y_wire_ip{ip}.{BEAM}"]
        )
    else:
        line.element_refs[f"{name_wire}.{BEAM}"].xma = (
            sign * line.vars[f"d_wire_ip{ip}.{BEAM}"] + line.vars[f"co_x_wire_ip{ip}.{BEAM}"]
        )


# %%
for ip in [2, 8]:
    collider.vars[f"on_x{ip}h"] = 0.0
    collider.vars[f"on_x{ip}v"] = 0.0
    collider.vars[f"on_sep{ip}h"] = 0.0
    collider.vars[f"on_sep{ip}v"] = 0.0

for ip in [1, 2, 5, 8]:
    print(8 * "*", f"IP{ip}", 8 * "*")
    if ip in [2, 8]:
        print(f"on_sep{ip}h:\t ", collider.vars[f"on_sep{ip}h"]._get_value())
        print(f"on_sep{ip}v:\t ", collider.vars[f"on_sep{ip}v"]._get_value())
    else:
        print(f"on_x{ip}:\t\t ", collider.vars[f"on_x{ip}"]._get_value())
        print(f"on_sep{ip}:\t ", collider.vars[f"on_sep{ip}"]._get_value())
    print(f"on_oh{ip}:\t\t ", collider.vars[f"on_oh{ip}"]._get_value())
    print(f"on_ov{ip}:\t\t ", collider.vars[f"on_ov{ip}"]._get_value())
    print(f"on_a{ip}:\t\t ", collider.vars[f"on_a{ip}"]._get_value())

print(8 * "*", "others settings", 8 * "*")
print("on_alice_normalized:\t", collider.vars["on_alice_normalized"]._get_value())
print("on_lhcb_normalized:\t", collider.vars["on_lhcb_normalized"]._get_value())
print("on_disp:\t\t", collider.vars["on_disp"]._get_value())

# %%
# import json file
# Lod knob for both IPs

with open(
    f"/afs/cern.ch/work/c/cdroin/private/example_DA_study_runIII_wire/master_study/master_jobs/knobs_wire/knob_dict_350A_8sigma@30cm_ip5_beta30_{BEAM}.json"
) as f:
    data_ip1 = data_ip5 = json.load(f)


side = "r" if BEAM == "b2" else "l"
line.vars[f"d_wire_ip1.{BEAM}"] = (
    data_ip1["tct_opening_in_sigma"] * data_ip1[f"sigma_y_at_tctpv_4{side}1_{BEAM}"]
    + data_ip1["wire_retraction"]
)
line.vars[f"d_wire_ip5.{BEAM}"] = (
    data_ip5["tct_opening_in_sigma"] * data_ip5[f"sigma_x_at_tctph_4{side}5_{BEAM}"]
    + data_ip5["wire_retraction"]
)


# %%
tw = line.twiss()
print(tw.qx, tw.qy)
tw.rows["ip.*"][["x", "px", "y", "py", "betx", "bety"]]
# %%

for ii in data_ip5["k_0"]:
    assert data_ip5["k_0"][ii] == line.vars[ii]._get_value()

# %%
k_list_for_matching = [
    f"kq5.l1{BEAM}",
    f"kq5.r1{BEAM}",
    f"kq6.l1{BEAM}",
    f"kq6.r1{BEAM}",
    f"kq7.l1{BEAM}",
    f"kq7.r1{BEAM}",
    f"kq8.l1{BEAM}",
    f"kq8.r1{BEAM}",
    f"kq9.l1{BEAM}",
    f"kq9.r1{BEAM}",
    f"kq10.l1{BEAM}",
    f"kq10.r1{BEAM}",
    f"kqtl11.r1{BEAM}",
    f"kqt12.r1{BEAM}",
    f"kqt13.r1{BEAM}",
    f"kq4.l5{BEAM}",
    f"kq4.r5{BEAM}",
    f"kq5.l5{BEAM}",
    f"kq5.r5{BEAM}",
    f"kq6.l5{BEAM}",
    f"kq6.r5{BEAM}",
    f"kq7.l5{BEAM}",
    f"kq7.r5{BEAM}",
    f"kq8.l5{BEAM}",
    f"kq8.r5{BEAM}",
    f"kq9.l5{BEAM}",
    f"kq9.r5{BEAM}",
    f"kq10.l5{BEAM}",
    f"kq10.r5{BEAM}",
    f"kqtl11.r5{BEAM}",
    f"kqt12.r5{BEAM}",
    f"kqt13.r5{BEAM}",
]


def reset_delta_k(k_list):
    for kk in k_list:
        collider.vars[f"{kk}_delta"] = 0.000000


reset_delta_k(k_list_for_matching)

for ii in k_list_for_matching:
    collider.vars[f"{ii}_0"] = collider.vars[ii]._get_value()
    collider.vars[f"{ii}_delta"] = 0.000000
    if "r1" in ii:
        collider.vars[ii] = (
            collider.vars[f"{ii}_0"]
            + collider.vars[f"{ii}_delta"] * collider.vars["i_wire_ip1.b2"] / 350
        )
    if "l1" in ii:
        collider.vars[ii] = (
            collider.vars[f"{ii}_0"]
            + collider.vars[f"{ii}_delta"] * collider.vars["i_wire_ip1.b2"] / 350
        )

    if "r5" in ii:
        collider.vars[ii] = (
            collider.vars[f"{ii}_0"]
            + collider.vars[f"{ii}_delta"] * collider.vars["i_wire_ip5.b2"] / 350
        )
    if "l5" in ii:
        collider.vars[ii] = (
            collider.vars[f"{ii}_0"]
            + collider.vars[f"{ii}_delta"] * collider.vars["i_wire_ip5.b2"] / 350
        )


for ii in data_ip5["k_delta"]:
    collider.vars[f"{ii}_delta"] = data_ip5["k_delta"][ii]

tw_ref = line.twiss(method="4d")


# %%
line.vars["i_wire_ip1.b2"] = data_ip5["i_wire_ip1.b2"]
line.vars["i_wire_ip5.b2"] = data_ip5["i_wire_ip5.b2"]
# %%

tw_new = line.twiss(method="4d")
plt.plot(tw_ref["s"], (tw_new["betx"] - tw_ref["betx"]) / tw_ref["betx"], label="ref")
plt.plot(tw_ref["s"], (tw_new["bety"] - tw_ref["bety"]) / tw_ref["bety"], label="ref")

# %%

print(tw_ref.qx, tw_new.qx)
print(tw_ref.qy, tw_new.qy)
# %%
