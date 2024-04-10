# Imports
import json
import logging

import numpy as np
import xtrack as xt
from scipy.constants import c as clight
from scipy.optimize import minimize_scalar


# Function to generate dictionnary containing the orbit correction setup
def generate_orbit_correction_setup():
    correction_setup = {}
    correction_setup["lhcb1"] = {
        "IR1 left": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="e.ds.r8.b1",
            end="e.ds.l1.b1",
            vary=(
                "corr_co_acbh14.l1b1",
                "corr_co_acbh12.l1b1",
                "corr_co_acbv15.l1b1",
                "corr_co_acbv13.l1b1",
            ),
            targets=("e.ds.l1.b1",),
        ),
        "IR1 right": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="s.ds.r1.b1",
            end="s.ds.l2.b1",
            vary=(
                "corr_co_acbh13.r1b1",
                "corr_co_acbh15.r1b1",
                "corr_co_acbv12.r1b1",
                "corr_co_acbv14.r1b1",
            ),
            targets=("s.ds.l2.b1",),
        ),
        "IR5 left": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="e.ds.r4.b1",
            end="e.ds.l5.b1",
            vary=(
                "corr_co_acbh14.l5b1",
                "corr_co_acbh12.l5b1",
                "corr_co_acbv15.l5b1",
                "corr_co_acbv13.l5b1",
            ),
            targets=("e.ds.l5.b1",),
        ),
        "IR5 right": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="s.ds.r5.b1",
            end="s.ds.l6.b1",
            vary=(
                "corr_co_acbh13.r5b1",
                "corr_co_acbh15.r5b1",
                "corr_co_acbv12.r5b1",
                "corr_co_acbv14.r5b1",
            ),
            targets=("s.ds.l6.b1",),
        ),
        "IP1": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="e.ds.l1.b1",
            end="s.ds.r1.b1",
            vary=(
                "corr_co_acbch6.l1b1",
                "corr_co_acbcv5.l1b1",
                "corr_co_acbch5.r1b1",
                "corr_co_acbcv6.r1b1",
                "corr_co_acbyhs4.l1b1",
                "corr_co_acbyhs4.r1b1",
                "corr_co_acbyvs4.l1b1",
                "corr_co_acbyvs4.r1b1",
            ),
            targets=("ip1", "s.ds.r1.b1"),
        ),
        "IP2": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="e.ds.l2.b1",
            end="s.ds.r2.b1",
            vary=(
                "corr_co_acbyhs5.l2b1",
                "corr_co_acbchs5.r2b1",
                "corr_co_acbyvs5.l2b1",
                "corr_co_acbcvs5.r2b1",
                "corr_co_acbyhs4.l2b1",
                "corr_co_acbyhs4.r2b1",
                "corr_co_acbyvs4.l2b1",
                "corr_co_acbyvs4.r2b1",
            ),
            targets=("ip2", "s.ds.r2.b1"),
        ),
        "IP5": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="e.ds.l5.b1",
            end="s.ds.r5.b1",
            vary=(
                "corr_co_acbch6.l5b1",
                "corr_co_acbcv5.l5b1",
                "corr_co_acbch5.r5b1",
                "corr_co_acbcv6.r5b1",
                "corr_co_acbyhs4.l5b1",
                "corr_co_acbyhs4.r5b1",
                "corr_co_acbyvs4.l5b1",
                "corr_co_acbyvs4.r5b1",
            ),
            targets=("ip5", "s.ds.r5.b1"),
        ),
        "IP8": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="e.ds.l8.b1",
            end="s.ds.r8.b1",
            vary=(
                "corr_co_acbch5.l8b1",
                "corr_co_acbyhs4.l8b1",
                "corr_co_acbyhs4.r8b1",
                "corr_co_acbyhs5.r8b1",
                "corr_co_acbcvs5.l8b1",
                "corr_co_acbyvs4.l8b1",
                "corr_co_acbyvs4.r8b1",
                "corr_co_acbyvs5.r8b1",
            ),
            targets=("ip8", "s.ds.r8.b1"),
        ),
    }

    correction_setup["lhcb2"] = {
        "IR1 left": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="e.ds.l1.b2",
            end="e.ds.r8.b2",
            vary=(
                "corr_co_acbh13.l1b2",
                "corr_co_acbh15.l1b2",
                "corr_co_acbv12.l1b2",
                "corr_co_acbv14.l1b2",
            ),
            targets=("e.ds.r8.b2",),
        ),
        "IR1 right": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="s.ds.l2.b2",
            end="s.ds.r1.b2",
            vary=(
                "corr_co_acbh12.r1b2",
                "corr_co_acbh14.r1b2",
                "corr_co_acbv13.r1b2",
                "corr_co_acbv15.r1b2",
            ),
            targets=("s.ds.r1.b2",),
        ),
        "IR5 left": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="e.ds.l5.b2",
            end="e.ds.r4.b2",
            vary=(
                "corr_co_acbh13.l5b2",
                "corr_co_acbh15.l5b2",
                "corr_co_acbv12.l5b2",
                "corr_co_acbv14.l5b2",
            ),
            targets=("e.ds.r4.b2",),
        ),
        "IR5 right": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="s.ds.l6.b2",
            end="s.ds.r5.b2",
            vary=(
                "corr_co_acbh12.r5b2",
                "corr_co_acbh14.r5b2",
                "corr_co_acbv13.r5b2",
                "corr_co_acbv15.r5b2",
            ),
            targets=("s.ds.r5.b2",),
        ),
        "IP1": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="s.ds.r1.b2",
            end="e.ds.l1.b2",
            vary=(
                "corr_co_acbch6.r1b2",
                "corr_co_acbcv5.r1b2",
                "corr_co_acbch5.l1b2",
                "corr_co_acbcv6.l1b2",
                "corr_co_acbyhs4.l1b2",
                "corr_co_acbyhs4.r1b2",
                "corr_co_acbyvs4.l1b2",
                "corr_co_acbyvs4.r1b2",
            ),
            targets=(
                "ip1",
                "e.ds.l1.b2",
            ),
        ),
        "IP2": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="s.ds.r2.b2",
            end="e.ds.l2.b2",
            vary=(
                "corr_co_acbyhs5.l2b2",
                "corr_co_acbchs5.r2b2",
                "corr_co_acbyvs5.l2b2",
                "corr_co_acbcvs5.r2b2",
                "corr_co_acbyhs4.l2b2",
                "corr_co_acbyhs4.r2b2",
                "corr_co_acbyvs4.l2b2",
                "corr_co_acbyvs4.r2b2",
            ),
            targets=("ip2", "e.ds.l2.b2"),
        ),
        "IP5": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="s.ds.r5.b2",
            end="e.ds.l5.b2",
            vary=(
                "corr_co_acbch6.r5b2",
                "corr_co_acbcv5.r5b2",
                "corr_co_acbch5.l5b2",
                "corr_co_acbcv6.l5b2",
                "corr_co_acbyhs4.l5b2",
                "corr_co_acbyhs4.r5b2",
                "corr_co_acbyvs4.l5b2",
                "corr_co_acbyvs4.r5b2",
            ),
            targets=(
                "ip5",
                "e.ds.l5.b2",
            ),
        ),
        "IP8": dict(
            ref_with_knobs={"on_corr_co": 0, "on_disp": 0},
            start="s.ds.r8.b2",
            end="e.ds.l8.b2",
            vary=(
                "corr_co_acbchs5.l8b2",
                "corr_co_acbyhs5.r8b2",
                "corr_co_acbcvs5.l8b2",
                "corr_co_acbyvs5.r8b2",
                "corr_co_acbyhs4.l8b2",
                "corr_co_acbyhs4.r8b2",
                "corr_co_acbyvs4.l8b2",
                "corr_co_acbyvs4.r8b2",
            ),
            targets=(
                "ip8",
                "e.ds.l8.b2",
            ),
        ),
    }
    return correction_setup


def luminosity_leveling(
    collider,
    config_lumi_leveling,
    config_beambeam,
    additional_targets_lumi=[],
    crab=False,
):
    for ip_name in config_lumi_leveling.keys():
        print(f"\n --- Leveling in {ip_name} ---")

        config_this_ip = config_lumi_leveling[ip_name]
        bump_range = config_this_ip["bump_range"]

        assert config_this_ip[
            "preserve_angles_at_ip"
        ], "Only preserve_angles_at_ip=True is supported for now"
        assert config_this_ip[
            "preserve_bump_closure"
        ], "Only preserve_bump_closure=True is supported for now"

        beta0_b1 = collider.lhcb1.particle_ref.beta0[0]
        f_rev = 1 / (collider.lhcb1.get_length() / (beta0_b1 * clight))

        targets = []
        vary = []

        if "luminosity" in config_this_ip.keys():
            targets.append(
                xt.TargetLuminosity(
                    ip_name=ip_name,
                    luminosity=config_this_ip["luminosity"],
                    crab=crab,
                    tol=1e30,  # 0.01 * config_this_ip["luminosity"],
                    f_rev=f_rev,
                    num_colliding_bunches=config_this_ip["num_colliding_bunches"],
                    num_particles_per_bunch=config_beambeam["num_particles_per_bunch"],
                    sigma_z=config_beambeam["sigma_z"],
                    nemitt_x=config_beambeam["nemitt_x"],
                    nemitt_y=config_beambeam["nemitt_y"],
                    log=True,
                )
            )

            # Added this line for constraints
            targets.extend(additional_targets_lumi)
        elif "separation_in_sigmas" in config_this_ip.keys():
            targets.append(
                xt.TargetSeparation(
                    ip_name=ip_name,
                    separation_norm=config_this_ip["separation_in_sigmas"],
                    tol=1e-4,  # in sigmas
                    plane=config_this_ip["plane"],
                    nemitt_x=config_beambeam["nemitt_x"],
                    nemitt_y=config_beambeam["nemitt_y"],
                )
            )
        else:
            raise ValueError("Either `luminosity` or `separation_in_sigmas` must be specified")

        if config_this_ip["impose_separation_orthogonal_to_crossing"]:
            targets.append(xt.TargetSeparationOrthogonalToCrossing(ip_name="ip8"))
        vary.append(xt.VaryList(config_this_ip["knobs"], step=1e-4))

        # Target and knobs to rematch the crossing angles and close the bumps
        for line_name in ["lhcb1", "lhcb2"]:
            targets += [
                # Preserve crossing angle
                xt.TargetList(
                    ["px", "py"], at=ip_name, line=line_name, value="preserve", tol=1e-7, scale=1e3
                ),
                # Close the bumps
                xt.TargetList(
                    ["x", "y"],
                    at=bump_range[line_name][-1],
                    line=line_name,
                    value="preserve",
                    tol=1e-5,
                    scale=1,
                ),
                xt.TargetList(
                    ["px", "py"],
                    at=bump_range[line_name][-1],
                    line=line_name,
                    value="preserve",
                    tol=1e-5,
                    scale=1e3,
                ),
            ]

        vary.append(xt.VaryList(config_this_ip["corrector_knob_names"], step=1e-7))

        # Match
        tw0 = collider.twiss(lines=["lhcb1", "lhcb2"])
        collider.match(
            lines=["lhcb1", "lhcb2"],
            start=[bump_range["lhcb1"][0], bump_range["lhcb2"][0]],
            end=[bump_range["lhcb1"][-1], bump_range["lhcb2"][-1]],
            init=tw0,
            init_at=xt.START,
            targets=targets,
            vary=vary,
        )

    return collider


def compute_PU(luminosity, num_colliding_bunches, T_rev0, cross_section=81e-27):
    return luminosity / num_colliding_bunches * cross_section * T_rev0


def luminosity_leveling_ip1_5(
    collider,
    config_collider,
    config_bb,
    crab=False,
):
    # Get Twiss
    twiss_b1 = collider["lhcb1"].twiss()
    twiss_b2 = collider["lhcb2"].twiss()

    def compute_lumi(I):
        luminosity = xt.lumi.luminosity_from_twiss(
            n_colliding_bunches=config_collider["config_lumi_leveling_ip1_5"][
                "num_colliding_bunches"
            ],
            num_particles_per_bunch=I,
            ip_name="ip1",
            nemitt_x=config_bb["nemitt_x"],
            nemitt_y=config_bb["nemitt_y"],
            sigma_z=config_bb["sigma_z"],
            twiss_b1=twiss_b1,
            twiss_b2=twiss_b2,
            crab=crab,
        )
        return luminosity

    def f(I):
        luminosity = compute_lumi(I)

        PU = compute_PU(
            luminosity,
            config_collider["config_lumi_leveling_ip1_5"]["num_colliding_bunches"],
            twiss_b1["T_rev0"],
        )
        penalty_PU = max(
            0,
            (PU - config_collider["config_lumi_leveling_ip1_5"]["constraints"]["max_PU"]) * 1e35,
        )  # in units of 1e-35
        penalty_excess_lumi = max(
            0,
            (luminosity - config_collider["config_lumi_leveling_ip1_5"]["luminosity"]) * 10,
        )  # in units of 1e-35 if luminosity is in units of 1e34

        return (
            abs(luminosity - config_collider["config_lumi_leveling_ip1_5"]["luminosity"])
            + penalty_PU
            + penalty_excess_lumi
        )

    # Do the optimization
    res = minimize_scalar(
        f,
        bounds=(
            1e10,
            float(config_collider["config_lumi_leveling_ip1_5"]["constraints"]["max_intensity"]),
        ),
        method="bounded",
        options={"xatol": 1e7},
    )
    if not res.success:
        logging.warning("Optimization for leveling in IP 1/5 failed. Please check the constraints.")
    else:
        print(
            f"Optimization for leveling in IP 1/5 succeeded with I={res.x:.2e} particles per bunch"
        )
    return res.x


def install_wire(collider, config_wire):
    for beam in ["b1", "b2"]:
        line = collider[f"lhc{beam}"]

        # Create the knob for wire current and wire distance to the beam
        line.vars[f"i_wire_ip1.{beam}"] = 0.0
        line.vars[f"d_wire_ip1.{beam}"] = 0.01
        line.vars[f"i_wire_ip5.{beam}"] = 0.0
        line.vars[f"d_wire_ip5.{beam}"] = 0.01

        # Insert unconfigured wires on the line
        side = "r" if beam == "b2" else "l"
        sign = 1 if beam == "b2" else -1
        l_name_wire = [
            f"bbwc.t.4{side}1",
            f"bbwc.b.4{side}1",
            f"bbwc.e.4{side}5",
            f"bbwc.i.4{side}5",
        ]
        l_name_tct = [f"tctpv.4{side}1", f"tctpv.4{side}1", f"tctph.4{side}5", f"tctph.4{side}5"]
        l_h_dist = sign * np.array([0.0, 0.0, 1.0, -1.0])
        l_v_dist = sign * np.array([1.0, -1.0, 0.0, 0.0])

        # Tw to get the position of the tct, but need to discard tracker afterwards to unfreeze the line
        tw = line.twiss()
        l_s_tct = [
            ((tw.rows[f"{name_tct}.{beam}_entry"].s + tw.rows[f"{name_tct}.{beam}_exit"].s) / 2)[0]
            for name_tct in l_name_tct
        ]
        line.discard_tracker()
        for name_wire, name_tct, h_dist, v_dist, s_tct in zip(
            l_name_wire, l_name_tct, l_h_dist, l_v_dist, l_s_tct
        ):
            line.insert_element(
                name=f"{name_wire}.{beam}",
                element=xt.Wire(
                    L_phy=1,
                    L_int=2,
                    current=0.0,
                    xma=h_dist,
                    yma=v_dist,  # very far from the beam
                ),
                at_s=s_tct,
            )

        # Get closed orbit position at the location of the wire
        tw = line.twiss()
        # Careful, tct are repeated so only take one out of two
        x_tct_ip1, x_tct_ip5 = [
            ((tw.rows[f"{name_tct}.{beam}_entry"].x + tw.rows[f"{name_tct}.{beam}_exit"].x) / 2)[0]
            for name_tct in l_name_tct[::2]
        ]
        y_tct_ip1, y_tct_ip5 = [
            ((tw.rows[f"{name_tct}.{beam}_entry"].y + tw.rows[f"{name_tct}.{beam}_exit"].y) / 2)[0]
            for name_tct in l_name_tct[::2]
        ]

        # Create corresponding knob for closed orbit
        for co_wire, co in zip(
            [
                f"co_y_wire_ip1.{beam}",
                f"co_x_wire_ip1.{beam}",
                f"co_y_wire_ip5.{beam}",
                f"co_x_wire_ip5.{beam}",
            ],
            [y_tct_ip1, x_tct_ip1, y_tct_ip5, x_tct_ip5],
        ):
            line.vars[co_wire] = co

        # Create knob for current scaling, and wire distance scaling
        for name_wire in l_name_wire:
            # Check IP
            if "r1" in name_wire or "l1" in name_wire:
                ip = 1
            elif "r5" in name_wire or "l5" in name_wire:
                ip = 5
            else:
                raise ValueError("Invalid wire name")

            # Check plane
            if ".t." in name_wire or ".b." in name_wire:
                plane = "y"
                sign = 1 if ".t." in name_wire else -1
            elif ".e." in name_wire or ".i." in name_wire:
                plane = "x"
                sign = 1 if ".e." in name_wire else -1
            else:
                raise ValueError("Invalid wire name")

            # Assign knob
            line.element_refs[f"{name_wire}.{beam}"].current = line.vars[f"i_wire_ip{ip}.{beam}"]
            if plane == "y":
                line.element_refs[f"{name_wire}.{beam}"].yma = (
                    line.vars[f"d_wire_ip{ip}.{beam}"] + line.vars[f"co_y_wire_ip{ip}.{beam}"]
                )
            else:
                line.element_refs[f"{name_wire}.{beam}"].xma = (
                    sign * line.vars[f"d_wire_ip{ip}.{beam}"]
                    + line.vars[f"co_x_wire_ip{ip}.{beam}"]
                )

        # Lod knob for both IPs
        with open(config_wire["ip1"][beam]["path_knob"]) as f:
            data_ip1 = json.load(f)

        with open(config_wire["ip5"][beam]["path_knob"]) as f:
            data_ip5 = json.load(f)

        # Update wire distance
        side = "r" if beam == "b2" else "l"
        line.vars[f"d_wire_ip1.{beam}"] = (
            data_ip1["tct_opening_in_sigma"] * data_ip1[f"sigma_y_at_tctpv_4{side}1_{beam}"]
            + data_ip1["wire_retraction"]
        )
        line.vars[f"d_wire_ip5.{beam}"] = (
            data_ip5["tct_opening_in_sigma"] * data_ip5[f"sigma_x_at_tctph_4{side}5_{beam}"]
            + data_ip5["wire_retraction"]
        )

        # Assert initial k are correct in the knob
        for k0 in data_ip1["k_0"]:
            assert data_ip1["k_0"][k0] == line.vars[k0]._get_value()

        # Define list of k for the matching
        k_list_for_matching = [
            f"kq5.l1{beam}",
            f"kq5.r1{beam}",
            f"kq6.l1{beam}",
            f"kq6.r1{beam}",
            f"kq7.l1{beam}",
            f"kq7.r1{beam}",
            f"kq8.l1{beam}",
            f"kq8.r1{beam}",
            f"kq9.l1{beam}",
            f"kq9.r1{beam}",
            f"kq10.l1{beam}",
            f"kq10.r1{beam}",
            f"kqtl11.r1{beam}",
            f"kqt12.r1{beam}",
            f"kqt13.r1{beam}",
            f"kq4.l5{beam}",
            f"kq4.r5{beam}",
            f"kq5.l5{beam}",
            f"kq5.r5{beam}",
            f"kq6.l5{beam}",
            f"kq6.r5{beam}",
            f"kq7.l5{beam}",
            f"kq7.r5{beam}",
            f"kq8.l5{beam}",
            f"kq8.r5{beam}",
            f"kq9.l5{beam}",
            f"kq9.r5{beam}",
            f"kq10.l5{beam}",
            f"kq10.r5{beam}",
            f"kqtl11.r5{beam}",
            f"kqt12.r5{beam}",
            f"kqt13.r5{beam}",
        ]

        # Define/reset the delta_k as knobs, used to scale the knob with current
        def reset_delta_k(k_list):
            for kk in k_list:
                collider.vars[f"{kk}_delta"] = 0.000000

        reset_delta_k(k_list_for_matching)

        # Set the delta_k
        for data in [data_ip1, data_ip5]:
            for delta_k in data["k_delta"]:
                collider.vars[f"{delta_k}_delta"] = data["k_delta"][delta_k]

        # Set the k
        for k in k_list_for_matching:
            collider.vars[f"{k}_0"] = collider.vars[k]._get_value()
            # collider.vars[f'{k}_delta'] = 0.000000
            if "r1" in k or "l1" in k:
                collider.vars[k] = (
                    collider.vars[f"{k}_0"]
                    + collider.vars[f"{k}_delta"] * collider.vars[f"i_wire_ip1.{beam}"] / 350
                )
            elif "r5" in k or "l5" in k:
                collider.vars[k] = (
                    collider.vars[f"{k}_0"]
                    + collider.vars[f"{k}_delta"] * collider.vars[f"i_wire_ip5.{beam}"] / 350
                )

        # Set the current
        line.vars[f"i_wire_ip1.{beam}"] = config_wire["ip1"][beam]["i"]
        line.vars[f"i_wire_ip5.{beam}"] = config_wire["ip5"][beam]["i"]

    return collider


if __name__ == "__main__":
    correction_setup = generate_orbit_correction_setup()
    for nn in ["lhcb1", "lhcb2"]:
        with open(f"corr_co_{nn}.json", "w") as fid:
            json.dump(correction_setup[nn], fid, indent=4)
