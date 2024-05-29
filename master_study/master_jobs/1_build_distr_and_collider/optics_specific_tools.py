import numpy as np
import xmask as xm


def check_madx_lattices(mad):
    assert mad.globals["qxb1"] == mad.globals["qxb2"]
    assert mad.globals["qyb1"] == mad.globals["qyb2"]
    assert mad.globals["qpxb1"] == mad.globals["qpxb2"]
    assert mad.globals["qpyb1"] == mad.globals["qpyb2"]

    try:
        assert np.isclose(mad.table.summ.q1, mad.globals["qxb1"], atol=1e-02)
        assert np.isclose(mad.table.summ.q2, mad.globals["qyb1"], atol=1e-02)
        assert np.isclose(mad.table.summ.dq1, mad.globals["qpxb1"], atol=5e-01)
        assert np.isclose(mad.table.summ.dq2, mad.globals["qpyb1"], atol=5e-01)
    except AssertionError:
        print("Warning: some of the Qx, Qy, DQx, DQy values are not close to the expected ones")

    df = mad.table.twiss.dframe()
    for my_ip in [1, 2, 5, 8]:
        # assert np.isclose(df.loc[f"ip{my_ip}"].betx, mad.globals[f"betx_IP{my_ip}"], rtol=1e-02)
        # assert np.isclose(df.loc[f"ip{my_ip}"].bety, mad.globals[f"bety_IP{my_ip}"], rtol=1e-02)
        assert np.isclose(df.loc[f"ip{my_ip}"].betx, mad.globals[f"betxIP{my_ip}b1"], rtol=1e-02)
        assert np.isclose(df.loc[f"ip{my_ip}"].bety, mad.globals[f"betyIP{my_ip}b1"], rtol=1e-02)

    mad.twiss()
    df = mad.table.twiss.dframe()

    try:
        assert df["x"].std() < 1e-6
        assert df["y"].std() < 1e-6
    except AssertionError:
        print("Warning: the standard deviation of x and y are not close to zero")


def check_xsuite_lattices(my_line):
    tw = my_line.twiss(method="6d", matrix_stability_tol=100)
    print(f"--- Now displaying Twiss result at all IPS for line {my_line}---")
    print(tw[:, "ip.*"])
    # print qx and qy
    print(f"--- Now displaying Qx and Qy for line {my_line}---")
    print(tw.qx, tw.qy)


def build_sequence(mad, mylhcbeam, beam_config, ignore_cycling=False, slice_factor=8):
    # Select beam
    mad.input(f"mylhcbeam = {mylhcbeam}")
    mad.input("option, -echo,warn, -info;")

    # optics dependent macros (for splitting)
    mad.call("acc-models-lhc/runII/2018/toolkit/macro.madx")

    assert mylhcbeam in [1, 2, 4], "Invalid mylhcbeam (it should be in [1, 2, 4])"

    if mylhcbeam in [1, 2]:
        mad.call("acc-models-lhc/runII/2018/lhc_as-built.seq")
    else:
        mad.call("acc-models-lhc/runII/2018/lhcb4_as-built.seq")

    # New IR7 MQW layout and cabling
    mad.call("acc-models-lhc/runIII/RunIII_dev/IR7-Run3seqedit.madx")

    # Makethin part
    if slice_factor > 0:
        # the variable in the macro is slice_factor
        mad.input(f"slicefactor={slice_factor};")
        mad.call("acc-models-lhc/runII/2018/toolkit/myslice.madx")
        if mylhcbeam == 1:
            xm.attach_beam_to_sequence(mad.sequence["lhcb1"], 1, beam_config["lhcb1"])
            xm.attach_beam_to_sequence(mad.sequence["lhcb2"], 2, beam_config["lhcb2"])
        elif mylhcbeam == 4:
            xm.attach_beam_to_sequence(mad.sequence["lhcb2"], 4, beam_config["lhcb2"])
        else:
            raise ValueError("Invalid mylhcbeam")
        # mad.beam()
        for my_sequence in ["lhcb1", "lhcb2"]:
            if my_sequence in list(mad.sequence):
                mad.input(
                    f"use, sequence={my_sequence}; makethin,"
                    f"sequence={my_sequence}, style=teapot, makedipedge=true;"
                )
    else:
        print("WARNING: The sequences are not thin!")

    # Cycling w.r.t. to IP3 (mandatory to find closed orbit in collision in the presence of errors)
    if not ignore_cycling:
        for my_sequence in ["lhcb1", "lhcb2"]:
            if my_sequence in list(mad.sequence):
                mad.input(
                    f"seqedit, sequence={my_sequence}; flatten;"
                    "cycle, start=IP3; flatten; endedit;"
                )


def apply_optics(mad, optics_file):
    mad.call(optics_file)
    mad.call("ir7_strengths.madx")
    mad.input("on_alice := on_alice_normalized * 7000. / nrj;")
    mad.input("on_lhcb := on_lhcb_normalized * 7000. / nrj;")


def apply_BFPP(mad):
    mad.input("""acbch8.r2b1        :=   6.336517325e-05 * ON_BFPP.R2 / 7.8;
                acbch10.r2b1       :=   2.102863759e-05 * ON_BFPP.R2 / 7.8;
                acbh12.r2b1        :=   4.404997133e-05 * ON_BFPP.R2 / 7.8;

                acbch7.r1b1        :=   4.259479019e-06 * ON_BFPP.R1 / 2.5;
                acbch9.r1b1        :=   1.794045373e-05 * ON_BFPP.R1 / 2.5;
                acbh13.r1b1        :=   1.371178403e-05 * ON_BFPP.R1 / 2.5;

                acbch7.r5b1        :=   2.153161387e-06 * ON_BFPP.R5 / 1.3;
                acbch9.r5b1        :=   9.314782805e-06 * ON_BFPP.R5 / 1.3;
                acbh13.r5b1        :=   7.12996247e-06 * ON_BFPP.R5 / 1.3;

                acbch8.r8b1        :=   3.521812667e-05 * ON_BFPP.R8 / 4.6;
                acbch10.r8b1       :=   1.064966564e-05 * ON_BFPP.R8 / 4.6;
                acbh12.r8b1        :=   2.786990521e-05 * ON_BFPP.R8 / 4.6;
            """)
