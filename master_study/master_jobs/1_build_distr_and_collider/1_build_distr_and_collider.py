"""This script is used to build the base collider with Xmask, configuring only the optics. Functions
in this script are called sequentially."""

# ==================================================================================================
# --- Imports
# ==================================================================================================

# Import standard library modules
import itertools
import json
import logging
import os
import shutil

# Import third-party modules
import numpy as np

# Import user-defined modules
import optics_specific_tools as ost
import pandas as pd
import tree_maker
import xmask as xm
import xmask.lhc as xlhc
import xobjects as xo
import yaml
from cpymad.madx import Madx


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
# --- Function to get context
# ==================================================================================================
def get_context(configuration):
    if configuration["context"] == "cupy":
        context = xo.ContextCupy()
    elif configuration["context"] == "opencl":
        context = xo.ContextPyopencl()
    elif configuration["context"] == "cpu":
        context = xo.ContextCpu()
    else:
        logging.warning("context not recognized, using cpu")
        context = xo.ContextCpu()
    return context


# ==================================================================================================
# --- Function to load configuration file
# ==================================================================================================
def load_configuration(config_path="config.yaml"):
    # Load configuration
    with open(config_path, "r") as fid:
        configuration = yaml.safe_load(fid)

    # Get configuration for the particles distribution and the collider separately
    config_particles = configuration["config_particles"]
    config_mad = configuration["config_mad"]

    return configuration, config_particles, config_mad


# ==================================================================================================
# --- Function to build particle distribution and write it to file
# ==================================================================================================
def build_particle_distribution(config_particles):
    # Define radius distribution
    r_min = config_particles["r_min"]
    r_max = config_particles["r_max"]
    n_r = config_particles["n_r"]
    radial_list = np.linspace(r_min, r_max, n_r, endpoint=False)

    # Filter out particles with low and high amplitude to accelerate simulation
    # radial_list = radial_list[(radial_list >= 4.5) & (radial_list <= 7.5)]

    # Define angle distribution
    n_angles = config_particles["n_angles"]
    theta_list = np.linspace(0, 90, n_angles + 2)[1:-1]

    # Define particle distribution as a cartesian product of the above
    particle_list = [
        (particle_id, ii[1], ii[0])
        for particle_id, ii in enumerate(itertools.product(theta_list, radial_list))
    ]

    # Split distribution into several chunks for parallelization
    n_split = config_particles["n_split"]
    particle_list = list(np.array_split(particle_list, n_split))

    # Return distribution
    return particle_list


def write_particle_distribution(particle_list):
    # Write distribution to parquet files
    distributions_folder = "particles"
    os.makedirs(distributions_folder, exist_ok=True)
    for idx_chunk, my_list in enumerate(particle_list):
        pd.DataFrame(
            my_list,
            columns=["particle_id", "normalized amplitude in xy-plane", "angle in xy-plane [deg]"],
        ).to_parquet(f"{distributions_folder}/{idx_chunk:02}.parquet")


# ==================================================================================================
# --- Function to build collider from mad model
# ==================================================================================================
def load_optics_runIII(path_optics,  path_settings = None):

    # Get run
    run = path_optics.split('/')[1]

    # Build sequence
    mad = Madx()

    # Prepare for building sequence
    mad.input("""
    option,-echo,-info;
    system,"mkdir temp";
    """)

    # Build sequence
    mad.input(f"""
    call,file="modules/{run}/lhc.seq";
    """)


    # Apply macro
    mad.input(f"""
    call,file="modules/{run}/toolkit/macro.madx";
    """)

    mad.input(f"""exec,mk_beam(450);""")

    # Injection optics
    mad.input(f"""
    call,file="{path_optics}";
    """)

    # Phase knob
    if path_settings is not None:
        mad.input(f"""
        call,file="{path_settings}";
        """)
    
    return mad


def macros_runIII(mad):
    mad.input("""
    twiss_opt: macro = {
    set,format=".15g";
    select,flag=twiss,clear;
    select,flag=twiss,
        column=name,s,l,
                lrad,angle,k1l,k2l,k3l,k1sl,k2sl,k3sl,hkick,vkick,kick,tilt,
                betx,bety,alfx,alfy,dx,dpx,dy,dpy,mux,muy,x,y,px,py,t,pt,
                wx,wy,phix,phiy,n1,ddx,ddy,ddpx,ddpy,
                keyword,aper_1,aper_2,aper_3,aper_4,
                apoff_1,apoff_2,
                aptol_1,aptol_2,aptol_3,apertype,mech_sep;
    select,flag=aperture,
        column=name,s,n1,aper_1,aper_2,aper_3,aper_4,rtol,xtol,ytol,
                apoff_1,apoff_2,
                betx,bety,dx,dy,x,y,apertype;
    };
    """)


    mad.input("""
    CHECK_IP(BIM): macro = {
    exec,twiss_opt;
    use,sequence=lhcBIM;
    if (mylhcbeam<3) {
    twiss,file=twiss_lhcBIM.tfs;
    } else {
    twiss,file=twiss_lhcb4.tfs;
    };
    refbetxIP1BIM=table(twiss,IP1,betx); refbetyIP1BIM=table(twiss,IP1,bety);
    refbetxIP5BIM=table(twiss,IP5,betx); refbetyIP5BIM=table(twiss,IP5,bety);
    refbetxIP2BIM=table(twiss,IP2,betx); refbetyIP2BIM=table(twiss,IP2,bety);
    refbetxIP8BIM=table(twiss,IP8,betx); refbetyIP8BIM=table(twiss,IP8,bety);
    refqxBIM=table(summ,q1); refqyBIM=table(summ,q2);
    refdqxBIM=table(summ,dq1); refdqyBIM=table(summ,dq2);
    refxIP1BIM=table(twiss,IP1,x); refyIP1BIM=table(twiss,IP1,y);
    refxIP5BIM=table(twiss,IP5,x); refyIP5BIM=table(twiss,IP5,y);
    refxIP2BIM=table(twiss,IP2,x); refyIP2BIM=table(twiss,IP2,y);
    refxIP8BIM=table(twiss,IP8,x); refyIP8BIM=table(twiss,IP8,y);
    refpxIP1BIM=table(twiss,IP1,px); refpyIP1BIM=table(twiss,IP1,py);
    refpxIP5BIM=table(twiss,IP5,px); refpyIP5BIM=table(twiss,IP5,py);
    refpxIP2BIM=table(twiss,IP2,px); refpyIP2BIM=table(twiss,IP2,py);
    refpxIP8BIM=table(twiss,IP8,px); refpyIP8BIM=table(twiss,IP8,py);
    refxIP3BIM=table(twiss,IP3,x); refyIP3BIM=table(twiss,IP3,y);
    refxIP4BIM=table(twiss,IP4,x); refyIP4BIM=table(twiss,IP4,y);
    refxIP6BIM=table(twiss,IP6,x); refyIP6BIM=table(twiss,IP6,y);
    refxIP7BIM=table(twiss,IP7,x); refyIP7BIM=table(twiss,IP7,y);
    refpxIP3BIM=table(twiss,IP3,px); refpyIP3BIM=table(twiss,IP3,py);
    refpxIP4BIM=table(twiss,IP4,px); refpyIP4BIM=table(twiss,IP4,py);
    refpxIP6BIM=table(twiss,IP6,px); refpyIP6BIM=table(twiss,IP6,py);
    refpxIP7BIM=table(twiss,IP7,px); refpyIP7BIM=table(twiss,IP7,py);
    value,refbetxIP1BIM,refbetyIP1BIM;
    value,refbetxIP5BIM,refbetyIP5BIM;
    value,refbetxIP2BIM,refbetyIP2BIM;
    value,refbetxIP8BIM,refbetyIP8BIM;
    value,refqxBIM,refqyBIM;
    value,refdqxBIM,refdqyBIM;
    value,refxIP1BIM,refyIP1BIM;
    value,refxIP5BIM,refyIP5BIM;
    value,refxIP2BIM,refyIP2BIM;
    value,refxIP8BIM,refyIP8BIM;
    value,refpxIP1BIM,refpyIP1BIM;
    value,refpxIP5BIM,refpyIP5BIM;
    value,refpxIP2BIM,refpyIP2BIM;
    value,refpxIP8BIM,refpyIP8BIM;
    };
    """)

    return mad

def check_and_load_twiss_runIII(mad):
    
    mad.input("""
    exec,check_ip(b1);
    exec,check_ip(b2);
    """
    )

    # Load Twiss
    tb1=optics.open('twiss_lhcb1.tfs')
    tb2=optics.open('twiss_lhcb2.tfs')
    
    # Impose strength for octupoles
    tb1.k3l[tb1//'mo.*']=0.1
    tb2.k3l[tb2//'mo.*']=0.1
    
    return mad, tb1, tb2


# Function to compute RDTs
def compute_RDT(t, RDT = (0,4)):
    return abs(np.cumsum(t.drvterm(*RDT)*t.k3l))

# Get all RDTs
def get_all_RDTs(t):
    #return t.s, {RDT : driving_term_oct(t, *RDT) for RDT in [(4,0), (4,0), (1,3), (3,1), (2,2)]}
    return t.s, {RDT : compute_RDT(t, RDT) for RDT in [(0,4), (4,0), (1,3), (3,1), (2,2)]}

# Plot RDTs
def plot_RDT(s, dic_RDTs, title = None, title_save = "RDT.pdf"):
    for type_RDT, RDT in dic_RDTs.items():
        plt.plot(s,RDT, label = str(type_RDT))
    plt.xlabel('s [m]')
    plt.ylabel('RDTs (a.u.)')
    plt.legend()
    plt.grid()
    if title is not None:
        plt.title(title)
    plt.savefig(title_save, bbox_inches='tight')
    #plt.show()
    
def return_mu_values_runIII(t):
    # Get relevant markers
    cond =[True if x.startswith('s.ds.l') or x.startswith('e.ds.r') else False for x in t.name ]
    l_markers = t.name[cond]
    l_s_markers = t.s[cond]
    l_mux_markers = t.mux[cond]
    l_muy_markers = t.muy[cond]

    # Add last element at the beginning to close the ring
    l_markers = np.insert(l_markers, 0, l_markers[-1])
    l_s_markers = np.insert(l_s_markers, 0, l_s_markers[-1])
    l_mux_markers = np.insert(l_mux_markers, 0, l_mux_markers[-1]-t.mux[-1])
    l_muy_markers = np.insert(l_muy_markers, 0, l_muy_markers[-1]-t.muy[-1])
    
    
    # Get list of mu values in the arcs and straight sections
    l_mux_strengths = l_mux_markers[1:] - l_mux_markers[:-1]
    l_muy_strengths = l_muy_markers[1:] - l_muy_markers[:-1]
    l_mux_ss = l_mux_strengths[0::2]
    l_mux_arc = l_mux_strengths[1::2]
    l_muy_ss = l_muy_strengths[0::2]
    l_muy_arc = l_muy_strengths[1::2]
    
    return l_mux_ss, l_mux_arc, l_muy_ss, l_muy_arc
    
def load_RDT_runIII():
    # Load paths
    path_optics_with_knob= "modules/runIII2023/operation/optics/R2023a_A11mC11mA10mL10m_PhaseKnob100ON.madx"
    path_settings = "modules/runIII2023/scenarios/pp_lumi/RAMP-SQUEEZE-6.8TeV-ATS-2m-2023_V1/0/settings.madx"

    # Plot RDTs with and without phase knob
    mad = load_optics_runIII(path_optics_with_knob, path_hl = None, path_settings = path_settings)
    mad = macros_runIII(mad)
    mad, tb1, tb2 = check_and_load_twiss_runIII(mad)
    s_with_knob, dic_RDTs_with_knob = get_all_RDTs(tb1)
    plot_RDT(s_with_knob, dic_RDTs_with_knob, title = 'RunIII with knob', title_save = "RDT_with_knob.pdf")

    # Remove phase knob
    mad.input("phase_change.b1=0.0;")
    mad.input("phase_change.b2=0.0;")

    # Replot
    mad, tb1, tb2 = check_and_load_twiss_runIII(mad)
    s_without_knob, dic_RDTs_without_knob = get_all_RDTs(tb1)

    plot_RDT(s_without_knob, dic_RDTs_without_knob, title = 'RunIII without knob', title_save = "RDT_without_knob.pdf")
    
    # Reactivate phase knob and get phase advance
    mad.input("phase_change.b1=1.0;")
    mad.input("phase_change.b2=1.0;")
    mad, tb1, tb2 = check_and_load_twiss_runIII(mad)
    ll_mu_b1_runIII_with_knob = return_mu_values_runIII(tb1)
    ll_mu_b2_runIII_with_knob = return_mu_values_runIII(tb2)
    
    return ll_mu_b1_runIII_with_knob, ll_mu_b2_runIII_with_knob


def rematch_optics(mad, path_rematch = "modules/hllhc16/toolkit/rematch_hllhc.madx"):
    mad.input(f"""
    call,file="{path_rematch}";      
    """)
    return mad

def build_collider_from_mad(config_mad, context, ll_mu_b1_runIII_with_knob, ll_mu_b2_runIII_with_knob, sanity_checks=True):
    # Make mad environment
    xm.make_mad_environment(links=config_mad["links"])

    # Start mad
    mad_b1b2 = Madx(command_log="mad_collider.log")

    mad_b4 = Madx(command_log="mad_b4.log")

    # Build sequences
    ost.build_sequence(mad_b1b2, mylhcbeam=1, ignore_cycling = True)
    ost.build_sequence(mad_b4, mylhcbeam=4, ignore_cycling = True)

    # Apply optics (only for b1b2, b4 will be generated from b1b2)
    ost.apply_optics(mad_b1b2, optics_file=config_mad["optics_file"])

    # Rematch optics
    mad_b1b2 = rematch_optics(mad_b1b2)

    # Impose run III with knob phases to HL
    l_mux_ss_b1, l_mux_arc_b1, l_muy_ss_b1, l_muy_arc_b1 = ll_mu_b1_runIII_with_knob
    l_mux_ss_b2, l_mux_arc_b2, l_muy_ss_b2, l_muy_arc_b2 = ll_mu_b2_runIII_with_knob

    for ip, (mux_ss_b1, muy_ss_b1, mux_ss_b2, muy_ss_b2) in enumerate(zip(l_mux_ss_b1, l_muy_ss_b1, l_mux_ss_b2, l_muy_ss_b2)):
        mad_b1b2.input(f"""
        muxIP{ip+1}b1={mux_ss_b1};
        muyIP{ip+1}b1={muy_ss_b1};
        muxIP{ip+1}b2={mux_ss_b2};
        muyIP{ip+1}b2={muy_ss_b2};
        """)
        
    for ip, (mux_arc_b1, muy_arc_b1, mux_arc_b2, muy_arc_b2) in enumerate(zip(l_mux_arc_b1, l_muy_arc_b1, l_mux_arc_b2, l_muy_arc_b2)):
        mad_b1b2.input(f"""
        mux{ip+1}{(ip+2)%8}b1={mux_arc_b1};
        muy{ip+1}{(ip+2)%8}b1={muy_arc_b1};
        mux{ip+1}{(ip+2)%8}b2={mux_arc_b2};
        muy{ip+1}{(ip+2)%8}b2={muy_arc_b2};
        """)

    # Rematch optics
    mad_b1b2 = rematch_optics(mad_b1b2)

    mad_b1b2, tb1, tb2 = check_and_load_twiss_runIII(mad_b1b2)
    s, dic_RDTs = get_all_RDTs(tb1)
    plot_RDT(s, dic_RDTs, title = 'HL with RunIII phase rematch', title_save = "HL_rematched_RDT.pdf")


    if sanity_checks:
        mad_b1b2.use(sequence="lhcb1")
        mad_b1b2.twiss()
        ost.check_madx_lattices(mad_b1b2)
        mad_b1b2.use(sequence="lhcb2")
        mad_b1b2.twiss()
        ost.check_madx_lattices(mad_b1b2)

    # Rematch optics
    mad = rematch_optics(mad_b1b2)

    # # Apply optics (only for b4, just for check)
    # ost.apply_optics(mad_b4, optics_file=config_mad["optics_file"])
    # if sanity_checks:
    #     mad_b4.use(sequence="lhcb2")
    #     mad_b4.twiss()
    #     ost.check_madx_lattices(mad_b1b2)

    # Build xsuite collider
    collider = xlhc.build_xsuite_collider(
        sequence_b1=mad_b1b2.sequence.lhcb1,
        sequence_b2=mad_b1b2.sequence.lhcb2,
        sequence_b4=mad_b4.sequence.lhcb2,
        beam_config=config_mad["beam_config"],
        enable_imperfections=config_mad["enable_imperfections"],
        enable_knob_synthesis=config_mad["enable_knob_synthesis"],
        rename_coupling_knobs=config_mad["rename_coupling_knobs"],
        pars_for_imperfections=config_mad["pars_for_imperfections"],
        ver_lhc_run=config_mad["ver_lhc_run"],
        ver_hllhc_optics=config_mad["ver_hllhc_optics"],
    )
    collider.build_trackers(_context=context)

    if sanity_checks:
        collider["lhcb1"].twiss(method="4d")
        collider["lhcb2"].twiss(method="4d")
    # Return collider
    return collider


def activate_RF_and_twiss(collider, config_mad, context, sanity_checks=True):
    # Define a RF system (values are not so immportant as they're defined later)
    print("--- Now Computing Twiss assuming:")
    if config_mad["ver_hllhc_optics"] == 1.6:
        dic_rf = {"vrf400": 16.0, "lagrf400.b1": 0.5, "lagrf400.b2": 0.5}
        for knob, val in dic_rf.items():
            print(f"    {knob} = {val}")
    elif config_mad["ver_lhc_run"] == 3.0:
        dic_rf = {"vrf400": 12.0, "lagrf400.b1": 0.5, "lagrf400.b2": 0.0}
        for knob, val in dic_rf.items():
            print(f"    {knob} = {val}")
    print("---")

    # Rebuild tracker if needed
    try:
        collider.build_trackers(_context=context)
    except:
        print("Skipping rebuilding tracker")

    for knob, val in dic_rf.items():
        collider.vars[knob] = val

    if sanity_checks:
        for my_line in ["lhcb1", "lhcb2"]:
            ost.check_xsuite_lattices(collider[my_line])

    return collider


def clean():
    # Remove all the temporaty files created in the process of building collider
    os.remove("mad_collider.log")
    os.remove("mad_b4.log")
    shutil.rmtree("temp")
    os.unlink("errors")
    os.unlink("acc-models-lhc")


# ==================================================================================================
# --- Main function for building distribution and collider
# ==================================================================================================
def build_distr_and_collider(config_file="config.yaml"):
    # Get configuration
    configuration, config_particles, config_mad = load_configuration(config_file)

    # Get context
    context = get_context(configuration)

    # Get sanity checks flag
    sanity_checks = configuration["sanity_checks"]

    # Tag start of the job
    tree_maker_tagging(configuration, tag="started")

    # Build particle distribution
    particle_list = build_particle_distribution(config_particles)

    # Write particle distribution to file
    write_particle_distribution(particle_list)
    
    # Get phasing from runIII
    ll_mu_b1_runIII_with_knob, ll_mu_b2_runIII_with_knob = load_RDT_runIII()

    # Build collider from mad model
    collider = build_collider_from_mad(config_mad, context, ll_mu_b1_runIII_with_knob, ll_mu_b2_runIII_with_knob, sanity_checks)

    # Twiss to ensure eveyrthing is ok
    collider = activate_RF_and_twiss(collider, config_mad, context, sanity_checks)

    # Clean temporary files
    clean()

    # Save collider to json
    os.makedirs("collider", exist_ok=True)
    collider.to_json("collider/collider.json")

    # Tag end of the job
    tree_maker_tagging(configuration, tag="completed")


# ==================================================================================================
# --- Script for execution
# ==================================================================================================

if __name__ == "__main__":
    build_distr_and_collider()
