#!/bin/bash
source /afs/cern.ch/work/c/cdroin/private/example_DA_study/master_study/../activate_miniforge.sh
cd /afs/cern.ch/work/c/cdroin/private/example_DA_study/master_study/scans/dynamic_collapse_new/base_collider/xtrack_0000/gen_3
python 3_dynamic_configure.py > output_python.txt 2> error_python.txt
rm -rf final_* modules optics_repository optics_toolkit tools tracking_tools temp mad_collider.log __pycache__ twiss* errors fc* optics_orbit_at*
