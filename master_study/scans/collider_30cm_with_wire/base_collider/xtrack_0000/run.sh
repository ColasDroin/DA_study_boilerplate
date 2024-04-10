#!/bin/bash
source /afs/cern.ch/work/c/cdroin/private/example_DA_study_runIII_wire/master_study/../activate_miniforge.sh
cd /afs/cern.ch/work/c/cdroin/private/example_DA_study_runIII_wire/master_study/scans/collider_30cm_with_wire/base_collider/xtrack_0000
python 2_configure_and_track.py > output_python.txt 2> error_python.txt
rm -rf final_* modules optics_repository optics_toolkit tools tracking_tools temp mad_collider.log __pycache__ twiss* errors fc* optics_orbit_at*
