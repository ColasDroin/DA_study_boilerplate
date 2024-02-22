#!/bin/bash
source /afs/cern.ch/work/c/cdroin/private/example_DA_study_runIII_ions/master_study/../activate_miniforge.sh
cd /afs/cern.ch/work/c/cdroin/private/example_DA_study_runIII_ions/master_study/scans/MD2024_clean/base_collider
python 1_build_distr_and_collider.py > output_python.txt 2> error_python.txt
rm -rf final_* modules optics_repository optics_toolkit tools tracking_tools temp mad_collider.log __pycache__ twiss* errors fc* optics_orbit_at*
