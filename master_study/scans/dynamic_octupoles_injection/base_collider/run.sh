#!/bin/bash
source /storage-hpc/gpfs_data/HPC/home_recovery/cdroin/dynamic_collapse/master_study/../activate_miniforge.sh
cd /storage-hpc/gpfs_data/HPC/home_recovery/cdroin/dynamic_collapse/master_study/scans/dynamic_octupoles_injection/base_collider
python 1_build_collider.py > output_python.txt 2> error_python.txt
rm -rf final_* modules optics_repository optics_toolkit tools tracking_tools temp mad_collider.log __pycache__ twiss* errors fc* optics_orbit_at*
