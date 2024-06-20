#!/bin/bash
source /storage-hpc/gpfs_data/HPC/home_recovery/cdroin/runIII_ions_personal_bologna/master_study/../activate_miniforge.sh
cd /home/HPC/cdroin/runIII_ions_personal_bologna/master_study/
python 002_chronjob.py > output_python.txt 2> error_python.txt
python 002_chronjob_alt.py > output_python_alt.txt 2> error_python_alt.txt