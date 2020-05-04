#!/usr/bin/bash
[ -d "results" ] && rm -rf results
mkdir results && cd results
echo "localhost slots=10" > hostfile
export OMP_NUM_THREADS=1
mpirun --hostfile hostfile -n 6 python ../mixed_formulation.py