#!/bin/bash
#SBATCH -p RM-gpu
#SBATCH -t 30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node 1

for ((d = 256; d < 65536; d *= 4)); do
    ./matrixMulCUBLAS m=$d q=$d n=$d
done
