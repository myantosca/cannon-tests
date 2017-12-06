#!/bin/bash
#SBATCH -p RM
#SBATCH -t 30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node 1

export TS="$(date +%yy-%m-%d.%H.%M.%S)"
export DOUT="./results/matrixMulCUBLAS/k80/${TS}"

mkdir -p ${DOUT}

for ((d = 256; d < 65536; d *= 4)); do
    ./mkl_cblas_sgemm -m $d -q $d -n $d -s 30 1> ./mkl_cblas_sgemm.$d.csv 2> /dev/null
done
