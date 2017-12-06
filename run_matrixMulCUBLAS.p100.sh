#!/bin/bash
#SBATCH -p GPU
#SBATCH -t 30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:p100:2

export TS="$(date +%yy-%m-%d.%H.%M.%S)"
export DOUT="./results/matrixMulCUBLAS/p100/${TS}"

mkdir -p ${DOUT}
for ((d = 256; d <= 65536; d *= 4)); do
    ./matrixMulCUBLAS m=$d q=$d n=$d 1> ${DOUT}/matrixMulCUBLAS.p100.$d.csv 2> /dev/null
done
