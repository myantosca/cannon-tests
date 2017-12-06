#!/bin/bash
#SBATCH -p GPU
#SBATCH -t 30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:k80:1

export TS="$(date +%yy-%m-%d.%H.%M.%S)"
export DOUT="./results/matrixMulCUBLAS/k80/${TS}"

mkdir -p ${DOUT}
for ((d = 256; d < 65536; d *= 4)); do
    ./matrixMulCUBLAS m=$d q=$d n=$d 1> ./matrixMulCUBLAS.k80.$d.csv 2> /dev/null
done
