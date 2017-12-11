#!/bin/bash
#SBATCH -p RM
#SBATCH -t 04:00:00
#SBATCH -N 12
#SBATCH --ntasks-per-node 24

export TS="$(date +%Y-%m-%d.%H.%M.%S)"
export DOUT="./results/cannon-so/${TS}"

mkdir -p ${DOUT}

for ((p = 256; p >=1; p /=4)); do
    for ((d = 256; d <= 16384; d *= 4)); do
	mpirun -np $p ./cannon-so -m $d -q $d -n $d -s 10 1> ${DOUT}/cannon-so.$p.$d.csv 2> /dev/null
    done
done
