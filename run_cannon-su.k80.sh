#!/bin/bash
#SBATCH -p GPU
#SBATCH -t 08:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:k80:4

export TS="$(date +%Y-%m-%d.%H.%M.%S)"
export DOUT="./results/cannon-su/k80/${TS}"

mkdir -p ${DOUT}

for ((d = 256; d <= 16384; d *= 4)); do
    for ((p = 256; p >= 1; p /=4)); do
	./cannon-su -p $p -m $d -q $d -n $d -s 10 1> ${DOUT}/cannon-su.k80.$p.$d.csv 2> ${DOUT}/cannon-su.k80.$p.$d.err
    done
done
