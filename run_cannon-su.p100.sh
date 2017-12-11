#!/bin/bash
#SBATCH -p GPU
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:p100:2

export TS="$(date +%yy-%m-%d.%H.%M.%S)"
export DOUT="./results/cannon-su/p100/${TS}"

mkdir -p ${DOUT}

for ((p = 1; p <= 256; p *=4)); do
    for ((d = 256; d <= 16384; d *= 4)); do
	./cannon-su -p $p -m $d -q $d -n $d -s 10 1> ${DOUT}/cannon-su.p100.$d.csv 2> /dev/null
    done
done
