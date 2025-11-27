#!/bin/bash
# Generate job pack file for FV solver parameter sweep
# Usage: ./generate_pack.sh > jobs.pack
#        bsub -pack jobs.pack
# Or:    bsub -pack <(./generate_pack.sh)

# ===================
# Resource settings
# ===================
QUEUE="hpc"
WALLTIME="2:00"
CORES=4
MEMORY="8GB"

# ===================
# Parameter sweep
# ===================
N_VALUES=(32 64 128 256)
RE_VALUES=(100 400 1000)
#TODO: Aske her kan du gøre det...

# ===================
# Generate job lines
# ===================
for N in "${N_VALUES[@]}"; do
    for RE in "${RE_VALUES[@]}"; do
        echo "-J FV-N${N}-Re${RE} -q $QUEUE -W $WALLTIME -n $CORES -R \"rusage[mem=$MEMORY]\" uv run python Experiments/FV-Solver/compute_LDC.py --N $N --Re $RE"
    done
done
