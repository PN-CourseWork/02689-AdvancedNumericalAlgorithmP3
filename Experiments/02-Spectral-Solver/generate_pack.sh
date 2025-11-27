#!/bin/bash
# Generate job pack file for spectral solver parameter sweep
# Usage: ./generate_pack.sh > jobs.pack
#        bsub -pack jobs.pack
# Or:    bsub -pack <(./generate_pack.sh)

# ===================
# Resource settings
# ===================
QUEUE="hpc"
WALLTIME="1:00"
CORES=4
MEMORY="8GB"

# ===================
# Parameter sweep
# ===================
N_VALUES=(11 15 19 23)
RE_VALUES=(100 400 1000)
#TODO: Askeeeeeeeeeeeeeeeeeee

# ===================
# Generate job lines
# ===================
for N in "${N_VALUES[@]}"; do
    for RE in "${RE_VALUES[@]}"; do
        echo "-J Spectral-N${N}-Re${RE} -q $QUEUE -W $WALLTIME -n $CORES -R \"rusage[mem=$MEMORY]\" uv run python Experiments/Spectral-Solver/compute_spectral_chebyshev.py --N $N --Re $RE"
    done
done
