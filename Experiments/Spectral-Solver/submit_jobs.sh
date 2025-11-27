#!/bin/bash
#BSUB -J Spectral-LDC[1-9]
#BSUB -q hpc
#BSUB -W 1:00
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -o logs/Spectral_%J_%I.out
#BSUB -e logs/Spectral_%J_%I.err

# Spectral-Solver job array for lid-driven cavity
# Submits jobs for multiple polynomial orders N (nodes = N+1)

# Polynomial orders to run 
N_VALUES=(11 13 15 19 23 25 27 29 31)
RE=100

# Get N for this array task
N=${N_VALUES[$((LSB_JOBINDEX-1))]}

echo "=========================================="
echo "Spectral-Solver: N=$N (${N}+1 nodes), Re=$RE"
echo "=========================================="

module load petsc
uv sync

uv run python Experiments/Spectral-Solver/compute_spectral_chebyshev.py --N $N --Re $RE

echo "Done!"
