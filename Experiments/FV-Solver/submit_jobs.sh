#!/bin/bash
#BSUB -J FV-LDC[1-2]
#BSUB -q hpc
#BSUB -W 2:00
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -o logs/FV_%J_%I.out
#BSUB -e logs/FV_%J_%I.err

# FV-Solver job array for lid-driven cavity
# Submits jobs for multiple grid sizes N

# Grid sizes to run (array index 1-4)
N_VALUES=(64 128)
RE=100

# Get N for this array task
N=${N_VALUES[$((LSB_JOBINDEX-1))]}

echo "=========================================="
echo "FV-Solver: N=$N, Re=$RE"
echo "=========================================="

module load petsc
uv sync

uv run python Experiments/FV-Solver/compute_LDC.py --N $N --Re $RE

echo "Done!"
