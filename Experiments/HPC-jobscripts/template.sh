#!/bin/bash
#BSUB -J template
#BSUB -o Experiments/HPC-jobscripts/logs/template_%J.out
#BSUB -e Experiments/HPC-jobscripts/logs/template_%J.err
#BSUB -n 1
#BSUB -W 00:20
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=5GB]"

module load python3/3.11.1
module load petsc

uv sync
uv run python Experiments/Spectral-Solver/compute_spectral_chebyshev.py
