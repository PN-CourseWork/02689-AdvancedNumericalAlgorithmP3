#!/bin/bash
#BSUB -J optuna_corner_smoothing
#BSUB -q hpc
#BSUB -W 4:00
#BSUB -n 20
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/optuna_%J.out
#BSUB -e logs/optuna_%J.err

# Optuna corner_smoothing optimization for spectral solver
# Runs both FV L2 error and Botella vortex objectives
#
# Submit: bsub < scripts/optuna_corner_smoothing.sh

set -e
mkdir -p logs

echo "=== Optuna Corner Smoothing Optimization ==="
echo "Started: $(date)"
echo ""

# Run FV L2 error objective
echo "=== Objective 1: FV L2 Error ==="
uv run python main.py -m +experiment/optimization=corner_smoothing \
    optuna.objective=fv_l2_error \
    mlflow=coolify

echo ""

# Run Botella vortex objective
echo "=== Objective 2: Botella Vortex ==="
uv run python main.py -m +experiment/optimization=corner_smoothing \
    optuna.objective=botella_vortex \
    mlflow=coolify

echo ""
echo "=== Optimization Complete ==="
echo "Finished: $(date)"
