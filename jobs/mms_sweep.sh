#!/bin/bash
#BSUB -J mms_sweep
#BSUB -q hpc
#BSUB -W 4:00
#BSUB -n 16
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/mms_sweep_%J.out
#BSUB -e logs/mms_sweep_%J.err

# MMS (Method of Manufactured Solutions) validation sweep
# Tests spectral convergence across multiple N values and Reynolds numbers
#
# Usage:
#   bsub < jobs/mms_sweep.sh

echo "============================================"
echo "MMS Validation Sweep"
echo "============================================"
echo "Job ID: $LSB_JOBID"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "============================================"

# Create logs directory if needed
mkdir -p logs

# Change to project directory
cd $HOME/ANA-P3 || { echo "Failed to cd to project dir"; exit 1; }

# Run the MMS sweep (parameters from conf/experiment/validation/mms/spectral.yaml)
echo ""
echo "Starting MMS sweep..."
echo ""

uv run python scripts/test_sg_mms.py -m

echo ""
echo "============================================"
echo "MMS sweep completed at $(date)"
echo "============================================"

# List generated figures
echo ""
echo "Generated figures:"
ls -la figures/mms_*.pdf 2>/dev/null || echo "No figures found"
