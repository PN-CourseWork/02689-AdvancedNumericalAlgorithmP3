#!/bin/bash
# Setup script for ANA Project 3 with PETSc support
# This script creates a clean virtual environment with petsc4py installed

set -e  # Exit on error

echo "=========================================="
echo "Setting up ANA Project 3 Environment"
echo "=========================================="

# Remove existing venv
if [ -d ".venv" ]; then
    echo "Removing existing .venv..."
    rm -rf .venv
fi

# Remove stray pip directory if it exists
if [ -d "pip" ]; then
    echo "Removing stray pip directory..."
    rm -rf pip
fi

# Create fresh uv venv with pip included
echo "Creating fresh virtual environment with uv..."
uv venv --seed

# First sync with uv to install all dependencies
echo "Running uv sync..."
uv sync

# Then install petsc4py via pip AFTER uv sync
# (so it doesn't get removed by uv sync)
echo "Installing petsc4py with pip..."
source .venv/bin/activate
pip install petsc4py

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To verify PETSc installation, run:"
echo "  python -c 'from petsc4py import PETSc; print(\"PETSc version:\", PETSc.Sys.getVersion())'"
echo ""
