#!/usr/bin/env python3
"""
Lid-Driven Cavity Flow Computation
===================================

This script computes the lid-driven cavity flow problem using a finite volume
method with the SIMPLE algorithm for pressure-velocity coupling.

The problem is solved on a collocated grid with:
- Reynolds number: Re = 100
- Grid resolution: 64x64 cells
- Relaxation factors: α_uv = 0.6, α_p = 0.2
"""

# %% Imports and Setup
# Import the finite volume solver and utility functions
from ldc import FVSolver
from utils import get_project_root

# Setup data output directory
project_root = get_project_root()
data_dir = project_root / "data" / "FV-Solver"
data_dir.mkdir(parents=True, exist_ok=True)

# %% Configure Solver
# Initialize the FV solver with problem parameters
solver = FVSolver(
    Re=100.0,       # Reynolds number
    nx=64,          # Grid cells in x-direction
    ny=64,          # Grid cells in y-direction
    alpha_uv=0.6,   # Velocity under-relaxation factor
    alpha_p=0.2     # Pressure under-relaxation factor
)

print(f"Solver configured: Re={solver.Re}, Grid={solver.nx}x{solver.ny}")

# %% Solve the System
# Run the SIMPLE iteration until convergence
solver.solve(tolerance=1e-5, max_iter=500)

# %% Display Results
# Print convergence information
print(f"\nSolution Status:")
print(f"  Converged: {solver.metadata.converged}")
print(f"  Iterations: {solver.metadata.iterations}")
print(f"  Final residual: {solver.metadata.final_residual:.6e}")

# %% Save Results to HDF5
# Save the complete solution (fields + metadata) to HDF5 format
output_file = data_dir / "LDC_Re100.h5"
solver.save(output_file)

print(f"\nResults saved to: {output_file}")
