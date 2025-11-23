"""
Lid-Driven Cavity Flow Computation
===================================

This script computes the lid-driven cavity flow problem using a finite volume
method with the SIMPLE algorithm for pressure-velocity coupling on a collocated grid.
"""

# %%
# Problem Setup
# -------------
# Configure the solver with Reynolds number Re=100, grid resolution 64x64,
# and appropriate relaxation factors.

from ldc import FVSolver
from utils import get_project_root

project_root = get_project_root()
data_dir = project_root / "data" / "FV-Solver"
data_dir.mkdir(parents=True, exist_ok=True)

solver = FVSolver(
    Re=100.0,       # Reynolds number
    nx=32,          # Grid cells in x-direction
    ny=32,          # Grid cells in y-direction
    alpha_uv=0.6,   # Velocity under-relaxation factor
    alpha_p=0.2     # Pressure under-relaxation factor
)

print(f"Solver configured: Re={solver.config.Re}, Grid={solver.config.nx}x{solver.config.ny}")

# %%
# Run SIMPLE Iteration
# --------------------
# Solve the incompressible Navier-Stokes equations using the SIMPLE algorithm.

solver.solve(tolerance=1e-5, max_iter=10000)

# %%
# Convergence Results
# -------------------
# Display convergence statistics from the SIMPLE iteration.

print("\nSolution Status:")
print(f"  Converged: {solver.metadata.converged}")
print(f"  Iterations: {solver.metadata.iterations}")
print(f"  Final residual: {solver.metadata.final_residual:.6e}")

# %%
# Save Solution
# -------------
# Export the complete solution (velocity, pressure fields, and metadata) to HDF5.

N = solver.config.nx  # Number of grid cells
Re = int(solver.config.Re)
output_file = data_dir / f"LDC_N{N}_Re{Re}.h5"
solver.save(output_file)

print(f"\nResults saved to: {output_file}")
