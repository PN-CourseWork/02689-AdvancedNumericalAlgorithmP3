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
    Re=100.0,  # Reynolds number
    nx=16,  # Grid cells in x-direction
    ny=16,  # Grid cells in y-direction
    alpha_uv=0.7,  # Velocity under-relaxation factor
    alpha_p=0.3,  # Pressure under-relaxation factor
    convection_scheme="TVD",
)

print(
    f"Solver configured: Re={solver.config.Re}, Grid={solver.config.nx}x{solver.config.ny}"
)

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
# Get timeseries data
# -------------
energy = solver.time_series.energy
palinstropy = solver.time_series.palinstropy
enstrophy = solver.time_series.enstrophy
print(f"ENERGY: {energy}")
print(f"palinstrophy: {palinstropy}")
print(f"enstrophy: {enstrophy}")

