"""
Lid-Driven Cavity Flow Computation - Spectral Method
=====================================================

This script computes the lid-driven cavity flow problem using a pseudospectral
method with Legendre-Gauss-Lobatto nodes and RK4 time integration.
"""

# %%
# Problem Setup
# -------------
# Configure the spectral solver with Reynolds number Re=100, polynomial order Nx=Ny=16.

from ldc import SpectralSolver
from utils import get_project_root

project_root = get_project_root()
data_dir = project_root / "data" / "Spectral-Solver"
data_dir.mkdir(parents=True, exist_ok=True)

solver = SpectralSolver(
    Re=100.0,            # Reynolds number
    Nx=15,               # Polynomial order in x (nodes = Nx+1 = 16)
    Ny=15,               # Polynomial order in y (nodes = Ny+1 = 16)
    basis_type="legendre",  # Use Legendre-Gauss-Lobatto basis
    CFL=0.70,            # CFL number for adaptive time stepping
    beta_squared=5.0,    # Artificial compressibility coefficient
    corner_smoothing=0.15 # Lid velocity smoothing near corners
)

print(f"Solver configured: Re={solver.config.Re}, Grid={(solver.config.Nx+1)}x{(solver.config.Ny+1)}, CFL={solver.config.CFL}")
print(f"Total nodes: {(solver.config.Nx+1)*(solver.config.Ny+1)}")

# %%
# Run Pseudo Time-Stepping
# -------------------------
# Solve the incompressible Navier-Stokes equations using RK4 with artificial compressibility.
# Note: Explicit RK4 requires small CFL (~0.1) for stability at Re=100

solver.solve(tolerance=1e-5, max_iter=20000)

# %%
# Convergence Results
# -------------------
# Display convergence statistics from the pseudo time-stepping.

print(f"\nSolution Status:")
print(f"  Converged: {solver.config.converged}")
if solver.config.iterations is not None:
    print(f"  Iterations: {solver.config.iterations}")
if solver.config.final_residual is not None:
    print(f"  Final residual: {solver.config.final_residual:.6e}")

# %%
# Save Solution
# -------------
# Export the complete solution (velocity, pressure fields, and metadata) to HDF5.

output_file = data_dir / "LDC_Spectral_Re100.h5"
solver.save(output_file)

print(f"\nResults saved to: {output_file}")
