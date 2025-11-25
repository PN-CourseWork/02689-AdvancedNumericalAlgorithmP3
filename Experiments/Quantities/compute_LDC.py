"""
Lid-Driven Cavity Flow Computation
===================================

This script computes the lid-driven cavity flow problem using a finite volume
method with the SIMPLE algorithm for pressure-velocity coupling on a collocated grid.
This version tracks conserved quantities (energy, enstrophy, palinstrophy) during iteration.

A small diagnostic block has been added right after the solver finishes to print
grid spacing, derivative operator presence, and a recomputed energy for debugging.
"""
# Make project src/ visible so `from ldc import FVSolver` works when running script directly.
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]   # climb from Experiments/Quantities/compute_LDC.py to project root
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# %%
# Problem Setup
# -------------
# Configure the solver with Reynolds number Re=100, grid resolution 16x16,
# and appropriate relaxation factors.

from ldc import FVSolver
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
data_dir = project_root / "data" / "Quantities"
data_dir.mkdir(parents=True, exist_ok=True)

solver = FVSolver(
    Re=1000.0,  # Reynolds number
    nx=255,  # Grid cells in x-direction
    ny=255,  # Grid cells in y-direction
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

# --------------------------------------------------
# DIAGNOSTIC BLOCK: prints spacing, ops, and quick energy
# --------------------------------------------------
print("\n--- DIAGNOSTIC CHECK ---")
# Try common attribute names: some solvers store nx/ny on config, others as attributes
print("dx_min:", getattr(solver, "dx_min", None))
print("dy_min:", getattr(solver, "dy_min", None))
# prefer solver.config if present
nx_val = getattr(solver, "nx", None) or getattr(getattr(solver, 'config', None), 'nx', None)
ny_val = getattr(solver, "ny", None) or getattr(getattr(solver, 'config', None), 'ny', None)
print("Nx (if present):", nx_val)
print("Ny (if present):", ny_val)
print("Has Dx/Dy?  Dx:", hasattr(solver, "Dx"), "Dy:", hasattr(solver, "Dy"))

# fields
try:
    print("u / v shapes:", solver.fields.u.shape, solver.fields.v.shape)
    print("u min/max:", solver.fields.u.min(), solver.fields.u.max())
    print("v min/max:", solver.fields.v.min(), solver.fields.v.max())
except Exception as e:
    print("Could not inspect fields:", e)

# recompute energy using best-effort dx/dy
import numpy as _np
u = getattr(solver.fields, 'u', None)
v = getattr(solver.fields, 'v', None)
if u is not None and v is not None:
    # infer Nx/Ny if missing
    Nx = nx_val
    Ny = ny_val
    if (Nx is None or Ny is None) and hasattr(solver.fields, 'grid_points'):
        pts = solver.fields.grid_points
        try:
            Ncells = pts.shape[0]
            N = int(_np.sqrt(Ncells))
            Nx = Nx or N
            Ny = Ny or N
        except Exception:
            pass

    dx = getattr(solver, 'dx_min', None)
    dy = getattr(solver, 'dy_min', None)
    if dx is None or dy is None:
        # try inference from Nx/Ny
        if Nx and Ny:
            dx = dx or (1.0 / Nx)
            dy = dy or (1.0 / Ny)
    dx = dx if dx is not None else 1.0
    dy = dy if dy is not None else 1.0
    print("Inferred grid:", Nx, Ny)
    print("Using dx,dy:", dx, dy)
    dA = float(dx) * float(dy)
    energy_direct = 0.5 * (float(_np.dot(u, u)) + float(_np.dot(v, v))) * dA
    print("Energy_direct (using current dx,dy):", energy_direct)
else:
    print("Fields u/v not available to recompute energy.")
print("--- END DIAGNOSTIC CHECK ---\n")

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
# Export the complete solution (velocity, pressure fields, metadata, and time series) to HDF5.

output_file = data_dir / "LDC_Re100_quantities.h5"
solver.save(output_file)

print(f"\nResults saved to: {output_file}")
