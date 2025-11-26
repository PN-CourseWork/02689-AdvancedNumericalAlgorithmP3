"""
Lid-Driven Cavity Flow Computation
===================================

This script computes the lid-driven cavity flow problem using a finite volume
method with the SIMPLE algorithm for pressure-velocity coupling on a collocated grid.
This version tracks conserved quantities (energy, enstrophy, palinstrophy) during iteration.

A small diagnostic block has been added right after the solver finishes to print
grid spacing, derivative operator presence, solver scaling, and robust recomputed
energy/enstrophy/palinstrophy for debugging.
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
    Re=400.0,  # Reynolds number
    nx=128,  # Grid cells in x-direction
    ny=128,  # Grid cells in y-direction
    alpha_uv=0.5,  # Velocity under-relaxation factor
    alpha_p=0.5,  # Pressure under-relaxation factor
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
# DIAGNOSTIC BLOCK: prints spacing, ops, and robust recomputation
# --------------------------------------------------
print("\n--- DIAGNOSTIC CHECK ---")

# Basic solver attributes and scaling/nondimensionalization info
print("Solver config (subset):")
try:
    print("  Re:", solver.config.Re)
    print("  Lx, Ly:", getattr(solver.config, "Lx", None), getattr(solver.config, "Ly", None))
    print("  lid_velocity:", getattr(solver.config, "lid_velocity", None))
except Exception:
    print("  (could not access solver.config attributes)")

print("Solver attributes:")
print("  rho:", getattr(solver, "rho", None))
print("  mu:", getattr(solver, "mu", None))
# implied kinematic viscosity if mu and rho present
try:
    mu_val = float(getattr(solver, "mu", float("nan")))
    rho_val = float(getattr(solver, "rho", 1.0))
    print("  implied nu = mu / rho =", mu_val / rho_val)
except Exception:
    pass

# Try common attribute names: some solvers store nx/ny on config, others as attributes
print("dx_min:", getattr(solver, "dx_min", None))
print("dy_min:", getattr(solver, "dy_min", None))
nx_val = getattr(solver, "nx", None) or getattr(getattr(solver, 'config', None), 'nx', None)
ny_val = getattr(solver, "ny", None) or getattr(getattr(solver, 'config', None), 'ny', None)
print("Nx (if present):", nx_val)
print("Ny (if present):", ny_val)
print("Has Dx/Dy?  Dx:", hasattr(solver, "Dx"), "Dy:", hasattr(solver, "Dy"))

# fields basic inspection
try:
    print("u / v shapes:", solver.fields.u.shape, solver.fields.v.shape)
    print("u min/max:", solver.fields.u.min(), solver.fields.u.max())
    print("v min/max:", solver.fields.v.min(), solver.fields.v.max())
except Exception as e:
    print("Could not inspect fields:", e)

# If mesh object available, inspect cell volumes and grid points
mesh = getattr(solver, "mesh", None)
if mesh is not None:
    try:
        if hasattr(mesh, "cell_volumes"):
            cv = mesh.cell_volumes
            import numpy as _np
            print("mesh.cell_volumes: min, mean, max =", float(_np.min(cv)), float(_np.mean(cv)), float(_np.max(cv)))
        if hasattr(mesh, "cell_centers"):
            pts_sample = mesh.cell_centers[:8]
            print("mesh.cell_centers (first 8):", pts_sample)
    except Exception as e:
        print("Could not inspect mesh:", e)
else:
    print("solver.mesh not present")

# recompute energy / enstrophy / palinstrophy robustly using solver.fields (best-effort)
import numpy as _np

u = getattr(solver.fields, 'u', None)
v = getattr(solver.fields, 'v', None)
pts = getattr(solver.fields, 'grid_points', None)

if u is None or v is None:
    print("Fields u/v not available to recompute diagnostics.")
else:
    # Quick direct-energy using best-effort dx/dy (keeps original printed value for comparison)
    dx = getattr(solver, 'dx_min', None)
    dy = getattr(solver, 'dy_min', None)
    Nx = nx_val
    Ny = ny_val
    if (dx is None or dy is None) and (Nx and Ny):
        dx = dx or (1.0 / Nx)
        dy = dy or (1.0 / Ny)
    dx = dx if dx is not None else 1.0
    dy = dy if dy is not None else 1.0
    dA = float(dx) * float(dy)
    energy_direct = 0.5 * (float(_np.dot(u, u)) + float(_np.dot(v, v))) * dA
    print("\n-- Quick direct energy (dot-product) --")
    print("Inferred grid:", Nx, Ny)
    print("Using dx,dy:", dx, dy)
    print("Energy_direct (using current dx,dy):", energy_direct)

    # Now a robust structured-grid mapping and gradient-based recompute
    print("\n-- Robust structured-grid recomputation (uses grid_points if available) --")
    try:
        if pts is None:
            # try mesh.cell_centers as a fallback
            if mesh is not None and hasattr(mesh, "cell_centers"):
                pts = mesh.cell_centers.copy()
            else:
                raise RuntimeError("No grid_points or mesh.cell_centers available for structured mapping.")

        pts = _np.asarray(pts)
        if pts.ndim != 2 or pts.shape[1] < 2:
            raise RuntimeError("grid_points has unexpected shape")

        xs = _np.sort(_np.unique(pts[:, 0]))
        ys = _np.sort(_np.unique(pts[:, 1]))
        Nx = len(xs); Ny = len(ys)
        if Nx < 2 or Ny < 2:
            raise RuntimeError("Not enough unique coordinates to form structured grid")

        # build structured arrays by explicit coordinate -> index mapping (tolerant to float error)
        def nearest_index(array, value):
            return int(_np.argmin(_np.abs(array - value)))

        U = _np.full((Ny, Nx), _np.nan, dtype=float)
        V = _np.full((Ny, Nx), _np.nan, dtype=float)

        # try pull cell_volumes from mesh first, else None
        cell_volumes = None
        if mesh is not None and hasattr(mesh, "cell_volumes"):
            try:
                cell_volumes = _np.asarray(mesh.cell_volumes)
            except Exception:
                cell_volumes = None

        CV2 = None
        if cell_volumes is not None:
            CV2 = _np.full((Ny, Nx), _np.nan, dtype=float)

        # fill arrays
        for k, (x, y) in enumerate(pts):
            ix = nearest_index(xs, x)
            iy = nearest_index(ys, y)
            U[iy, ix] = float(u[k])
            V[iy, ix] = float(v[k])
            if CV2 is not None:
                CV2[iy, ix] = float(cell_volumes[k])

        # check for NaNs (mapping problem)
        nan_count = int(_np.isnan(U).sum() + _np.isnan(V).sum())
        if nan_count > 0:
            print(f"WARNING: Mapping produced {nan_count} NaNs. The grid_points ordering may not match fields ordering.")
            # print a small sample to help debugging
            print("Sample pts[:8]:", pts[:8])
        else:
            # compute spacing (assume uniform structured)
            dx_grid = float(xs[1] - xs[0])
            dy_grid = float(ys[1] - ys[0])
            print("Structured grid inferred: Nx,Ny =", Nx, Ny, "dx,dy =", dx_grid, dy_grid)

            # use CV2 if available, else uniform area
            if CV2 is not None and not _np.isnan(CV2).any():
                cell_vol_2d = CV2
                print("Using mesh.cell_volumes for area weighting.")
            else:
                cell_vol_2d = _np.full_like(U, dx_grid * dy_grid, dtype=float)
                print("No cell_volumes found; using uniform dx*dy for area weighting.")

            # energy (grid-based)
            energy_grid = 0.5 * _np.sum((U * U + V * V) * cell_vol_2d)

            # enstrophy: use np.gradient (handles boundaries with one-sided differences)
            dU_dy = _np.gradient(U, dy_grid, axis=0)
            dV_dx = _np.gradient(V, dx_grid, axis=1)
            omega = dV_dx - dU_dy
            enstrophy_grid = 0.5 * _np.sum((omega * omega) * cell_vol_2d)

            # palinstrophy: gradients of omega
            domega_dy = _np.gradient(omega, dy_grid, axis=0)
            domega_dx = _np.gradient(omega, dx_grid, axis=1)
            palinstrophy_grid = _np.sum((domega_dx * domega_dx + domega_dy * domega_dy) * cell_vol_2d)

            print("\nRecomputed (structured-grid) diagnostics:")
            print(f"  Energy (grid)     = {energy_grid:.6e}")
            print(f"  Enstrophy (grid)  = {enstrophy_grid:.6e}")
            print(f"  Palinstrophy(grid)= {palinstrophy_grid:.6e}")

    except Exception as e:
        print("Could not perform robust recomputation:", e)

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