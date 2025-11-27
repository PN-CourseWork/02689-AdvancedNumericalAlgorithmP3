"""
Lid-Driven Cavity Flow - Finite Volume SIMPLE
==============================================

Finite volume solver using the SIMPLE algorithm for pressure-velocity coupling
on a collocated grid with Rhie-Chow interpolation.

Usage::

    uv run python compute_LDC.py --N 32 --Re 100
    uv run python compute_LDC.py --N 64 --Re 400 --tol 1e-6
"""

# %%
# Setup and Configuration
# -----------------------
# Parse command line arguments and setup directories.

import argparse
import os

from ldc import FVSolver
from utils import get_project_root, LDCPlotter, GhiaValidator, plot_validation

parser = argparse.ArgumentParser(description="FV-SIMPLE solver for lid-driven cavity")
parser.add_argument("--N", type=int, default=32, help="Grid cells in each direction (default: 32)")
parser.add_argument("--Re", type=int, default=100, help="Reynolds number (default: 100)")
parser.add_argument("--tol", type=float, default=1e-7, help="Convergence tolerance (default: 1e-7)")
parser.add_argument("--max-iter", type=int, default=50000, help="Max iterations (default: 50000)")
args = parser.parse_args()

N = args.N
Re_number = args.Re

project_root = get_project_root()
data_dir = project_root / "data" / "FV-Solver"
fig_dir = project_root / "figures" / "FV-Solver"
data_dir.mkdir(parents=True, exist_ok=True)
fig_dir.mkdir(parents=True, exist_ok=True)

# %%
# Initialize Solver
# -----------------
# Create the finite volume solver with SIMPLE algorithm.

solver = FVSolver(
    Re=Re_number,
    nx=N,
    ny=N,
    alpha_uv=0.6,
    alpha_p=0.3,
    convection_scheme="TVD",
    limiter="MUSCL",
    linear_solver_tol=1e-8,
)

print(f"Solver configured: Re={solver.params.Re}, Grid={solver.params.nx}x{solver.params.ny}")
print(f"  Convection scheme: {solver.params.convection_scheme} with {solver.params.limiter} limiter")
print(f"  Linear solver: PETSc (BiCGSTAB + GAMG), tol={solver.params.linear_solver_tol:.0e}")

# %%
# MLflow Tracking
# ---------------
# Setup MLflow experiment tracking with nested runs.

Re_str = f"Re{Re_number}"
is_hpc = "LSB_JOBID" in os.environ
experiment_name = "HPC-FV-Solver" if is_hpc else "FV-Solver"
solver.mlflow_start(experiment_name, f"N{N}_{Re_str}", parent_run_name=Re_str)

# %%
# Solve
# -----
# Run the solver until convergence or max iterations.

solver.solve(tolerance=args.tol, max_iter=args.max_iter)

# %%
# Save Results
# ------------
# Save solution to HDF5 and log as MLflow artifact.

output_file = data_dir / f"LDC_N{N}_{Re_str}.h5"
solver.save(output_file)
solver.mlflow_log_artifact(str(output_file))
print(f"\nResults saved to: {output_file}")

# %%
# Validation Plots
# ----------------
# Generate plots and log to MLflow.

plotter = LDCPlotter(output_file)
validator = GhiaValidator(output_file, Re=Re_number, method_label='FV-SIMPLE')

fig_path = fig_dir / f"ghia_validation_N{N}_{Re_str}.pdf"
plot_validation(validator, output_path=fig_path)
solver.mlflow_log_artifact(str(fig_path))
print(f"  ✓ Ghia validation saved")

fig_path = fig_dir / f"convergence_N{N}_{Re_str}.pdf"
plotter.plot_convergence(output_path=fig_path)
solver.mlflow_log_artifact(str(fig_path))
print(f"  ✓ Convergence saved")

fig_path = fig_dir / f"fields_N{N}_{Re_str}.pdf"
plotter.plot_fields(output_path=fig_path)
solver.mlflow_log_artifact(str(fig_path))
print(f"  ✓ Fields saved")

fig_path = fig_dir / f"streamlines_N{N}_{Re_str}.pdf"
plotter.plot_streamlines(output_path=fig_path)
solver.mlflow_log_artifact(str(fig_path))
print(f"  ✓ Streamlines saved")

# %%
# Summary
# -------
# End MLflow run and print summary.

solver.mlflow_end()

print(f"\nSolution Status:")
print(f"  Converged: {solver.metrics.converged}")
print(f"  Iterations: {solver.metrics.iterations}")
print(f"  Final residual: {solver.metrics.final_residual:.6e}")
print(f"  Wall time: {solver.metrics.wall_time_seconds:.2f} seconds")
