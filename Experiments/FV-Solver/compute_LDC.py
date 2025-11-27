"""
Lid-Driven Cavity Flow Computation
===================================

This script computes the lid-driven cavity flow problem using a finite volume
method with the SIMPLE algorithm for pressure-velocity coupling on a collocated grid.

Usage:
    python compute_LDC.py --N 32 --Re 100
    python compute_LDC.py --N 64 --Re 400 --tol 1e-6 --max-iter 100000
"""

import argparse
import os

from ldc import FVSolver
from utils import get_project_root, LDCPlotter, GhiaValidator, plot_validation

# Parse command line arguments
parser = argparse.ArgumentParser(description="FV-SIMPLE solver for lid-driven cavity")
parser.add_argument("--N", type=int, default=32, help="Grid cells in each direction (default: 32)")
parser.add_argument("--Re", type=int, default=100, help="Reynolds number (default: 100)")
parser.add_argument("--tol", type=float, default=1e-7, help="Convergence tolerance (default: 1e-5)")
parser.add_argument("--max-iter", type=int, default=50000, help="Max iterations (default: 50000)")
args = parser.parse_args()

N = args.N
Re_number = args.Re

# Setup directories
project_root = get_project_root()
data_dir = project_root / "data" / "FV-Solver"
fig_dir = project_root / "figures" / "FV-Solver"
data_dir.mkdir(parents=True, exist_ok=True)
fig_dir.mkdir(parents=True, exist_ok=True)

# Create solver
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

# MLflow setup with nested runs (auto-detect HPC via LSF env vars)
Re_str = f"Re{Re_number}"
is_hpc = "LSB_JOBID" in os.environ
experiment_name = "HPC-FV-Solver" if is_hpc else "FV-Solver"
solver.mlflow_start(experiment_name, f"N{N}_{Re_str}", parent_run_name=Re_str)

# Solve
solver.solve(tolerance=args.tol, max_iter=args.max_iter)

# Save solution
output_file = data_dir / f"LDC_N{N}_{Re_str}.h5"
solver.save(output_file)
solver.mlflow_log_artifact(str(output_file))
print(f"\nResults saved to: {output_file}")

# Generate and log plots
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

# End MLflow run
solver.mlflow_end()

# Summary
print(f"\nSolution Status:")
print(f"  Converged: {solver.metrics.converged}")
print(f"  Iterations: {solver.metrics.iterations}")
print(f"  Final residual: {solver.metrics.final_residual:.6e}")
print(f"  Wall time: {solver.metrics.wall_time_seconds:.2f} seconds")
