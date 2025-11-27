"""
Lid-Driven Cavity Flow Computation - Chebyshev Basis
====================================================

Spectral solver using Chebyshev-Gauss-Lobatto nodes
as described in Zhang et al. (2010).

Usage:
    python compute_spectral_chebyshev.py --N 19 --Re 100
    python compute_spectral_chebyshev.py --N 31 --Re 400 --tol 1e-7 --max-iter 200000
"""

import argparse
import os

from ldc import SpectralSolver
from utils import get_project_root, LDCPlotter, GhiaValidator, plot_validation

# Parse command line arguments
parser = argparse.ArgumentParser(description="Spectral solver for lid-driven cavity (Chebyshev basis)")
parser.add_argument("--N", type=int, default=19, help="Polynomial order, nodes = N+1 (default: 19)")
parser.add_argument("--Re", type=int, default=100, help="Reynolds number (default: 100)")
parser.add_argument("--tol", type=float, default=1e-7, help="Convergence tolerance (default: 1e-6)")
parser.add_argument("--max-iter", type=int, default=200000, help="Max iterations (default: 100000)")
args = parser.parse_args()

N = args.N  # Polynomial order (nodes = N+1)
Re_number = args.Re
N_nodes = N + 1

# Setup directories
project_root = get_project_root()
data_dir = project_root / "data" / "Spectral-Solver" / "Chebyshev"
fig_dir = project_root / "figures" / "Spectral-Solver" / "Chebyshev"
data_dir.mkdir(parents=True, exist_ok=True)
fig_dir.mkdir(parents=True, exist_ok=True)

# Create solver
print("Initializing solver object")
solver = SpectralSolver(
    Re=Re_number,
    nx=N,
    ny=N,
    basis_type="chebyshev",
    CFL=0.70,
    beta_squared=5.0,
    corner_smoothing=0.15
)

print(f"Solver configured: Re={solver.params.Re}, Grid={N_nodes}x{N_nodes}, CFL={solver.params.CFL}")
print(f"Total nodes: {N_nodes * N_nodes}")

# MLflow setup with nested runs (auto-detect HPC via LSF env vars)
Re_str = f"Re{Re_number}"
is_hpc = "LSB_JOBID" in os.environ
experiment_name = "HPC-Spectral-Chebyshev" if is_hpc else "Spectral-Chebyshev"
solver.mlflow_start(experiment_name, f"N{N_nodes}_{Re_str}", parent_run_name=Re_str)

# Solve
solver.solve(tolerance=args.tol, max_iter=args.max_iter)

# Save solution
output_file = data_dir / f"LDC_N{N_nodes}_{Re_str}.h5"
solver.save(output_file)
solver.mlflow_log_artifact(str(output_file))
print(f"\nResults saved to: {output_file}")

# Generate and log plots
plotter = LDCPlotter(output_file)
validator = GhiaValidator(output_file, Re=Re_number, method_label='Spectral-Chebyshev')

fig_path = fig_dir / f"ghia_validation_N{N_nodes}_{Re_str}.pdf"
plot_validation(validator, output_path=fig_path)
solver.mlflow_log_artifact(str(fig_path))
print(f"  ✓ Ghia validation saved")

fig_path = fig_dir / f"convergence_N{N_nodes}_{Re_str}.pdf"
plotter.plot_convergence(output_path=fig_path)
solver.mlflow_log_artifact(str(fig_path))
print(f"  ✓ Convergence saved")

fig_path = fig_dir / f"fields_N{N_nodes}_{Re_str}.pdf"
plotter.plot_fields(output_path=fig_path)
solver.mlflow_log_artifact(str(fig_path))
print(f"  ✓ Fields saved")

fig_path = fig_dir / f"streamlines_N{N_nodes}_{Re_str}.pdf"
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
