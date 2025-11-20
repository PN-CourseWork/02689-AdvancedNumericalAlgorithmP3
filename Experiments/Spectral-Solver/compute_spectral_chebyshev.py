"""
Lid-Driven Cavity Flow Computation - Chebyshev Basis
====================================================

Test the spectral solver using Chebyshev-Gauss-Lobatto nodes
as described in Zhang et al. (2010).
"""

# %%
# Problem Setup
# -------------
# Configure the spectral solver with Chebyshev basis

from ldc import SpectralSolver
from utils import get_project_root

project_root = get_project_root()
data_dir = project_root / "data" / "Spectral-Solver"
data_dir.mkdir(parents=True, exist_ok=True)

solver = SpectralSolver(
    Re=100.0,            # Reynolds number
    Nx=15,               # Polynomial order in x (nodes = Nx+1 = 16)
    Ny=15,               # Polynomial order in y (nodes = Ny+1 = 16)
    basis_type="chebyshev",  # Use Chebyshev-Gauss-Lobatto (Zhang et al. 2010)
    CFL=0.70,            # CFL number for adaptive time stepping
    beta_squared=5.0,    # Artificial compressibility coefficient
    corner_smoothing=0.15 # Lid velocity smoothing near corners
)

print(f"Solver configured: Re={solver.config.Re}, Grid={(solver.config.Nx+1)}x{(solver.config.Ny+1)}, CFL={solver.config.CFL}")
print(f"Total nodes: {(solver.config.Nx+1)*(solver.config.Ny+1)}")

# %%
# Run Pseudo Time-Stepping
# -------------------------

solver.solve(tolerance=1e-5, max_iter=20000)

# %%
# Convergence Results
# -------------------

print(f"\nSolution Status:")
print(f"  Converged: {solver.config.converged}")
if solver.config.iterations is not None:
    print(f"  Iterations: {solver.config.iterations}")
if solver.config.final_residual is not None:
    print(f"  Final residual: {solver.config.final_residual:.6e}")

# %%
# Save Solution
# -------------

output_file = data_dir / "LDC_Spectral_Chebyshev_Re100.h5"
solver.save(output_file)

print(f"\nResults saved to: {output_file}")
