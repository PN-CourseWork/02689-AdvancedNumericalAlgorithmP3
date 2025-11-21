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

import sys
from pathlib import Path

# Add project root (one level above 'src') to PYTHONPATH
project_root = Path(__file__).resolve().parents[2]   # goes up from Experiments/Quantities to project root
sys.path.append(str(project_root / "src"))
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
#print(f"ENERGY: {energy}")
#print(f"palinstrophy: {palinstropy}")
#print(f"enstrophy: {enstrophy}")

#TODO: ASKE Plot them as a function of iterations
# %%
# Plot conserved quantities vs iteration
# -------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_ldc_timeseries(solver, savepath: str | Path | None = None, show: bool = True):
    """
    Plot energy, enstrophy and palinstrophy as a function of iteration.

    Parameters
    ----------
    solver : solver instance
        Expects solver.time_series with attributes energy, enstrophy, palinstropy
        which are lists (or arrays) collected in the solve() loop.
    savepath : str | Path | None
        If provided, save the figure to this path.
    show : bool
        If True, call plt.show() after plotting.
    """
    # Create DataFrame directly from time_series
    df = pd.DataFrame({
        'Energy': solver.time_series.energy,
        'Enstrophy': solver.time_series.enstrophy,
        'Palinstropy': solver.time_series.palinstropy
    })

    # Melt to long format for seaborn
    df_long = df.reset_index().melt(id_vars='index', var_name='Quantity', value_name='Value')
    df_long.rename(columns={'index': 'Iteration'}, inplace=True)

    # Plot using FacetGrid
    g = sns.relplot(data=df_long, x='Iteration', y='Value', row='Quantity',
                    kind='line', facet_kws={'sharey': False})

    g.set_titles(row_template="{row_name}")
    g.fig.suptitle("Lid-driven cavity: conserved quantities vs iteration", y=1.00)

    if savepath is not None:
        g.savefig(savepath, dpi=200, bbox_inches='tight')
        print(f"Saved figure to {savepath}")

    return g.fig

# Example usage (place after solver.solve(...))
# --------------------------------------------
# solver.solve(tolerance=1e-5, max_iter=10000)
plot_ldc_timeseries(solver, savepath=data_dir / "ldc_quantities_vs_iteration.pdf")
