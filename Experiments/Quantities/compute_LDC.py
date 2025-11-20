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
print(f"ENERGY: {energy}")
print(f"palinstrophy: {palinstropy}")
print(f"enstrophy: {enstrophy}")

#TODO: ASKE Plot them as a function of iterations
# %%
# Plot conserved quantities vs iteration
# -------------------------------------
import numpy as np
import matplotlib.pyplot as plt
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
        If provided, save the figure to this path (PNG).
    show : bool
        If True, call plt.show() after plotting.
    """
    ts = getattr(solver, "time_series", None)
    if ts is None:
        raise RuntimeError("Solver has no time_series. Run solver.solve() first.")

    # extract series, convert to numpy arrays and handle None
    energy = np.asarray(ts.energy) if ts.energy is not None else np.array([])
    enstrophy = np.asarray(ts.enstrophy) if ts.enstrophy is not None else np.array([])
    palinstropy = np.asarray(ts.palinstropy) if ts.palinstropy is not None else np.array([])

    # Determine a common iteration axis (use length of longest series)
    lengths = [len(energy), len(enstrophy), len(palinstropy)]
    maxlen = max(lengths)
    if maxlen == 0:
        raise RuntimeError("No time-series data available to plot.")

    it = np.arange(maxlen)

    # Helper to pad series to maxlen for plotting
    def pad_to(arr, n):
        if arr.size == 0:
            return np.full(n, np.nan)
        if arr.size == n:
            return arr
        # pad with NaNs at the end (so plotting shows values up to last recorded iter)
        padded = np.full(n, np.nan)
        padded[: arr.size] = arr
        return padded

    energy_p = pad_to(energy, maxlen)
    enstrophy_p = pad_to(enstrophy, maxlen)
    palinstropy_p = pad_to(palinstropy, maxlen)

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    plt.subplots_adjust(hspace=0.25)

    # Plot settings
    axes[0].plot(it, energy_p, "-o", ms=3, lw=1.2, label="Energy")
    axes[0].set_ylabel("Energy")
    axes[0].grid(True)
    axes[0].legend(loc="best")

    axes[1].plot(it, enstrophy_p, "-o", ms=3, lw=1.2, label="Enstrophy", color="C1")
    axes[1].set_ylabel("Enstrophy")
    axes[1].grid(True)
    axes[1].legend(loc="best")

    axes[2].plot(it, palinstropy_p, "-o", ms=3, lw=1.2, label="Palinstropy", color="C2")
    axes[2].set_ylabel("Palinstropy")
    axes[2].set_xlabel("Iteration")
    axes[2].grid(True)
    axes[2].legend(loc="best")

    # Option: show iteration of convergence if available in metadata
    meta = getattr(solver, "metadata", None)
    if meta is not None and getattr(meta, "iterations", None) is not None:
        it_conv = int(meta.iterations)
        for ax in axes:
            ax.axvline(it_conv, color="0.5", linestyle="--", alpha=0.6)
            ax.text(it_conv, 0.95, "conv.", transform=ax.get_xaxis_transform(),
                    ha="left", va="top", fontsize=8, color="0.5")

    plt.suptitle("Lid-driven cavity: conserved quantities vs iteration")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=200)
        print(f"Saved figure to {savepath}")

    if show:
        plt.show()

    return fig

# Example usage (place after solver.solve(...))
# --------------------------------------------
# solver.solve(tolerance=1e-5, max_iter=10000)
plot_ldc_timeseries(solver, savepath=data_dir / "ldc_quantities_vs_iteration.png", show=True)