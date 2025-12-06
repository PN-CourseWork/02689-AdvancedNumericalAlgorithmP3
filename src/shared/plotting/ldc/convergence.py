"""
Convergence Plots for LDC.

Generates convergence history plots showing residuals and other metrics.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

log = logging.getLogger(__name__)


def plot_convergence(
    timeseries_df: pd.DataFrame, Re: float, solver: str, N: int, output_dir: Path
) -> Path:
    """Plot convergence history (residuals over iterations)."""
    if timeseries_df.empty:
        log.warning("No timeseries data available for convergence plot")
        return None

    fig, ax = plt.subplots()

    # Plot available metrics
    for col in timeseries_df.columns:
        if col != "iteration":
            data = timeseries_df[col].dropna()
            if len(data) > 0:
                ax.semilogy(
                    timeseries_df.loc[data.index, "iteration"],
                    data,
                    label=col.capitalize(),
                )

    ax.set_xlabel(r"Iteration")
    ax.set_ylabel(r"Value")
    solver_label = solver.upper().replace("_", r"\_")
    ax.set_title(
        rf"\textbf{{Convergence History}} --- {solver_label}, $N={N}$, $\mathrm{{Re}}={Re:.0f}$"
    )
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)

    output_path = output_dir / "convergence.pdf"
    fig.savefig(output_path)
    plt.close(fig)

    return output_path
