"""LDC results plotter for single and multiple runs."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import h5py


class LDCPlotter:
    """Plotter for lid-driven cavity simulation results.

    Handles both single and multiple runs for comparison using HDF5 files.

    Parameters
    ----------
    runs : dict, str, Path, or list
        Single run or list of runs. Can be:
        - str/Path: Path to HDF5 file
        - dict: Dictionary with 'h5_path' (and optionally 'label')
        - list: List of any of the above (requires 'label' in dicts)

    Examples
    --------
    >>> # Single run
    >>> plotter = LDCPlotter('run.h5')
    >>> plotter.plot_convergence()

    >>> # Multiple runs with labels
    >>> plotter = LDCPlotter([
    ...     {'h5_path': 'run1.h5', 'label': '32x32'},
    ...     {'h5_path': 'run2.h5', 'label': '64x64'}
    ... ])
    """

    def __init__(self, runs: dict | str | Path | list):
        """Initialize plotter and load data."""
        # Normalize to list of dicts
        if isinstance(runs, (str, Path)):
            self.runs = [{"h5_path": runs}]
            self.single_run = True
        elif isinstance(runs, dict):
            self.runs = [runs]
            self.single_run = True
        else:
            self.runs = runs
            self.single_run = False

        # Validate and load all runs
        self.run_data = []
        for i, run in enumerate(self.runs):
            # Normalize run to dict if it's a path
            if isinstance(run, (str, Path)):
                run = {"h5_path": run}
            run_info = self._load_run(run, run_idx=i)
            self.run_data.append(run_info)

        # For single run, expose data at top level
        if self.single_run:
            rd = self.run_data[0]
            self.Re = rd["Re"]
            self.iter_residuals = rd["iter_residuals"]
            self.x = rd["x"]
            self.y = rd["y"]
            self.u = rd["u"]
            self.v = rd["v"]
            self.p = rd["p"]
            self.vel_mag = rd["vel_mag"]
            self.h5_path = rd["h5_path"]

    def _load_run(self, run: dict, run_idx: int) -> dict:
        """Load data for a single run from HDF5 file."""
        h5_path = Path(run["h5_path"])

        if not h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

        with h5py.File(h5_path, "r") as f:
            # Load metadata
            Re = f.attrs["Re"]

            # Load time-series
            iter_residuals = f["time_series/iter_residual"][:]

            # Load spatial fields
            grid_points = f["grid_points"][:]
            x = grid_points[:, 0]
            y = grid_points[:, 1]
            u = f["fields/u"][:]
            v = f["fields/v"][:]
            p = f["fields/p"][:]
            vel_mag = f["fields/velocity_magnitude"][:]

        # Get label (use provided or generate from filename)
        if not self.single_run and "label" not in run:
            raise ValueError(
                f"Run {run_idx} missing 'label' key (required for multiple runs)"
            )
        label = run.get("label", h5_path.stem)

        return {
            "Re": Re,
            "iter_residuals": iter_residuals,
            "x": x,
            "y": y,
            "u": u,
            "v": v,
            "p": p,
            "vel_mag": vel_mag,
            "label": label,
            "h5_path": h5_path,
        }

    def plot_convergence(
        self, output_path: Optional[Path | str] = None, show: bool = False
    ):
        """Plot convergence history using seaborn.

        Automatically handles single or multiple runs using hue.

        Parameters
        ----------
        output_path : Path or str, optional
            Path to save figure. If None, figure is not saved.
        show : bool, default False
            Whether to show the plot.
        """
        # Create DataFrame for all runs
        dfs = []
        for rd in self.run_data:
            df = pd.DataFrame(
                {
                    "Iteration": range(len(rd["iter_residuals"])),
                    "Residual": rd["iter_residuals"],
                    "Run": rd["label"],
                }
            )
            dfs.append(df)

        combined_df = pd.concat(dfs, ignore_index=True)

        # Create plot using seaborn relplot
        g = sns.relplot(
            data=combined_df,
            x="Iteration",
            y="Residual",
            hue="Run" if not self.single_run else None,
            kind="line",
            height=5,
            aspect=1.6,
            linewidth=2,
            legend="auto" if not self.single_run else False,
        )

        # Set log scale for y-axis
        g.ax.set_yscale("log")
        g.ax.grid(True, alpha=0.3)

        # Set title based on single/multi run
        if self.single_run:
            g.ax.set_title(
                f"Convergence History (Re = {self.run_data[0]['Re']:.0f})",
                fontweight="bold",
            )
        else:
            g.ax.set_title("Convergence Comparison", fontweight="bold")

        if output_path:
            g.savefig(output_path, bbox_inches="tight", dpi=300)
            print(f"Convergence plot saved to: {output_path}")

    def plot_velocity_fields(
        self, output_path: Optional[Path | str] = None, show: bool = False
    ):
        """Plot velocity components using matplotlib tricontourf.

        Parameters
        ----------
        output_path : Path or str, optional
            Path to save figure. If None, figure is not saved.
        show : bool, default False
            Whether to show the plot.
        """
        if not self.single_run:
            raise ValueError("Field plotting only available for single run.")

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # U velocity
        axes[0].tricontourf(self.x, self.y, self.u, levels=20, cmap="RdBu_r")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].set_title("U velocity")
        axes[0].set_aspect("equal")

        # V velocity
        axes[1].tricontourf(self.x, self.y, self.v, levels=20, cmap="RdBu_r")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        axes[1].set_title("V velocity")
        axes[1].set_aspect("equal")

        # Velocity magnitude
        axes[2].tricontourf(self.x, self.y, self.vel_mag, levels=20, cmap="viridis")
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("y")
        axes[2].set_title("Velocity magnitude")
        axes[2].set_aspect("equal")

        plt.suptitle(f"Lid-Driven Cavity: Re = {self.Re:.0f}", fontweight="bold")
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, bbox_inches="tight", dpi=300)
            print(f"Velocity plot saved to: {output_path}")

    def plot_pressure(
        self, output_path: Optional[Path | str] = None, show: bool = False
    ):
        """Plot pressure field using matplotlib tricontourf.

        Parameters
        ----------
        output_path : Path or str, optional
            Path to save figure. If None, figure is not saved.
        show : bool, default False
            Whether to show the plot.
        """
        if not self.single_run:
            raise ValueError("Field plotting only available for single run.")

        fig, ax = plt.subplots(figsize=(8, 7))
        cf = ax.tricontourf(self.x, self.y, self.p, levels=20, cmap="coolwarm")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Pressure field (Re = {self.Re:.0f})")
        ax.set_aspect("equal")
        plt.colorbar(cf, ax=ax, label="Pressure")
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, bbox_inches="tight", dpi=300)
            print(f"Pressure plot saved to: {output_path}")
