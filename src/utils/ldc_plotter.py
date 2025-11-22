"""LDC results plotter for single and multiple runs."""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class LDCPlotter:
    """Plotter for lid-driven cavity simulation results.

    Clean DataFrame-native implementation for plotting LDC solutions.

    Parameters
    ----------
    runs : dict, str, Path, or list
        Single run or list of runs. Can be:
        - str/Path: Path to HDF5 file
        - dict: Dictionary with 'h5_path' (and optionally 'label')
        - list: List of any of the above (requires 'label' in dicts)

    Attributes
    ----------
    fields : pd.DataFrame
        Spatial fields (x, y, u, v, p) for all runs
    time_series : pd.DataFrame
        Time series data (residuals) for all runs
    metadata : pd.DataFrame
        Configuration and convergence metadata for all runs

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

    def __init__(self, runs):
        """Initialize plotter and load data as DataFrames.

        Parameters
        ----------
        runs : dict, str, Path, or list
            Single run or list of runs to load.
        """
        # Normalize to list
        if not isinstance(runs, list):
            runs = [runs]

        # Load all runs
        fields_list = []
        time_series_list = []
        metadata_list = []

        for run in runs:
            # Normalize run to dict
            if isinstance(run, (str, Path)):
                run = {"h5_path": run, "label": Path(run).stem}

            h5_path = Path(run["h5_path"])
            if not h5_path.exists():
                raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

            label = run.get("label", h5_path.stem)

            # Load DataFrames and add metadata
            metadata_df = pd.read_hdf(h5_path, 'metadata').assign(run=label)
            fields_df = pd.read_hdf(h5_path, 'fields').assign(run=label)
            time_series_df = pd.read_hdf(h5_path, 'time_series').assign(
                run=label,
                iteration=lambda df: range(len(df))
            )

            fields_list.append(fields_df)
            time_series_list.append(time_series_df)
            metadata_list.append(metadata_df)

        # Concatenate all runs
        self.fields = pd.concat(fields_list, ignore_index=True)
        self.time_series = pd.concat(time_series_list, ignore_index=True)
        self.metadata = pd.concat(metadata_list, ignore_index=True)

    def _require_single_run(self):
        """Check that only single run is loaded (for field plotting)."""
        if self.metadata['run'].nunique() > 1:
            raise ValueError("Field plotting only available for single run.")

    def plot_convergence(self, output_path=None):
        """Plot convergence history using seaborn.

        Parameters
        ----------
        output_path : str or Path, optional
            Path to save figure. If None, figure is not saved.
        """
        n_runs = self.metadata['run'].nunique()

        g = sns.relplot(
            data=self.time_series,
            x="iteration",
            y="residual",
            hue="run" if n_runs > 1 else None,
            kind="line",
            height=5,
            aspect=1.6,
            linewidth=2,
            legend="auto" if n_runs > 1 else False,
        )

        g.ax.set_yscale("log")
        g.ax.grid(True, alpha=0.3)
        g.ax.set_xlabel("Iteration")
        g.ax.set_ylabel("Residual")

        if n_runs == 1:
            Re = self.metadata['Re'].iloc[0]
            g.ax.set_title(f"Convergence History (Re = {Re:.0f})", fontweight="bold")
        else:
            g.ax.set_title("Convergence Comparison", fontweight="bold")

        if output_path:
            g.savefig(output_path, bbox_inches="tight", dpi=300)
            print(f"Convergence plot saved to: {output_path}")

    def plot_velocity_fields(self, output_path=None):
        """Plot velocity components (u and v) using matplotlib tricontourf.

        Only available for single-run plotting.

        Parameters
        ----------
        output_path : str or Path, optional
            Path to save figure. If None, figure is not saved.
        """
        self._require_single_run()

        Re = self.metadata['Re'].iloc[0]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # U velocity
        cf_u = axes[0].tricontourf(
            self.fields['x'], self.fields['y'], self.fields['u'],
            levels=20, cmap="RdBu_r"
        )
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].set_title("U velocity", fontweight="bold")
        axes[0].set_aspect("equal")
        plt.colorbar(cf_u, ax=axes[0], label="u")

        # V velocity
        cf_v = axes[1].tricontourf(
            self.fields['x'], self.fields['y'], self.fields['v'],
            levels=20, cmap="RdBu_r"
        )
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        axes[1].set_title("V velocity", fontweight="bold")
        axes[1].set_aspect("equal")
        plt.colorbar(cf_v, ax=axes[1], label="v")

        fig.suptitle(f"Velocity Components (Re = {Re:.0f})", fontweight="bold")
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, bbox_inches="tight", dpi=300)
            print(f"Velocity fields plot saved to: {output_path}")

    def plot_pressure(self, output_path=None):
        """Plot pressure field using matplotlib tricontourf.

        Only available for single-run plotting.

        Parameters
        ----------
        output_path : str or Path, optional
            Path to save figure. If None, figure is not saved.
        """
        self._require_single_run()

        Re = self.metadata['Re'].iloc[0]
        fig, ax = plt.subplots(figsize=(8, 7))

        cf = ax.tricontourf(
            self.fields['x'], self.fields['y'], self.fields['p'],
            levels=20, cmap="coolwarm"
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Pressure Field (Re = {Re:.0f})", fontweight="bold")
        ax.set_aspect("equal")
        plt.colorbar(cf, ax=ax, label="Pressure")
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, bbox_inches="tight", dpi=300)
            print(f"Pressure plot saved to: {output_path}")

    def plot_velocity_magnitude(self, output_path=None):
        """Plot velocity magnitude with streamlines.

        Only available for single-run plotting.

        Parameters
        ----------
        output_path : str or Path, optional
            Path to save figure. If None, figure is not saved.
        """
        from scipy.interpolate import griddata

        self._require_single_run()

        Re = self.metadata['Re'].iloc[0]

        # Compute velocity magnitude
        u = self.fields['u'].values
        v = self.fields['v'].values
        vel_mag = np.sqrt(u**2 + v**2)

        fig, ax = plt.subplots(figsize=(8, 7))

        # Velocity magnitude contour
        cf = ax.tricontourf(
            self.fields['x'], self.fields['y'], vel_mag,
            levels=20, cmap="coolwarm"
        )

        # Create uniform grid for streamlines
        x = self.fields['x'].values
        y = self.fields['y'].values
        n_grid = 50
        x_uniform = np.linspace(x.min(), x.max(), n_grid)
        y_uniform = np.linspace(y.min(), y.max(), n_grid)
        X_uniform, Y_uniform = np.meshgrid(x_uniform, y_uniform)

        # Interpolate velocity onto uniform grid
        points = np.column_stack((x, y))
        u_uniform = griddata(points, u, (X_uniform, Y_uniform), method='cubic')
        v_uniform = griddata(points, v, (X_uniform, Y_uniform), method='cubic')

        # Streamlines
        stream = ax.streamplot(
            x_uniform, y_uniform, u_uniform, v_uniform,
            color='white', linewidth=1, density=1.5,
            arrowsize=1.2, arrowstyle='->'
        )
        stream.lines.set_alpha(0.6)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Velocity Magnitude with Streamlines (Re = {Re:.0f})", fontweight="bold")
        ax.set_aspect("equal")
        plt.colorbar(cf, ax=ax, label="Velocity magnitude")
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, bbox_inches="tight", dpi=300)
            print(f"Velocity magnitude plot saved to: {output_path}")
