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

    def plot_convergence(self, output_path=None, normalization_iters=5):
        """Plot all time series residuals normalized by early iteration maximum.

        Uses STAR-CCM+ style "Auto" normalization: normalizes by the maximum
        value observed in the first N iterations.

        Parameters
        ----------
        output_path : str or Path, optional
            Path to save figure. If None, figure is not saved.
        normalization_iters : int, optional
            Number of initial iterations to use for computing normalization
            reference (default: 5, following STAR-CCM+ convention).
        """
        n_runs = self.metadata['run'].nunique()

        # Get residual columns (exclude 'iteration' and 'run')
        residual_cols = [col for col in self.time_series.columns
                        if col not in ['iteration', 'run']]

        # Melt to long format
        df_long = self.time_series.melt(
            id_vars=['iteration', 'run'],
            value_vars=residual_cols,
            var_name='residual_type',
            value_name='residual_value'
        )

        # Normalize each (run, residual_type) group by max of first N iterations
        def normalize_group(group):
            # Get maximum value from first N iterations
            first_n = group.head(normalization_iters)
            ref_value = first_n.max()
            if ref_value != 0:
                return group / ref_value
            return group

        df_long['normalized_residual'] = df_long.groupby(['run', 'residual_type'])['residual_value'].transform(normalize_group)

        # Plot with seaborn
        g = sns.relplot(
            data=df_long,
            x="iteration",
            y="normalized_residual",
            hue="residual_type",
            style="run" if n_runs > 1 else None,
            kind="line",
            height=5,
            aspect=1.6,
            linewidth=2,
            legend="auto",
        )

        g.ax.set_yscale("log")
        g.ax.grid(True, alpha=0.3)
        g.ax.set_xlabel("Iteration")
        g.ax.set_ylabel("Normalized Metric")

        if n_runs == 1:
            Re = self.metadata['Re'].iloc[0]
            g.ax.set_title(f"Convergence History (Re = {Re:.0f})", fontweight="bold")
        else:
            g.ax.set_title("Convergence Comparison", fontweight="bold")

        if output_path:
            g.savefig(output_path, bbox_inches="tight", dpi=300)
            print(f"Convergence plot saved to: {output_path}")

    def plot_fields(self, output_path=None):
        """Plot all solution fields (pressure, u velocity, v velocity).

        Only available for single-run plotting.

        Parameters
        ----------
        output_path : str or Path, optional
            Path to save figure. If None, figure is not saved.
        """
        self._require_single_run()

        Re = self.metadata['Re'].iloc[0]

        # Determine grid size and reshape to 2D
        nx = self.fields['x'].nunique()
        ny = self.fields['y'].nunique()

        # Get unique x and y coordinates (sorted)
        x_unique = np.sort(self.fields['x'].unique())
        y_unique = np.sort(self.fields['y'].unique())

        # Create meshgrid
        X, Y = np.meshgrid(x_unique, y_unique)

        # Reshape fields to 2D - data is stored in C-order (row-major)
        P = self.fields['p'].values.reshape(ny, nx)
        U = self.fields['u'].values.reshape(ny, nx)
        V = self.fields['v'].values.reshape(ny, nx)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Pressure
        cf_p = axes[0].contourf(X, Y, P, levels=20, cmap="coolwarm")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].set_title("Pressure", fontweight="bold")
        axes[0].set_aspect("equal")
        plt.colorbar(cf_p, ax=axes[0], label="p")

        # U velocity
        cf_u = axes[1].contourf(X, Y, U, levels=20, cmap="RdBu_r")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        axes[1].set_title("U velocity", fontweight="bold")
        axes[1].set_aspect("equal")
        plt.colorbar(cf_u, ax=axes[1], label="u")

        # V velocity
        cf_v = axes[2].contourf(X, Y, V, levels=20, cmap="RdBu_r")
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("y")
        axes[2].set_title("V velocity", fontweight="bold")
        axes[2].set_aspect("equal")
        plt.colorbar(cf_v, ax=axes[2], label="v")

        fig.suptitle(f"Solution Fields (Re = {Re:.0f})", fontweight="bold")
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, bbox_inches="tight", dpi=300)
            print(f"Fields plot saved to: {output_path}")

    def plot_streamlines(self, output_path=None):
        """Plot velocity magnitude with streamlines.

        Only available for single-run plotting.

        Parameters
        ----------
        output_path : str or Path, optional
            Path to save figure. If None, figure is not saved.
        """
        from scipy.interpolate import RectBivariateSpline

        self._require_single_run()

        Re = self.metadata['Re'].iloc[0]

        # Determine grid size and reshape to 2D
        nx = self.fields['x'].nunique()
        ny = self.fields['y'].nunique()

        # Get unique x and y coordinates (sorted)
        x_unique = np.sort(self.fields['x'].unique())
        y_unique = np.sort(self.fields['y'].unique())

        # Reshape velocity fields to 2D - data is stored in C-order (row-major)
        U = self.fields['u'].values.reshape(ny, nx)
        V = self.fields['v'].values.reshape(ny, nx)
        vel_mag = np.sqrt(U**2 + V**2)

        # Create meshgrid for plotting
        X, Y = np.meshgrid(x_unique, y_unique)

        fig, ax = plt.subplots(figsize=(8, 7))

        # Velocity magnitude contour
        cf = ax.contourf(X, Y, vel_mag, levels=20, cmap="coolwarm")

        # Interpolate velocity onto finer uniform grid for smooth streamlines
        n_grid = 100
        x_fine = np.linspace(x_unique.min(), x_unique.max(), n_grid)
        y_fine = np.linspace(y_unique.min(), y_unique.max(), n_grid)
        X_fine, Y_fine = np.meshgrid(x_fine, y_fine)

        # Use RectBivariateSpline for structured grid interpolation
        u_interp = RectBivariateSpline(y_unique, x_unique, U)
        v_interp = RectBivariateSpline(y_unique, x_unique, V)

        U_fine = u_interp(y_fine, x_fine)
        V_fine = v_interp(y_fine, x_fine)

        # Streamlines
        stream = ax.streamplot(
            x_fine, y_fine, U_fine, V_fine,
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
            print(f"Streamlines plot saved to: {output_path}")
