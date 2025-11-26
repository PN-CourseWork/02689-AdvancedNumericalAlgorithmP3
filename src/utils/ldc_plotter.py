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

    def plot_fields(self, output_path=None, interp_resolution=200):
        """Plot all solution fields (pressure, u velocity, v velocity).

        Only available for single-run plotting.

        Parameters
        ----------
        output_path : str or Path, optional
            Path to save figure. If None, figure is not saved.
        interp_resolution : int, optional
            Resolution for interpolated grid. If > original resolution,
            uses spectral interpolation for smooth visualization. Default 200.
        """
        self._require_single_run()

        Re = self.metadata['Re'].iloc[0]

        # Determine grid size and reshape to 2D
        nx = self.fields['x'].nunique()
        ny = self.fields['y'].nunique()

        # Get unique x and y coordinates (sorted)
        x_unique = np.sort(self.fields['x'].unique())
        y_unique = np.sort(self.fields['y'].unique())

        # Sort data by (y, x) to ensure consistent ordering for reshape
        sorted_fields = self.fields.sort_values(['y', 'x'])
        P_orig = sorted_fields['p'].values.reshape(ny, nx)
        U_orig = sorted_fields['u'].values.reshape(ny, nx)
        V_orig = sorted_fields['v'].values.reshape(ny, nx)

        # Interpolate to finer grid if requested
        if interp_resolution > max(nx, ny):
            from spectral.spectral import barycentric_weights, barycentric_interpolate

            # Create fine grid
            x_fine = np.linspace(x_unique[0], x_unique[-1], interp_resolution)
            y_fine = np.linspace(y_unique[0], y_unique[-1], interp_resolution)
            X, Y = np.meshgrid(x_fine, y_fine)

            # Compute barycentric weights once
            wx = barycentric_weights(x_unique)
            wy = barycentric_weights(y_unique)

            # Interpolate each field (2D tensor product interpolation)
            def interp_2d(field_2d):
                # First interpolate along x for each y
                temp = np.zeros((ny, interp_resolution))
                for j in range(ny):
                    temp[j, :] = barycentric_interpolate(x_unique, field_2d[j, :], x_fine, wx)
                # Then interpolate along y for each x
                result = np.zeros((interp_resolution, interp_resolution))
                for i in range(interp_resolution):
                    result[:, i] = barycentric_interpolate(y_unique, temp[:, i], y_fine, wy)
                return result

            P = interp_2d(P_orig)
            U = interp_2d(U_orig)
            V = interp_2d(V_orig)
        else:
            X, Y = np.meshgrid(x_unique, y_unique)
            P, U, V = P_orig, U_orig, V_orig

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

    def plot_streamlines(self, output_path=None, interp_resolution=200):
        """Plot velocity magnitude with streamlines.

        Only available for single-run plotting.

        Parameters
        ----------
        output_path : str or Path, optional
            Path to save figure. If None, figure is not saved.
        interp_resolution : int, optional
            Resolution for interpolated grid. Default 200.
        """
        self._require_single_run()

        Re = self.metadata['Re'].iloc[0]

        # Determine grid size and reshape to 2D
        nx = self.fields['x'].nunique()
        ny = self.fields['y'].nunique()

        # Get unique x and y coordinates (sorted)
        x_unique = np.sort(self.fields['x'].unique())
        y_unique = np.sort(self.fields['y'].unique())

        # Sort data by (y, x) to ensure consistent ordering for reshape
        sorted_fields = self.fields.sort_values(['y', 'x'])
        U_orig = sorted_fields['u'].values.reshape(ny, nx)
        V_orig = sorted_fields['v'].values.reshape(ny, nx)

        # Interpolate to finer grid if requested
        if interp_resolution > max(nx, ny):
            from spectral.spectral import barycentric_weights, barycentric_interpolate

            # Create fine grid
            x_fine = np.linspace(x_unique[0], x_unique[-1], interp_resolution)
            y_fine = np.linspace(y_unique[0], y_unique[-1], interp_resolution)
            X, Y = np.meshgrid(x_fine, y_fine)

            # Compute barycentric weights once
            wx = barycentric_weights(x_unique)
            wy = barycentric_weights(y_unique)

            # Interpolate each field (2D tensor product interpolation)
            def interp_2d(field_2d):
                temp = np.zeros((ny, interp_resolution))
                for j in range(ny):
                    temp[j, :] = barycentric_interpolate(x_unique, field_2d[j, :], x_fine, wx)
                result = np.zeros((interp_resolution, interp_resolution))
                for i in range(interp_resolution):
                    result[:, i] = barycentric_interpolate(y_unique, temp[:, i], y_fine, wy)
                return result

            U = interp_2d(U_orig)
            V = interp_2d(V_orig)
        else:
            X, Y = np.meshgrid(x_unique, y_unique)
            U, V = U_orig, V_orig

        vel_mag = np.sqrt(U**2 + V**2)

        fig, ax = plt.subplots(figsize=(8, 7))

        # Velocity magnitude contour
        cf = ax.contourf(X, Y, vel_mag, levels=20, cmap="coolwarm")

        # Get coordinates for streamlines
        x_stream = X[0, :]  # 1D x coordinates
        y_stream = Y[:, 0]  # 1D y coordinates

        # Streamlines using the (already interpolated) velocity field
        stream = ax.streamplot(
            x_stream, y_stream, U, V,
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
