"""Ghia benchmark validator for lid-driven cavity simulations."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import h5py

from utils.plotting import plt, sns, pd


class GhiaValidator:
    """Validator for lid-driven cavity results against Ghia et al. (1982) benchmark.

    Parameters
    ----------
    Re : float
        Reynolds number of the simulation (can also be inferred from HDF5 file).
    h5_path : Path or str
        Path to HDF5 file with solution fields.
    validation_data_dir : Path or str, optional
        Directory containing Ghia CSV files. If None, uses default location.
    """

    AVAILABLE_RE = [100, 400, 1000, 3200, 5000, 7500, 10000]

    def __init__(
        self,
        h5_path: Path | str,
        Re: Optional[float] = None,
        validation_data_dir: Optional[Path | str] = None
    ):
        """Initialize validator and load solution fields from HDF5 file."""
        self.h5_path = Path(h5_path)

        # Load solution fields from HDF5
        with h5py.File(self.h5_path, 'r') as f:
            # Get Re from metadata or use provided value
            self.Re = Re if Re is not None else f.attrs['Re']

            # Load fields
            grid_points = f['grid_points'][:]
            self.cell_centers = grid_points[:, :2]
            self.u = f['fields/u'][:]
            self.v = f['fields/v'][:]

        # Find closest available Reynolds number
        self.Re_closest = min(self.AVAILABLE_RE, key=lambda x: abs(x - self.Re))
        if abs(self.Re_closest - self.Re) > 0.1 * self.Re:
            print(f"Warning: Using Ghia data for Re={self.Re_closest}, "
                  f"requested Re={self.Re}")

        # Set validation data directory
        if validation_data_dir is None:
            # Default: project_root/data/validation/ghia
            from utils import get_project_root
            validation_data_dir = get_project_root() / "data" / "validation" / "ghia"
        self.validation_data_dir = Path(validation_data_dir)

        # Load Ghia benchmark data
        self._load_ghia_data()

    def _load_ghia_data(self):
        """Load Ghia benchmark data from CSV files."""
        # U velocity along vertical centerline
        u_file = self.validation_data_dir / f"ghia_Re{self.Re_closest}_u_centerline.csv"
        u_df = pd.read_csv(u_file)
        self.ghia_y = u_df['y'].values
        self.ghia_u = u_df['u'].values

        # V velocity along horizontal centerline
        v_file = self.validation_data_dir / f"ghia_Re{self.Re_closest}_v_centerline.csv"
        v_df = pd.read_csv(v_file)
        self.ghia_x = v_df['x'].values
        self.ghia_v = v_df['v'].values

    def _extract_centerline_u(self):
        """Extract u velocity along vertical centerline (x=0.5) using interpolation."""
        from scipy.interpolate import RectBivariateSpline

        # Get coordinates and round to avoid floating point precision issues
        x = np.round(self.cell_centers[:, 0], decimals=10)
        y = np.round(self.cell_centers[:, 1], decimals=10)

        x_unique = np.sort(np.unique(x))
        y_unique = np.sort(np.unique(y))

        nx = len(x_unique)
        ny = len(y_unique)

        # Sort by y first, then by x to get proper 2D grid ordering
        sort_indices = np.lexsort((x, y))
        u_sorted = self.u[sort_indices]
        u_grid = u_sorted.reshape((ny, nx))

        # Create interpolator using bicubic spline
        interp_u = RectBivariateSpline(y_unique, x_unique, u_grid, kx=3, ky=3)

        # Interpolate along vertical centerline (x=0.5)
        y_interp = np.linspace(0, 1, 200)
        u_centerline = interp_u(y_interp, 0.5, grid=False)

        return y_interp, u_centerline

    def _extract_centerline_v(self):
        """Extract v velocity along horizontal centerline (y=0.5) using interpolation."""
        from scipy.interpolate import RectBivariateSpline

        # Get coordinates and round to avoid floating point precision issues
        x = np.round(self.cell_centers[:, 0], decimals=10)
        y = np.round(self.cell_centers[:, 1], decimals=10)

        x_unique = np.sort(np.unique(x))
        y_unique = np.sort(np.unique(y))

        nx = len(x_unique)
        ny = len(y_unique)

        # Sort by y first, then by x to get proper 2D grid ordering
        sort_indices = np.lexsort((x, y))
        v_sorted = self.v[sort_indices]
        v_grid = v_sorted.reshape((ny, nx))

        # Create interpolator using bicubic spline
        interp_v = RectBivariateSpline(y_unique, x_unique, v_grid, kx=3, ky=3)

        # Interpolate along horizontal centerline (y=0.5)
        x_interp = np.linspace(0, 1, 200)
        v_centerline = interp_v(0.5, x_interp, grid=False)

        return x_interp, v_centerline

    def plot_validation(self, output_path: Optional[Path | str] = None, show: bool = False):
        """Plot velocity validation against Ghia benchmark using seaborn.

        Creates a two-panel figure with u and v velocity validation side-by-side.

        Parameters
        ----------
        output_path : Path or str, optional
            Path to save figure. If None, figure is not saved.
        show : bool, default False
            Whether to show the plot.
        """
        # Extract centerline data (interpolated)
        y_sim, u_sim = self._extract_centerline_u()
        x_sim, v_sim = self._extract_centerline_v()

        # Create figure with two subplots manually
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Left panel: U velocity along vertical centerline (y-position on y-axis)
        # Sort by y to ensure proper line connectivity
        sort_idx = np.argsort(y_sim)
        axes[0].plot(u_sim[sort_idx], y_sim[sort_idx], color='C0', label='Numerical Results')
        axes[0].scatter(self.ghia_u, self.ghia_y, marker='x', color='C1',
                       label='Ghia et al. (1982)')
        axes[0].set_xlabel('$u$')
        axes[0].set_ylabel('$y$')
        axes[0].set_title('U velocity\n(vertical centerline)')
        axes[0].legend(frameon=True, loc='best')

        # Right panel: V velocity along horizontal centerline (x-position on x-axis)
        # Sort by x to ensure proper line connectivity
        sort_idx = np.argsort(x_sim)
        axes[1].plot(x_sim[sort_idx], v_sim[sort_idx], color='C0', label='Numerical Results')
        axes[1].scatter(self.ghia_x, self.ghia_v, marker='x', color='C1',
                       label='Ghia et al. (1982)')
        axes[1].set_xlabel('$x$')
        axes[1].set_ylabel('$v$')
        axes[1].set_title('V velocity\n(horizontal centerline)')
        axes[1].legend(frameon=True, loc='best')

        # Set overall title
        fig.suptitle(f'Centerline Velocity Validation (Re = {self.Re:.0f})',
                     fontweight='bold', fontsize=14, y=1.02)


        if output_path:
            fig.savefig(output_path, bbox_inches='tight', dpi=300)
            print(f"Validation plot saved to: {output_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    def compute_errors(self):
        """Compute error metrics against Ghia benchmark data.

        Returns
        -------
        dict
            Dictionary containing error metrics for u and v velocities:
            - 'u_l2': L2 norm of u error
            - 'u_linf': L∞ (maximum) norm of u error
            - 'u_rms': Root mean square error for u
            - 'v_l2': L2 norm of v error
            - 'v_linf': L∞ (maximum) norm of v error
            - 'v_rms': Root mean square error for v
        """
        from scipy.interpolate import interp1d

        # Extract centerline data
        y_sim, u_sim = self._extract_centerline_u()
        x_sim, v_sim = self._extract_centerline_v()

        # Interpolate simulation results at Ghia benchmark points
        # For u velocity (vertical centerline)
        u_interp_func = interp1d(y_sim, u_sim, kind='cubic', fill_value='extrapolate')
        u_sim_at_ghia = u_interp_func(self.ghia_y)
        u_error = u_sim_at_ghia - self.ghia_u

        # For v velocity (horizontal centerline)
        v_interp_func = interp1d(x_sim, v_sim, kind='cubic', fill_value='extrapolate')
        v_sim_at_ghia = v_interp_func(self.ghia_x)
        v_error = v_sim_at_ghia - self.ghia_v

        # Compute error norms
        errors = {
            'u_l2': np.sqrt(np.sum(u_error**2)),
            'u_linf': np.max(np.abs(u_error)),
            'u_rms': np.sqrt(np.mean(u_error**2)),
            'v_l2': np.sqrt(np.sum(v_error**2)),
            'v_linf': np.max(np.abs(v_error)),
            'v_rms': np.sqrt(np.mean(v_error**2)),
        }

        return errors

    def print_summary(self):
        """Print validation summary with error metrics."""
        errors = self.compute_errors()

        print("\n" + "="*70)
        print(f"{'VALIDATION SUMMARY':^70}")
        print("="*70)
        print(f"  Reynolds number: Re = {self.Re:.0f}")
        print(f"  Benchmark: Ghia et al. (1982), Re = {self.Re_closest}")
        print(f"  Solution file: {self.h5_path.name}")
        print("-"*70)
        print(f"{'ERROR METRICS':^70}")
        print("-"*70)
        print(f"  {'Velocity':<12} {'L² Error':<15} {'L∞ Error':<15} {'RMS Error':<15}")
        print("-"*70)
        print(f"  {'u (vertical)':<12} {errors['u_l2']:<15.6e} {errors['u_linf']:<15.6e} {errors['u_rms']:<15.6e}")
        print(f"  {'v (horizontal)':<12} {errors['v_l2']:<15.6e} {errors['v_linf']:<15.6e} {errors['v_rms']:<15.6e}")
        print("="*70)

        # Provide interpretation
        if errors['u_rms'] < 1e-3 and errors['v_rms'] < 1e-3:
            quality = "EXCELLENT"
        elif errors['u_rms'] < 1e-2 and errors['v_rms'] < 1e-2:
            quality = "GOOD"
        elif errors['u_rms'] < 0.05 and errors['v_rms'] < 0.05:
            quality = "ACCEPTABLE"
        else:
            quality = "NEEDS IMPROVEMENT"

        print(f"  Overall validation quality: {quality}")
        print("="*70 + "\n")

