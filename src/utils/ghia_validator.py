"""Ghia benchmark validator for lid-driven cavity simulations."""

from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

from utils.plotting import plt, pd
from utils.field_interpolator import UnifiedFieldInterpolator


class GhiaValidator:
    """Validator for lid-driven cavity results against Ghia et al. (1982) benchmark.

    Clean DataFrame-native implementation for validating LDC solutions.

    Parameters
    ----------
    h5_path : str or Path
        Path to HDF5 file with solution fields.
    Re : float, optional
        Reynolds number (inferred from file if not provided).
    validation_data_dir : str or Path, optional
        Directory containing Ghia CSV files. If None, uses default location.
    method_label : str, optional
        Label for this method (for multi-method comparisons). Defaults to filename stem.

    Attributes
    ----------
    fields : pd.DataFrame
        Solution fields (x, y, u, v, p)
    ghia_u : pd.DataFrame
        Ghia benchmark u-velocity data
    ghia_v : pd.DataFrame
        Ghia benchmark v-velocity data
    Re : float
        Reynolds number
    method_label : str
        Method label for plotting
    """

    AVAILABLE_RE = [100, 400, 1000, 3200, 5000, 7500, 10000]

    def __init__(self, h5_path, Re=None, validation_data_dir=None, method_label=None):
        """Initialize validator and load data as DataFrames.

        Parameters
        ----------
        h5_path : str or Path
            Path to HDF5 file with solution fields.
        Re : float, optional
            Reynolds number (inferred from file if not provided).
        validation_data_dir : str or Path, optional
            Directory containing Ghia CSV files. If None, uses default location.
        method_label : str, optional
            Label for this method (for multi-method comparisons).
        """
        self.h5_path = Path(h5_path)

        # Create unified interpolator (handles both FV and Spectral grids)
        self.interpolator = UnifiedFieldInterpolator(self.h5_path)
        self.fields = self.interpolator.fields  # For compatibility

        # Load metadata
        metadata = pd.read_hdf(self.h5_path, 'metadata')

        # Get Reynolds number
        self.Re = Re if Re is not None else metadata['Re'].iloc[0]

        # Require exact match for Reynolds number
        if self.Re not in self.AVAILABLE_RE:
            raise ValueError(
                f"No Ghia benchmark data available for Re={self.Re}. "
                f"Available Re values: {self.AVAILABLE_RE}"
            )

        # Method label for multi-method comparisons
        self.method_label = method_label or self.h5_path.stem

        # Set validation data directory
        if validation_data_dir is None:
            from utils import get_project_root
            validation_data_dir = get_project_root() / "data" / "validation" / "ghia"
        self.validation_data_dir = Path(validation_data_dir)

        # Load Ghia benchmark DataFrames
        self._load_ghia_data()

    def _load_ghia_data(self):
        """Load Ghia benchmark data as DataFrames."""
        u_file = self.validation_data_dir / f"ghia_Re{int(self.Re)}_u_centerline.csv"
        v_file = self.validation_data_dir / f"ghia_Re{int(self.Re)}_v_centerline.csv"

        if not u_file.exists() or not v_file.exists():
            raise FileNotFoundError(
                f"Ghia data files not found for Re={self.Re} in {self.validation_data_dir}"
            )

        self.ghia_u = pd.read_csv(u_file)
        self.ghia_v = pd.read_csv(v_file)

    def _extract_centerline(self, field, centerline_axis):
        """Extract velocity along centerline using unified interpolator.

        Parameters
        ----------
        field : str
            Field to extract ('u' or 'v')
        centerline_axis : str
            Axis along which to extract ('x' or 'y')
            - 'x': horizontal centerline at y=0.5
            - 'y': vertical centerline at x=0.5

        Returns
        -------
        position : np.ndarray
            Position coordinates along centerline
        velocity : np.ndarray
            Velocity values along centerline
        """
        # Use unified interpolator (handles both FV and Spectral grids consistently)
        position, velocity = self.interpolator.extract_centerline(
            field=field,
            axis=centerline_axis,
            n_points=200
        )
        return position, velocity

    def get_validation_dataframe(self):
        """Create validation DataFrame for plotting with seaborn.

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame with columns: position, velocity, source, component, method
            - position: x or y coordinate
            - velocity: u or v value
            - source: 'Simulation' or 'Ghia et al. (1982)'
            - component: 'u' or 'v'
            - method: method label (for multi-method comparisons)
        """
        # Extract centerline data
        y_sim, u_sim = self._extract_centerline('u', 'y')
        x_sim, v_sim = self._extract_centerline('v', 'x')

        # Simulation data
        u_sim_df = pd.DataFrame({'position': y_sim, 'velocity': u_sim}).assign(
            source='Simulation',
            component='u (vertical centerline)',
            method=self.method_label
        )

        v_sim_df = pd.DataFrame({'position': x_sim, 'velocity': v_sim}).assign(
            source='Simulation',
            component='v (horizontal centerline)',
            method=self.method_label
        )

        # Ghia benchmark data (already DataFrames, just rename and add columns)
        u_ghia_df = self.ghia_u.rename(columns={'y': 'position', 'u': 'velocity'}).assign(
            source='Ghia et al. (1982)',
            component='u (vertical centerline)',
            method=self.method_label
        )

        v_ghia_df = self.ghia_v.rename(columns={'x': 'position', 'v': 'velocity'}).assign(
            source='Ghia et al. (1982)',
            component='v (horizontal centerline)',
            method=self.method_label
        )

        return pd.concat([u_sim_df, v_sim_df, u_ghia_df, v_ghia_df], ignore_index=True)

    def compute_errors(self):
        """Compute error metrics against Ghia benchmark data.

        Note: Excludes boundary points (y=0, y=1 for u; x=0, x=1 for v)
        since collocated grids store values at cell centers, not boundaries.

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
        # Extract centerline data
        y_sim, u_sim = self._extract_centerline('u', 'y')
        x_sim, v_sim = self._extract_centerline('v', 'x')

        # Filter out boundary points from Ghia data
        # For u-velocity: exclude y=0 and y=1
        eps = 1e-6
        ghia_u_interior = self.ghia_u[(self.ghia_u['y'] > eps) & (self.ghia_u['y'] < 1.0 - eps)]

        # For v-velocity: exclude x=0 and x=1
        ghia_v_interior = self.ghia_v[(self.ghia_v['x'] > eps) & (self.ghia_v['x'] < 1.0 - eps)]

        # Interpolate simulation results at interior Ghia benchmark points
        u_interp_func = interp1d(y_sim, u_sim, kind='cubic', fill_value='extrapolate')
        u_sim_at_ghia = u_interp_func(ghia_u_interior['y'].values)
        u_error = u_sim_at_ghia - ghia_u_interior['u'].values

        v_interp_func = interp1d(x_sim, v_sim, kind='cubic', fill_value='extrapolate')
        v_sim_at_ghia = v_interp_func(ghia_v_interior['x'].values)
        v_error = v_sim_at_ghia - ghia_v_interior['v'].values

        # Compute error norms
        return {
            'u_l2': np.sqrt(np.sum(u_error**2)),
            'u_linf': np.max(np.abs(u_error)),
            'u_rms': np.sqrt(np.mean(u_error**2)),
            'v_l2': np.sqrt(np.sum(v_error**2)),
            'v_linf': np.max(np.abs(v_error)),
            'v_rms': np.sqrt(np.mean(v_error**2)),
        }

    def print_summary(self):
        """Print validation summary with error metrics."""
        errors = self.compute_errors()

        print("\n" + "="*70)
        print(f"{'VALIDATION SUMMARY':^70}")
        print("="*70)
        print(f"  Reynolds number: Re = {self.Re:.0f}")
        print(f"  Benchmark: Ghia et al. (1982), Re = {self.Re:.0f}")
        print(f"  Solution file: {self.h5_path.name}")
        print(f"  Note: Boundary points excluded (interior points only)")
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


def plot_validation(validators, output_path=None):
    """Plot validation against Ghia benchmark using seaborn.

    Handles both single and multiple methods automatically.

    Parameters
    ----------
    validators : GhiaValidator or list of GhiaValidator
        Single validator or list of validators to compare (must all have same Re)
    output_path : str or Path, optional
        Path to save figure. If None, figure is not saved.
    """
    import seaborn as sns

    # Normalize to list
    if not isinstance(validators, list):
        validators = [validators]

    # Combine all validation DataFrames
    dfs = [v.get_validation_dataframe() for v in validators]
    df = pd.concat(dfs, ignore_index=True)

    # Get Re for title (use first validator's Re)
    Re = validators[0].Re

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: U velocity
    df_u = df[df['component'] == 'u (vertical centerline)']
    df_u_sim = df_u[df_u['source'] == 'Simulation'].sort_values(['method', 'position'])
    df_u_ghia = df_u[df_u['source'] == 'Ghia et al. (1982)'].drop_duplicates(subset=['position']).sort_values('position')

    # Use seaborn with hue='method' for simulation data
    sns.lineplot(
        data=df_u_sim,
        x='velocity', y='position',
        hue='method',
        ax=axes[0],
        linewidth=2.5,
        alpha=0.8,
        sort=False,
        estimator=None
    )
    # Ghia benchmark as scatter
    sns.scatterplot(
        data=df_u_ghia,
        x='velocity', y='position',
        ax=axes[0],
        marker='x', s=50, color='black',
        label='Ghia et al. (1982)', zorder=10
    )

    axes[0].set_xlabel("$u$", fontsize=12)
    axes[0].set_ylabel("$y$", fontsize=12)
    axes[0].set_title("U velocity (vertical centerline)", fontweight='bold')
    axes[0].legend(frameon=True, loc="best")
    axes[0].grid(True, alpha=0.3)

    # Right panel: V velocity
    df_v = df[df['component'] == 'v (horizontal centerline)']
    df_v_sim = df_v[df_v['source'] == 'Simulation'].sort_values(['method', 'position'])
    df_v_ghia = df_v[df_v['source'] == 'Ghia et al. (1982)'].drop_duplicates(subset=['position']).sort_values('position')

    # Use seaborn with hue='method' for simulation data
    sns.lineplot(
        data=df_v_sim,
        x='position', y='velocity',
        hue='method',
        ax=axes[1],
        linewidth=2.5,
        alpha=0.8,
        sort=False,
        estimator=None
    )
    # Ghia benchmark as scatter
    sns.scatterplot(
        data=df_v_ghia,
        x='position', y='velocity',
        ax=axes[1],
        marker='x', s=50, color='black',
        label='Ghia et al. (1982)', zorder=10
    )

    axes[1].set_xlabel("$x$", fontsize=12)
    axes[1].set_ylabel("$v$", fontsize=12)
    axes[1].set_title("V velocity (horizontal centerline)", fontweight='bold')
    axes[1].legend(frameon=True, loc="best")
    axes[1].grid(True, alpha=0.3)

    # Set overall title
    n_methods = len(validators)
    if n_methods == 1:
        title = f"Ghia Benchmark Validation (Re = {Re:.0f})"
    else:
        title = f"Ghia Benchmark Validation: Method Comparison (Re = {Re:.0f})"

    fig.suptitle(title, fontweight="bold", fontsize=14, y=1.00)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=300)
        print(f"Validation plot saved to: {output_path}")
