"""Abstract base solver for lid-driven cavity problem."""

from abc import ABC, abstractmethod
import numpy as np
from dataclasses import replace, asdict
import mlflow

from .datastructures import TimeSeries


class LidDrivenCavitySolver(ABC):
    """Abstract base solver for lid-driven cavity problem.

    Handles:
    - Configuration management
    - Iteration loop with residual computation
    - Result storage

    Subclasses must:
    - Set Config and ResultFields class attributes
    - Implement step() - perform one iteration
    - Implement _create_result_fields() - create result dataclass
    - Extend __init__() for solver-specific setup
    """

    Config = None
    ResultFields = None

    def __init__(self, config=None, **kwargs):
        """Initialize solver with configuration.

        Parameters
        ----------
        config : Config, optional
            Configuration object. If not provided, kwargs are used to create config.
        **kwargs
            Configuration parameters passed to Config class if config is None.
        """
        # Create config from kwargs if not provided
        if config is None:
            if self.Config is None:
                raise ValueError("Subclass must define Config class attribute")
            config = self.Config(**kwargs)

        self.config = config

    @abstractmethod
    def step(self):
        """Perform one iteration/time step of the solver.

        This method should:
        1. Update the solution fields (u, v, p)
        2. Return the updated fields as a tuple (u, v, p)

        The fields should be stored as instance variables that can be accessed
        for residual computation.

        Returns
        -------
        u : np.ndarray
            Updated u velocity field
        v : np.ndarray
            Updated v velocity field
        p : np.ndarray
            Updated pressure field
        """
        pass

    @abstractmethod
    def _create_result_fields(self):
        """Create result fields dataclass with solver-specific data.

        Must return an instance of self.ResultFields.
        """
        pass

    @abstractmethod
    def _compute_derivatives(self):
        """Compute velocity and pressure derivatives needed for residuals.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'du_dx', 'du_dy', 'dv_dx', 'dv_dy': velocity derivatives
            - 'dp_dx', 'dp_dy': pressure gradient components
            - 'lap_u', 'lap_v': velocity Laplacians
            All arrays should be 1D and have same length (all grid points).
        """
        pass

    def _compute_equation_residuals(self):
        """Compute physical equation residuals (discretization-independent).

        Evaluates how well current solution satisfies:
        - U-momentum: (u·∇)u + ∂p/∂x - (1/Re)∇²u = 0
        - V-momentum: (u·∇)v + ∂p/∂y - (1/Re)∇²v = 0
        - Continuity: ∂u/∂x + ∂v/∂y = 0

        Returns
        -------
        dict
            Dictionary with keys 'u_residual', 'v_residual', 'continuity_residual'
            containing L2 norms of the equation residuals.
        """
        # Get derivatives from solver-specific implementation
        deriv = self._compute_derivatives()

        u = self.arrays.u
        v = self.arrays.v
        nu = 1.0 / self.config.Re

        # U-momentum residual: R_u = (u·∇)u + ∂p/∂x - (1/Re)∇²u
        conv_u = u * deriv['du_dx'] + v * deriv['du_dy']
        R_u = conv_u + deriv['dp_dx'] - nu * deriv['lap_u']

        # V-momentum residual: R_v = (u·∇)v + ∂p/∂y - (1/Re)∇²v
        conv_v = u * deriv['dv_dx'] + v * deriv['dv_dy']
        R_v = conv_v + deriv['dp_dy'] - nu * deriv['lap_v']

        # Continuity residual: R_c = ∂u/∂x + ∂v/∂y
        R_c = deriv['du_dx'] + deriv['dv_dy']

        return {
            'u_residual': np.linalg.norm(R_u),
            'v_residual': np.linalg.norm(R_v),
            'continuity_residual': np.linalg.norm(R_c)
        }

    def _store_results(self, residual_history, final_iter_count, is_converged,
                       energy_history=None, enstrophy_history=None, palinstrophy_history=None):
        """Store solve results in self.fields, self.time_series, and self.metadata."""
        # Extract residuals
        rel_iter_residuals = [r['rel_iter'] for r in residual_history]
        u_residuals = [r['u_eq'] for r in residual_history]
        v_residuals = [r['v_eq'] for r in residual_history]
        continuity_residuals = [r.get('continuity', None) for r in residual_history]

        # Check if all continuity residuals are None
        if all(c is None for c in continuity_residuals):
            continuity_residuals = None

        # Create fields (subclasses can override _create_result_fields)
        self.fields = self._create_result_fields()

        # Create time series (same for all solvers)
        self.time_series = TimeSeries(
            rel_iter_residual=rel_iter_residuals,
            u_residual=u_residuals,
            v_residual=v_residuals,
            continuity_residual=continuity_residuals,
            energy=energy_history,
            enstrophy=enstrophy_history,
            palinstrophy=palinstrophy_history,
        )

        # Update metadata with convergence info
        self.metadata = replace(
            self.config,
            iterations=final_iter_count,
            converged=is_converged,
            final_residual=rel_iter_residuals[-1] if rel_iter_residuals else float('inf'),
        )

    def solve(self, tolerance: float = None, max_iter: int = None):
        """Solve the lid-driven cavity problem using iterative stepping.

        This method implements the common iteration loop with residual calculation.
        Subclasses implement step() to define one iteration.

        Stores results in solver attributes:
        - self.fields : Fields dataclass with solution fields
        - self.time_series : TimeSeries dataclass with time series data
        - self.metadata : Metadata dataclass with solver metadata

        Parameters
        ----------
        tolerance : float, optional
            Convergence tolerance. If None, uses config.tolerance.
        max_iter : int, optional
            Maximum iterations. If None, uses config.max_iterations.
        """
        import time

        # Use config values if not explicitly provided
        if tolerance is None:
            tolerance = self.config.tolerance
        if max_iter is None:
            max_iter = self.config.max_iterations

        # Store previous iteration for residual calculation
        u_prev = self.arrays.u.copy()
        v_prev = self.arrays.v.copy()

        # Residual history
        residual_history = []

        # Quantity tracking (energy, enstrophy, palinstrophy)
        energy_history = []
        enstrophy_history = []
        palinstrophy_history = []

        time_start = time.time()
        final_iter_count = 0
        is_converged = False

        for i in range(max_iter):
            final_iter_count = i + 1

            # Perform one iteration
            self.arrays.u, self.arrays.v, self.arrays.p = self.step()

            # Calculate normalized solution change: ||u^{n+1} - u^n||_2 / ||u^n||_2
            u_change_norm = np.linalg.norm(self.arrays.u - u_prev)
            v_change_norm = np.linalg.norm(self.arrays.v - v_prev)

            u_prev_norm = np.linalg.norm(u_prev) + 1e-12
            v_prev_norm = np.linalg.norm(v_prev) + 1e-12

            u_solution_change = u_change_norm / u_prev_norm
            v_solution_change = v_change_norm / v_prev_norm
            rel_iter_residual = max(u_solution_change, v_solution_change)

            # Compute equation residuals
            eq_residuals = self._compute_equation_residuals()

            # Only store residual history after first 10 iterations
            if i >= 10:
                residual_history.append({
                    "rel_iter": rel_iter_residual,
                    "u_eq": eq_residuals['u_residual'],
                    "v_eq": eq_residuals['v_residual'],
                    "continuity": eq_residuals.get('continuity_residual', None)
                })
                # Calculate and store conserved quantities
                energy_history.append(self._compute_energy())
                enstrophy_history.append(self._compute_enstrophy())
                palinstrophy_history.append(self._compute_palinstrophy())

            # Update previous iteration
            u_prev = self.arrays.u.copy()
            v_prev = self.arrays.v.copy()

            # Check convergence (only after warmup period)
            if i >= 10:
                is_converged = rel_iter_residual < tolerance
            else:
                is_converged = False

            if i % 50 == 0 or is_converged:
                print(f"Iteration {i}: u_res={u_solution_change:.6e}, v_res={v_solution_change:.6e}")

            if is_converged:
                print(f"Converged at iteration {i}")
                break

        time_end = time.time()
        print(f"Solver finished in {time_end - time_start:.2f} seconds.")

        # Store results
        self._store_results(
            residual_history, final_iter_count, is_converged,
            energy_history, enstrophy_history, palinstrophy_history
        )

    def save(self, filepath):
        """Save results to HDF5 file using pandas HDFStore.

        Parameters
        ----------
        filepath : str or Path
            Output file path.
        """
        from pathlib import Path
        import pandas as pd

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Convert dataclasses to DataFrames
        with pd.HDFStore(filepath, 'w') as store:
            store['metadata'] = self.metadata.to_dataframe()
            store['fields'] = self.fields.to_dataframe()
            store['time_series'] = self.time_series.to_dataframe()

    # =========================================================================
    # Conserved Quantity Calculations (for comparison with Saad reference data)
    # =========================================================================

    def _compute_energy(self) -> float:
        """Compute kinetic energy: E = 0.5 * ∫ (u² + v²) dA."""
        u = self.arrays.u
        v = self.arrays.v
        dA = self._get_cell_area()
        return 0.5 * float(np.sum(u * u + v * v) * dA)

    def _compute_enstrophy(self) -> float:
        """Compute enstrophy: Z = 0.5 * ∫ ω² dA, where ω = ∂v/∂x - ∂u/∂y."""
        omega = self._compute_vorticity()
        dA = self._get_cell_area()
        return 0.5 * float(np.sum(omega * omega) * dA)

    def _compute_palinstrophy(self) -> float:
        """Compute palinstrophy: P = ∫ (∂ω/∂x)² + (∂ω/∂y)² dA."""
        omega = self._compute_vorticity()
        domega_dx, domega_dy = self._compute_gradient(omega)
        dA = self._get_cell_area()
        return float(np.sum(domega_dx**2 + domega_dy**2) * dA)

    def _compute_vorticity(self) -> np.ndarray:
        """Compute vorticity ω = ∂v/∂x - ∂u/∂y."""
        deriv = self._compute_derivatives()
        return deriv['dv_dx'] - deriv['du_dy']

    def _compute_gradient(self, field: np.ndarray) -> tuple:
        """Compute gradient of a scalar field. Subclasses should override for accuracy."""
        # Default implementation uses finite differences
        dx = getattr(self, 'dx_min', 1.0 / np.sqrt(len(field)))
        dy = getattr(self, 'dy_min', dx)

        # Try to reshape to 2D grid
        n = len(field)
        nx = int(np.sqrt(n))
        if nx * nx == n:
            field_2d = field.reshape(nx, nx)
            df_dx = np.zeros_like(field_2d)
            df_dy = np.zeros_like(field_2d)
            # Central differences (interior)
            df_dx[1:-1, :] = (field_2d[2:, :] - field_2d[:-2, :]) / (2 * dx)
            df_dy[:, 1:-1] = (field_2d[:, 2:] - field_2d[:, :-2]) / (2 * dy)
            return df_dx.ravel(), df_dy.ravel()
        else:
            return np.zeros_like(field), np.zeros_like(field)

    def _get_cell_area(self) -> float:
        """Get cell area for integration. Subclasses should override."""
        dx = getattr(self, 'dx_min', None)
        dy = getattr(self, 'dy_min', None)
        if dx is not None and dy is not None:
            return dx * dy
        # Fallback: assume unit domain
        n = len(self.arrays.u)
        return 1.0 / n

    # ========================================================================
    # MLflow
    # ========================================================================

    def mlflow_start(self, experiment_name, run_name):
        """Start MLflow run (rank 0 only)."""

        mlflow.login()

        # Databricks requires absolute paths
        experiment_name = f"/Shared/ANA-P3/{experiment_name}"

        if mlflow.get_experiment_by_name(experiment_name) is None:
            mlflow.create_experiment(name=experiment_name)

        mlflow.set_experiment(experiment_name)
        mlflow.start_run(log_system_metrics=True, run_name=run_name)
        mlflow.log_params(asdict(self.config))
        #mlflow.log_params(dict(self.Config))

    def mlflow_end(self):
        """End MLflow run with metrics (rank 0 only)."""


        mlflow.log_metrics({
            "Iterations": self.config.iterations,
            "Converged": self.config.converged,
            "FinalResidual": self.config.final_residual,
        })
        mlflow.end_run()


