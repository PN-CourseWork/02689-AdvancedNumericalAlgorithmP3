"""Abstract base solver for lid-driven cavity problem."""

from abc import ABC, abstractmethod
import os
import time

import numpy as np
import mlflow

from dataclasses import asdict
from .datastructures import TimeSeries, Metrics, Fields


class LidDrivenCavitySolver(ABC):
    """Abstract base solver for lid-driven cavity problem.

    Handles:
    - Parameter management (input configuration)
    - Metrics tracking (output results)
    - Iteration loop with residual computation
    - MLflow logging

    Subclasses must:
    - Set Parameters class attribute (e.g., FVParameters)
    - Implement step() - perform one iteration
    - Call _init_fields(x, y) after setting up grid
    - Implement _compute_algebraic_residuals() for Ax-b residuals
    """

    Parameters = None  # Subclasses set this to FVParameters or SpectralParameters

    def __init__(self, params=None, **kwargs):
        """Initialize solver with parameters.

        Parameters
        ----------
        params : Parameters, optional
            Parameters object. If not provided, kwargs are used to create params.
        **kwargs
            Configuration parameters passed to Parameters class if params is None.
        """
        if params is None:
            if self.Parameters is None:
                raise ValueError("Subclass must define Parameters class attribute")
            params = self.Parameters(**kwargs)

        self.params = params
        self.metrics = Metrics()
        self.fields = None  # Initialized by subclass via _init_fields()
        self.time_series = None  # Populated after solve()

    def _init_fields(self, x: np.ndarray, y: np.ndarray):
        """Initialize output fields with grid coordinates.

        Called by subclass after setting up grid. Pre-allocates the Fields
        dataclass that will hold the final solution.

        Parameters
        ----------
        x : np.ndarray
            X coordinates of all grid points (1D array)
        y : np.ndarray
            Y coordinates of all grid points (1D array)
        """
        n_points = len(x)
        self.fields = Fields(
            u=np.zeros(n_points),
            v=np.zeros(n_points),
            p=np.zeros(n_points),
            x=x.copy(),
            y=y.copy(),
        )

    @abstractmethod
    def step(self):
        """Perform one iteration/time step of the solver.

        Returns
        -------
        u, v, p : np.ndarray
            Updated velocity and pressure fields
        """
        pass

    def _finalize_fields(self):
        """Copy final solution from internal arrays to output fields.

        Default implementation copies directly. Override if transformation
        is needed (e.g., spectral pressure interpolation from inner grid).
        """
        self.fields.u[:] = self.arrays.u
        self.fields.v[:] = self.arrays.v
        self.fields.p[:] = self.arrays.p

    @abstractmethod
    def _compute_algebraic_residuals(self):
        """Compute algebraic residuals (Ax - b) for the discretized equations.

        Returns
        -------
        dict
            Dictionary with keys 'u_residual', 'v_residual', 'continuity_residual'
            containing L2 norms of the algebraic residuals.
        """
        pass

    def _store_results(self, residual_history, final_iter_count, is_converged,
                       wall_time, energy_history=None, enstrophy_history=None,
                       palinstrophy_history=None, max_timeseries_points: int = 1000):
        """Store solve results in self.fields, self.time_series, and self.metrics."""
        # Extract residuals
        rel_iter_residuals = [r['rel_iter'] for r in residual_history]
        u_residuals = [r['u_eq'] for r in residual_history]
        v_residuals = [r['v_eq'] for r in residual_history]
        continuity_residuals = [r.get('continuity', None) for r in residual_history]

        # Check if all continuity residuals are None
        if all(c is None for c in continuity_residuals):
            continuity_residuals = None

        # Copy final solution to output fields
        self._finalize_fields()

        # Downsample time series to max_timeseries_points
        def downsample(data):
            if data is None or len(data) <= max_timeseries_points:
                return data
            indices = np.linspace(0, len(data) - 1, max_timeseries_points, dtype=int)
            return [data[i] for i in indices]

        # Create time series (downsampled)
        self.time_series = TimeSeries(
            rel_iter_residual=downsample(rel_iter_residuals),
            u_residual=downsample(u_residuals),
            v_residual=downsample(v_residuals),
            continuity_residual=downsample(continuity_residuals),
            energy=downsample(energy_history),
            enstrophy=downsample(enstrophy_history),
            palinstrophy=downsample(palinstrophy_history),
        )

        # Update metrics with convergence info (use FINAL values, not downsampled)
        self.metrics = Metrics(
            iterations=final_iter_count,
            converged=is_converged,
            final_residual=rel_iter_residuals[-1] if rel_iter_residuals else float('inf'),
            wall_time_seconds=wall_time,
            u_momentum_residual=u_residuals[-1] if u_residuals else 0.0,
            v_momentum_residual=v_residuals[-1] if v_residuals else 0.0,
            continuity_residual=continuity_residuals[-1] if continuity_residuals else 0.0,
            final_energy=energy_history[-1] if energy_history else 0.0,
            final_enstrophy=enstrophy_history[-1] if enstrophy_history else 0.0,
            final_palinstrophy=palinstrophy_history[-1] if palinstrophy_history else 0.0,
        )

    def solve(self, tolerance: float = None, max_iter: int = None):
        """Solve the lid-driven cavity problem using iterative stepping.

        This method implements the common iteration loop with residual calculation.
        Subclasses implement step() to define one iteration.

        Stores results in solver attributes:
        - self.fields : Fields dataclass with solution fields
        - self.time_series : TimeSeries dataclass with time series data
        - self.metrics : Metrics dataclass with solver metrics

        Parameters
        ----------
        tolerance : float, optional
            Convergence tolerance. If None, uses params.tolerance.
        max_iter : int, optional
            Maximum iterations. If None, uses params.max_iterations.
        """
        # Use params values if not explicitly provided
        if tolerance is None:
            tolerance = self.params.tolerance
        if max_iter is None:
            max_iter = self.params.max_iterations

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
        mlflow_time = 0.0  # Track time spent on MLflow logging
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

            # Compute algebraic residuals (Ax - b)
            eq_residuals = self._compute_algebraic_residuals()

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

                # Live MLflow logging every 50 iterations (timed separately)
                if mlflow.active_run():
                    t_log_start = time.time()
                    live_metrics = {
                        "rel_iter_residual": rel_iter_residual,
                        "u_residual": eq_residuals['u_residual'],
                        "v_residual": eq_residuals['v_residual'],
                    }
                    if 'continuity_residual' in eq_residuals:
                        live_metrics["continuity_residual"] = eq_residuals['continuity_residual']
                    if i >= 10:  # After warmup, also log conserved quantities
                        live_metrics["energy"] = energy_history[-1]
                        live_metrics["enstrophy"] = enstrophy_history[-1]
                    mlflow.log_metrics(live_metrics, step=i)
                    mlflow_time += time.time() - t_log_start

            if is_converged:
                print(f"Converged at iteration {i}")
                break

        time_end = time.time()
        wall_time = time_end - time_start - mlflow_time  # Exclude MLflow logging time
        print(f"Solver finished in {wall_time:.2f} seconds (excl. {mlflow_time:.2f}s logging).")

        # Store results
        self._store_results(
            residual_history, final_iter_count, is_converged, wall_time,
            energy_history, enstrophy_history, palinstrophy_history
        )

    def save(self, filepath):
        """Save complete solver state to HDF5 file.

        Saves params, metrics, time_series, and fields for later analysis.

        Parameters
        ----------
        filepath : str or Path
            Output file path (use .h5 extension).
        """
        from pathlib import Path

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        import pandas as pd
        with pd.HDFStore(filepath, mode='w', complevel=5) as store:
            store['params'] = self.params.to_dataframe()
            store['metrics'] = self.metrics.to_dataframe()
            store['time_series'] = self.time_series.to_dataframe()
            store['fields'] = self.fields.to_dataframe()

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

    def _compute_gradient(self, field: np.ndarray) -> tuple:
        """Compute gradient of scalar field using finite differences."""
        dx, dy = self.dx_min, self.dy_min
        shape = getattr(self, 'shape_full', (self.params.nx, self.params.ny))
        field_2d = np.pad(field.reshape(shape), 1, mode='edge')
        df_dx = (field_2d[1:-1, 2:] - field_2d[1:-1, :-2]) / (2 * dx)
        df_dy = (field_2d[2:, 1:-1] - field_2d[:-2, 1:-1]) / (2 * dy)
        return df_dx.ravel(), df_dy.ravel()

    def _compute_vorticity(self) -> np.ndarray:
        """Compute vorticity ω = ∂v/∂x - ∂u/∂y using finite differences."""
        dv_dx, _ = self._compute_gradient(self.arrays.v)
        _, du_dy = self._compute_gradient(self.arrays.u)
        return dv_dx - du_dy

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
    # MLflow Integration
    # ========================================================================

    def mlflow_start(self, experiment_name: str, run_name: str, parent_run_name: str = None):
        """Start MLflow run and log parameters.

        Parameters
        ----------
        experiment_name : str
            Name of the MLflow experiment.
        run_name : str
            Name of the run within the experiment.
        parent_run_name : str, optional
            If specified, creates a nested run under a parent with this name.
            Parent is created if it doesn't exist, or resumed if it does.
        """
        mlflow.login()

        # Databricks requires absolute paths
        experiment_name = f"/Shared/ANA-P3/{experiment_name}"

        if mlflow.get_experiment_by_name(experiment_name) is None:
            mlflow.create_experiment(name=experiment_name)

        mlflow.set_experiment(experiment_name)

        # Handle parent run if specified
        if parent_run_name:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            client = mlflow.tracking.MlflowClient()

            # Search for existing parent run
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"tags.mlflow.runName = '{parent_run_name}' AND tags.is_parent = 'true'",
                max_results=1
            )

            if runs:
                # Resume existing parent
                parent_run_id = runs[0].info.run_id
            else:
                # Create new parent run
                parent_run = client.create_run(
                    experiment_id=experiment.experiment_id,
                    run_name=parent_run_name,
                    tags={"is_parent": "true"}
                )
                parent_run_id = parent_run.info.run_id

            # Start nested child run
            mlflow.start_run(
                run_id=parent_run_id,
                log_system_metrics=False
            )
            mlflow.start_run(
                run_name=run_name,
                nested=True,
                log_system_metrics=True
            )
            self._mlflow_nested = True
        else:
            mlflow.start_run(log_system_metrics=True, run_name=run_name)
            self._mlflow_nested = False

        # Log all parameters from the params dataclass
        mlflow.log_params(asdict(self.params))

        # Log HPC job info if running on LSF cluster
        job_id = os.environ.get("LSB_JOBID")
        if job_id:
            mlflow.set_tag("lsf.job_id", job_id)
            job_index = os.environ.get("LSB_JOBINDEX", "")
            job_name = os.environ.get("LSB_JOBNAME", "")
            description = f"HPC Job: {job_name} (ID: {job_id}"
            if job_index:
                description += f", Index: {job_index}"
            description += ")"
            mlflow.set_tag("mlflow.note.content", description)

    def mlflow_end(self):
        """End MLflow run and log final metrics."""
        # Log final metrics from the metrics dataclass
        mlflow.log_metrics(asdict(self.metrics))

        # End child run
        mlflow.end_run()

        # End parent run if nested
        if getattr(self, '_mlflow_nested', False):
            mlflow.end_run()

    def mlflow_log_artifact(self, filepath: str):
        """Log an artifact (e.g., saved HDF5 file) to MLflow.

        Parameters
        ----------
        filepath : str
            Path to the file to log as artifact.
        """
        mlflow.log_artifact(filepath)


