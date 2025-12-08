"""Abstract base solver for lid-driven cavity problem."""

from abc import ABC, abstractmethod
import logging
import os
import time
from pathlib import Path

import numpy as np
import pyvista as pv
import mlflow

from dataclasses import asdict
from .datastructures import TimeSeries, Metrics, Fields

log = logging.getLogger(__name__)


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

    def _store_results(
        self,
        residual_history,
        final_iter_count,
        is_converged,
        wall_time,
        energy_history=None,
        enstrophy_history=None,
        palinstrophy_history=None,
        max_timeseries_points: int = 1000,
    ):
        """Store solve results in self.fields, self.time_series, and self.metrics."""
        # Extract residuals
        rel_iter_residuals = [r["rel_iter"] for r in residual_history]
        u_residuals = [r["u_eq"] for r in residual_history]
        v_residuals = [r["v_eq"] for r in residual_history]
        continuity_residuals = [r.get("continuity", None) for r in residual_history]

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

        # Compute vortex metrics (streamfunction-based)
        try:
            vortex_metrics = self.compute_vortex_metrics()
        except Exception as e:
            log.warning(f"Failed to compute vortex metrics: {e}")
            vortex_metrics = {}

        # Update metrics with convergence info (use FINAL values, not downsampled)
        self.metrics = Metrics(
            iterations=final_iter_count,
            converged=is_converged,
            final_residual=rel_iter_residuals[-1]
            if rel_iter_residuals
            else float("inf"),
            wall_time_seconds=wall_time,
            u_momentum_residual=u_residuals[-1] if u_residuals else 0.0,
            v_momentum_residual=v_residuals[-1] if v_residuals else 0.0,
            continuity_residual=continuity_residuals[-1]
            if continuity_residuals
            else 0.0,
            final_energy=energy_history[-1] if energy_history else 0.0,
            final_enstrophy=enstrophy_history[-1] if enstrophy_history else 0.0,
            final_palinstrophy=palinstrophy_history[-1]
            if palinstrophy_history
            else 0.0,
            # Vortex metrics
            psi_min=vortex_metrics.get("psi_min", 0.0),
            psi_min_x=vortex_metrics.get("psi_min_x", 0.0),
            psi_min_y=vortex_metrics.get("psi_min_y", 0.0),
            omega_center=vortex_metrics.get("omega_center", 0.0),
            omega_max=vortex_metrics.get("omega_max", 0.0),
            omega_max_x=vortex_metrics.get("omega_max_x", 0.0),
            omega_max_y=vortex_metrics.get("omega_max_y", 0.0),
            psi_BR=vortex_metrics.get("psi_BR", 0.0),
            psi_BR_x=vortex_metrics.get("psi_BR_x", 0.0),
            psi_BR_y=vortex_metrics.get("psi_BR_y", 0.0),
            omega_BR=vortex_metrics.get("omega_BR", 0.0),
            psi_BL=vortex_metrics.get("psi_BL", 0.0),
            psi_BL_x=vortex_metrics.get("psi_BL_x", 0.0),
            psi_BL_y=vortex_metrics.get("psi_BL_y", 0.0),
            omega_BL=vortex_metrics.get("omega_BL", 0.0),
            psi_TL=vortex_metrics.get("psi_TL", 0.0),
            psi_TL_x=vortex_metrics.get("psi_TL_x", 0.0),
            psi_TL_y=vortex_metrics.get("psi_TL_y", 0.0),
            omega_TL=vortex_metrics.get("omega_TL", 0.0),
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
                residual_history.append(
                    {
                        "rel_iter": rel_iter_residual,
                        "u_eq": eq_residuals["u_residual"],
                        "v_eq": eq_residuals["v_residual"],
                        "continuity": eq_residuals.get("continuity_residual", None),
                    }
                )
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
                log.info(
                    f"Iteration {i}: u_res={u_solution_change:.6e}, v_res={v_solution_change:.6e}"
                )

                # Live MLflow logging every 50 iterations (timed separately)
                if mlflow.active_run():
                    t_log_start = time.time()
                    live_metrics = {
                        "rel_iter_residual": rel_iter_residual,
                        "u_residual": eq_residuals["u_residual"],
                        "v_residual": eq_residuals["v_residual"],
                    }
                    if "continuity_residual" in eq_residuals:
                        live_metrics["continuity_residual"] = eq_residuals[
                            "continuity_residual"
                        ]
                    if i >= 10:  # After warmup, also log conserved quantities
                        live_metrics["energy"] = energy_history[-1]
                        live_metrics["enstrophy"] = enstrophy_history[-1]
                    mlflow.log_metrics(live_metrics, step=i)
                    mlflow_time += time.time() - t_log_start

            if is_converged:
                log.info(f"Converged at iteration {i}")
                break

        time_end = time.time()
        wall_time = time_end - time_start - mlflow_time  # Exclude MLflow logging time
        log.info(
            f"Solver finished in {wall_time:.2f} seconds (excl. {mlflow_time:.2f}s logging)."
        )

        # Store results
        self._store_results(
            residual_history,
            final_iter_count,
            is_converged,
            wall_time,
            energy_history,
            enstrophy_history,
            palinstrophy_history,
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

        with pd.HDFStore(filepath, mode="w", complevel=5) as store:
            store["params"] = self.params.to_dataframe()
            store["metrics"] = self.metrics.to_dataframe()
            store["time_series"] = self.time_series.to_dataframe()
            store["fields"] = self.fields.to_dataframe()

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
        """Compute palinstrophy: P = 0.5 * ∫ ||∇ω||² dA."""
        omega = self._compute_vorticity()
        domega_dx, domega_dy = self._compute_gradient(omega)
        dA = self._get_cell_area()
        return 0.5 * float(np.sum(domega_dx**2 + domega_dy**2) * dA)

    def _compute_gradient(
        self, field: np.ndarray, bc_walls: float = 0.0, bc_lid: float = None
    ) -> tuple:
        """Compute gradient of scalar field using finite differences.

        Uses proper ghost cell values for Dirichlet BCs:
        ghost = 2 * wall_value - interior_value

        Parameters
        ----------
        field : np.ndarray
            Scalar field values at cell centers
        bc_walls : float
            Dirichlet BC value at walls (bottom, left, right). Default 0.
        bc_lid : float or None
            Dirichlet BC value at top lid. If None, uses bc_walls.
        """
        if bc_lid is None:
            bc_lid = bc_walls

        dx, dy = self.dx_min, self.dy_min
        shape = getattr(self, "shape_full", (self.params.nx, self.params.ny))
        field_2d = field.reshape(shape)  # shape = (ny, nx)
        ny, nx = shape

        # Create padded array with proper ghost cell values for Dirichlet BCs
        # Ghost value = 2 * BC_value - interior_value
        field_padded = np.zeros((ny + 2, nx + 2), dtype=field.dtype)
        field_padded[1:-1, 1:-1] = field_2d

        # Bottom boundary (j=0 in original, row 0 in padded is ghost)
        field_padded[0, 1:-1] = 2 * bc_walls - field_2d[0, :]

        # Top boundary (j=ny-1 in original, row ny+1 in padded is ghost)
        field_padded[-1, 1:-1] = 2 * bc_lid - field_2d[-1, :]

        # Left boundary (i=0 in original, col 0 in padded is ghost)
        field_padded[1:-1, 0] = 2 * bc_walls - field_2d[:, 0]

        # Right boundary (i=nx-1 in original, col nx+1 in padded is ghost)
        field_padded[1:-1, -1] = 2 * bc_walls - field_2d[:, -1]

        # Corners (average of adjacent ghost values)
        field_padded[0, 0] = 0.5 * (field_padded[0, 1] + field_padded[1, 0])
        field_padded[0, -1] = 0.5 * (field_padded[0, -2] + field_padded[1, -1])
        field_padded[-1, 0] = 0.5 * (field_padded[-1, 1] + field_padded[-2, 0])
        field_padded[-1, -1] = 0.5 * (field_padded[-1, -2] + field_padded[-2, -1])

        # Central differences
        df_dx = (field_padded[1:-1, 2:] - field_padded[1:-1, :-2]) / (2 * dx)
        df_dy = (field_padded[2:, 1:-1] - field_padded[:-2, 1:-1]) / (2 * dy)
        return df_dx.ravel(), df_dy.ravel()

    def _compute_vorticity(self) -> np.ndarray:
        """Compute vorticity ω = ∂v/∂x - ∂u/∂y using finite differences.

        Uses proper boundary conditions for lid-driven cavity:
        - u: bc_walls=0, bc_lid=lid_velocity
        - v: bc_walls=0, bc_lid=0
        """
        # Get lid velocity from params (default 1.0 for lid-driven cavity)
        lid_velocity = getattr(self.params, "lid_velocity", 1.0)

        # v has zero BC on all walls including lid
        dv_dx, _ = self._compute_gradient(self.arrays.v, bc_walls=0.0, bc_lid=0.0)

        # u has zero BC on walls, lid_velocity on top
        _, du_dy = self._compute_gradient(self.arrays.u, bc_walls=0.0, bc_lid=lid_velocity)

        return dv_dx - du_dy

    def _get_cell_area(self) -> float:
        """Get cell area for integration. Subclasses should override."""
        dx = getattr(self, "dx_min", None)
        dy = getattr(self, "dy_min", None)
        if dx is not None and dy is not None:
            return dx * dy
        # Fallback: assume unit domain
        n = len(self.arrays.u)
        return 1.0 / n

    # =========================================================================
    # VTK Export - to_vtk() creates StructuredGrid with all fields
    # =========================================================================

    def to_vtk(self) -> pv.StructuredGrid:
        """Export solution to VTK StructuredGrid with all fields and metadata.

        Creates a structured grid with:
        - Primary fields: u, v, p
        - Derived fields: velocity_magnitude, vorticity, velocity (vector)
        - Metadata: Re, N, solver name

        Subclasses may override to use native differentiation for derived fields.

        Returns
        -------
        pv.StructuredGrid
            Solution on structured grid, ready for VTS export
        """
        # Get unique sorted coordinates
        x_unique = np.sort(np.unique(self.fields.x))
        y_unique = np.sort(np.unique(self.fields.y))
        nx, ny = len(x_unique), len(y_unique)

        # Reshape fields to 2D grid: U_2d[j, i] = u at (x_unique[i], y_unique[j])
        indices = np.lexsort((self.fields.x, self.fields.y))
        u_sorted = self.fields.u[indices]
        v_sorted = self.fields.v[indices]
        p_sorted = self.fields.p[indices]

        U_2d = u_sorted.reshape(ny, nx)
        V_2d = v_sorted.reshape(ny, nx)
        P_2d = p_sorted.reshape(ny, nx)

        # Create 3D grid (z=0 plane)
        X, Y = np.meshgrid(x_unique, y_unique)
        Z = np.zeros_like(X)
        grid = pv.StructuredGrid(X, Y, Z)

        # Add primary fields - use Fortran order to match VTK's column-major point ordering
        grid["u"] = U_2d.ravel("F")
        grid["v"] = V_2d.ravel("F")
        grid["pressure"] = P_2d.ravel("F")

        # Add derived fields
        grid["velocity_magnitude"] = np.sqrt(U_2d**2 + V_2d**2).ravel("F")

        # Compute vorticity using native differentiation
        vorticity = self._compute_vorticity_for_export(U_2d, V_2d, x_unique, y_unique)
        grid["vorticity"] = vorticity.ravel("F")

        # Add velocity vector field
        vectors = np.zeros((nx * ny, 3))
        vectors[:, 0] = U_2d.ravel("F")
        vectors[:, 1] = V_2d.ravel("F")
        grid["velocity"] = vectors

        # Add metadata
        grid.field_data["Re"] = np.array([self.params.Re])
        grid.field_data["N"] = np.array([self.params.nx])
        grid.field_data["solver"] = np.array([self.params.name])

        return grid

    def _compute_vorticity_for_export(
        self, U_2d: np.ndarray, V_2d: np.ndarray, x: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """Compute vorticity for VTK export. Override for native differentiation.

        Default uses scipy RectBivariateSpline for smooth derivatives.

        Parameters
        ----------
        U_2d, V_2d : np.ndarray
            2D velocity arrays (ny, nx)
        x, y : np.ndarray
            1D coordinate arrays

        Returns
        -------
        np.ndarray
            Vorticity field (ny, nx)
        """
        from scipy.interpolate import RectBivariateSpline

        U_spline = RectBivariateSpline(y, x, U_2d)
        V_spline = RectBivariateSpline(y, x, V_2d)
        dvdx = V_spline(y, x, dx=1)
        dudy = U_spline(y, x, dy=1)
        return dvdx - dudy

    def compute_global_quantities(self) -> dict:
        """Compute global quantities E, Z, P for the current solution.

        Returns
        -------
        dict
            {'E': kinetic_energy, 'Z': enstrophy, 'P': palinstrophy}
        """
        return {
            "E": self._compute_energy(),
            "Z": self._compute_enstrophy(),
            "P": self._compute_palinstrophy(),
        }

    # =========================================================================
    # Vortex Detection (Streamfunction-based, for comparison with Saad/Botella)
    # =========================================================================

    def _compute_streamfunction(self) -> tuple:
        """Compute streamfunction ψ by solving ∇²ψ = -ω.

        Uses Poisson solver with ψ=0 on all boundaries (closed cavity).

        Returns
        -------
        psi_2d : np.ndarray
            Streamfunction on 2D grid (ny, nx)
        x_unique : np.ndarray
            Unique x coordinates
        y_unique : np.ndarray
            Unique y coordinates
        """
        from scipy.sparse import diags
        from scipy.sparse.linalg import spsolve

        # Get vorticity
        omega = self._compute_vorticity()

        # Get grid info
        shape = getattr(self, "shape_full", (self.params.nx, self.params.ny))
        ny, nx = shape
        dx, dy = self.dx_min, self.dy_min

        # Reshape omega to 2D
        omega_2d = omega.reshape(shape)

        # Build Laplacian matrix for interior points with Dirichlet BC (ψ=0 on boundaries)
        # Interior grid is (ny-2) x (nx-2)
        n_interior = (ny - 2) * (nx - 2)

        # Coefficients for 5-point stencil
        cx = 1.0 / (dx * dx)
        cy = 1.0 / (dy * dy)
        cc = -2.0 * (cx + cy)

        # Build sparse matrix
        diag_main = np.full(n_interior, cc)
        diag_x = np.full(n_interior - 1, cx)
        diag_y = np.full(n_interior - (nx - 2), cy)

        # Remove connections across row boundaries for x-direction
        for i in range(1, ny - 2):
            idx = i * (nx - 2) - 1
            if idx < len(diag_x):
                diag_x[idx] = 0.0

        A = diags(
            [diag_y, diag_x, diag_main, diag_x, diag_y],
            [-(nx - 2), -1, 0, 1, (nx - 2)],
            format="csr",
        )

        # RHS: -omega on interior (boundary ψ=0 contributions are zero)
        rhs = -omega_2d[1:-1, 1:-1].ravel()

        # Solve
        psi_interior = spsolve(A, rhs)

        # Reconstruct full ψ with zero boundaries
        psi_2d = np.zeros((ny, nx))
        psi_2d[1:-1, 1:-1] = psi_interior.reshape(ny - 2, nx - 2)

        # Get coordinates
        x_unique = np.sort(np.unique(self.fields.x))
        y_unique = np.sort(np.unique(self.fields.y))

        return psi_2d, x_unique, y_unique

    def _find_primary_vortex(self) -> dict:
        """Find the primary vortex (global minimum of streamfunction).

        Returns
        -------
        dict
            {'psi_min': float, 'x': float, 'y': float}
        """
        psi_2d, x_unique, y_unique = self._compute_streamfunction()

        # Find global minimum
        min_idx = np.unravel_index(np.argmin(psi_2d), psi_2d.shape)
        psi_min = psi_2d[min_idx]
        x_min = x_unique[min_idx[1]]
        y_min = y_unique[min_idx[0]]

        return {"psi_min": float(psi_min), "x": float(x_min), "y": float(y_min)}

    def _find_corner_vortices(self) -> dict:
        """Find secondary corner vortices (BR, BL, TL).

        Corner vortices have opposite sign to primary vortex:
        - Primary vortex has ψ < 0 (clockwise rotation)
        - Secondary vortices have ψ > 0 (counter-clockwise rotation)

        Search regions:
        - BR (bottom-right): x > 0.5, y < 0.5
        - BL (bottom-left): x < 0.5, y < 0.5
        - TL (top-left): x < 0.5, y > 0.5

        Returns
        -------
        dict
            {'BR': {'psi': float, 'x': float, 'y': float}, 'BL': {...}, 'TL': {...}}
        """
        psi_2d, x_unique, y_unique = self._compute_streamfunction()

        # Create 2D coordinate arrays
        X, Y = np.meshgrid(x_unique, y_unique)

        results = {}

        # Define search regions (corners)
        regions = {
            "BR": (X > 0.5) & (Y < 0.5),  # Bottom-right
            "BL": (X < 0.5) & (Y < 0.5),  # Bottom-left
            "TL": (X < 0.5) & (Y > 0.5),  # Top-left
        }

        for name, mask in regions.items():
            # Secondary vortices have ψ > 0 (opposite sign to primary)
            psi_region = np.where(mask, psi_2d, -np.inf)
            max_idx = np.unravel_index(np.argmax(psi_region), psi_2d.shape)
            psi_val = psi_2d[max_idx]

            # Only report if we found a positive ψ (secondary vortex exists)
            if psi_val > 0:
                results[name] = {
                    "psi": float(psi_val),
                    "x": float(x_unique[max_idx[1]]),
                    "y": float(y_unique[max_idx[0]]),
                }
            else:
                results[name] = {"psi": 0.0, "x": 0.0, "y": 0.0}

        return results

    def _find_max_vorticity(self) -> dict:
        """Find maximum vorticity and its location.

        Returns
        -------
        dict
            {'omega_max': float, 'x': float, 'y': float}
        """
        omega = self._compute_vorticity()

        # Get grid shape
        shape = getattr(self, "shape_full", (self.params.nx, self.params.ny))
        omega_2d = omega.reshape(shape)

        # Find maximum (by absolute value, but track actual sign)
        max_abs_idx = np.unravel_index(np.argmax(np.abs(omega_2d)), omega_2d.shape)
        omega_max = omega_2d[max_abs_idx]

        # Get coordinates
        x_unique = np.sort(np.unique(self.fields.x))
        y_unique = np.sort(np.unique(self.fields.y))

        return {
            "omega_max": float(omega_max),
            "x": float(x_unique[max_abs_idx[1]]),
            "y": float(y_unique[max_abs_idx[0]]),
        }

    def compute_vortex_metrics(self) -> dict:
        """Compute all vortex-related metrics for validation.

        Returns
        -------
        dict
            Dictionary with primary vortex, corner vortices, and max vorticity
        """
        primary = self._find_primary_vortex()
        corners = self._find_corner_vortices()
        max_omega = self._find_max_vorticity()

        return {
            "psi_min": primary["psi_min"],
            "psi_min_x": primary["x"],
            "psi_min_y": primary["y"],
            "omega_max": max_omega["omega_max"],
            "omega_max_x": max_omega["x"],
            "omega_max_y": max_omega["y"],
            "psi_BR": corners["BR"]["psi"],
            "psi_BR_x": corners["BR"]["x"],
            "psi_BR_y": corners["BR"]["y"],
            "psi_BL": corners["BL"]["psi"],
            "psi_BL_x": corners["BL"]["x"],
            "psi_BL_y": corners["BL"]["y"],
            "psi_TL": corners["TL"]["psi"],
            "psi_TL_x": corners["TL"]["x"],
            "psi_TL_y": corners["TL"]["y"],
        }

    def save_vtk(self, filepath: Path):
        """Save solution to VTS file.

        Parameters
        ----------
        filepath : Path
            Output file path (should have .vts extension)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        grid = self.to_vtk()
        grid.save(str(filepath))

        log.info(f"Saved VTS to {filepath}")

    # ========================================================================
    # MLflow Integration
    # ========================================================================

    def mlflow_start(
        self, experiment_name: str, run_name: str, parent_run_name: str = None
    ):
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
                max_results=1,
            )

            if runs:
                # Resume existing parent
                parent_run_id = runs[0].info.run_id
            else:
                # Create new parent run
                parent_run = client.create_run(
                    experiment_id=experiment.experiment_id,
                    run_name=parent_run_name,
                    tags={"is_parent": "true"},
                )
                parent_run_id = parent_run.info.run_id

            # Start nested child run
            mlflow.start_run(run_id=parent_run_id, log_system_metrics=False)
            mlflow.start_run(run_name=run_name, nested=True, log_system_metrics=True)
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
        if getattr(self, "_mlflow_nested", False):
            mlflow.end_run()

    def mlflow_log_artifact(self, filepath: str):
        """Log an artifact (e.g., saved HDF5 file) to MLflow.

        Parameters
        ----------
        filepath : str
            Path to the file to log as artifact.
        """
        mlflow.log_artifact(filepath)

    def mlflow_log_validation_table(self, reference_csv: str = None):
        """Log validation metrics comparison table to MLflow.

        Creates a table comparing computed vortex metrics against Botella & Peyret
        reference values, including energy/enstrophy/palinstrophy.

        Parameters
        ----------
        reference_csv : str, optional
            Path to reference CSV file. If None, uses default for current Re.
        """
        import pandas as pd

        if not mlflow.active_run():
            log.warning("No active MLflow run - skipping validation table")
            return

        # Load reference data
        if reference_csv is None:
            Re = int(self.params.Re)
            reference_csv = f"data/validation/botella/botella_Re{Re}_vortex.csv"

        ref_path = Path(reference_csv)
        if not ref_path.exists():
            log.warning(f"Reference file not found: {ref_path}")
            return

        ref_df = pd.read_csv(ref_path, comment="#")
        ref = ref_df.iloc[0].to_dict()

        # Build validation table rows
        rows = []

        def add_row(metric_name, computed, reference, unit="", use_abs=False):
            """Add a row to the validation table."""
            if reference != 0 and reference is not None:
                # For vorticity, compare absolute values (sign convention may differ)
                if use_abs:
                    error_pct = abs(abs(computed) - abs(reference)) / abs(reference) * 100
                else:
                    error_pct = abs(computed - reference) / abs(reference) * 100
            else:
                error_pct = 0.0
            rows.append({
                "Metric": metric_name,
                "Computed": f"{computed:.6f}",
                "Reference": f"{reference:.6f}" if reference else "-",
                "Error (%)": f"{error_pct:.2f}" if reference else "-",
                "Unit": unit,
            })

        # Primary vortex metrics
        add_row("ψ_min (Primary)", self.metrics.psi_min, ref.get("psi_min"))
        add_row("x (Primary)", self.metrics.psi_min_x, ref.get("psi_min_x"))
        add_row("y (Primary)", self.metrics.psi_min_y, ref.get("psi_min_y"))
        add_row("ω_center (Primary)", self.metrics.omega_center, ref.get("omega_center"), use_abs=True)

        # Secondary vortex - Bottom Right
        add_row("ψ_BR", self.metrics.psi_BR, ref.get("psi_BR"))
        add_row("x_BR", self.metrics.psi_BR_x, ref.get("psi_BR_x"))
        add_row("y_BR", self.metrics.psi_BR_y, ref.get("psi_BR_y"))

        # Secondary vortex - Bottom Left
        add_row("ψ_BL", self.metrics.psi_BL, ref.get("psi_BL"))
        add_row("x_BL", self.metrics.psi_BL_x, ref.get("psi_BL_x"))
        add_row("y_BL", self.metrics.psi_BL_y, ref.get("psi_BL_y"))

        # Global quantities (no reference for these typically)
        rows.append({
            "Metric": "--- Global Quantities ---",
            "Computed": "",
            "Reference": "",
            "Error (%)": "",
            "Unit": "",
        })
        rows.append({
            "Metric": "Energy (E)",
            "Computed": f"{self.metrics.final_energy:.6e}",
            "Reference": "-",
            "Error (%)": "-",
            "Unit": "",
        })
        rows.append({
            "Metric": "Enstrophy (Z)",
            "Computed": f"{self.metrics.final_enstrophy:.6e}",
            "Reference": "-",
            "Error (%)": "-",
            "Unit": "",
        })
        rows.append({
            "Metric": "Palinstrophy (P)",
            "Computed": f"{self.metrics.final_palinstrophy:.6e}",
            "Reference": "-",
            "Error (%)": "-",
            "Unit": "",
        })

        # Create DataFrame and log to MLflow
        table_df = pd.DataFrame(rows)
        mlflow.log_table(table_df, artifact_file="validation_metrics.json")
        log.info("Logged validation metrics table to MLflow")

    # =========================================================================
    # Validation against reference FV solutions
    # =========================================================================

    def compute_validation_errors(
        self, reference_dir: str = "data/validation/fv", save_plots: bool = True
    ) -> dict:
        """Compute L2 errors against reference FV solution.

        Loads the reference solution for the current Re, evaluates computed
        solution at reference grid coordinates, and computes relative L2 errors.

        Parameters
        ----------
        reference_dir : str
            Directory containing reference solutions (Re100/, Re400/, Re1000/)
        save_plots : bool
            If True, save error distribution plots as MLflow artifacts

        Returns
        -------
        dict
            {'u_L2_error': float, 'v_L2_error': float} or empty dict if no reference
        """
        ref_path = Path(reference_dir) / f"Re{int(self.params.Re)}" / "solution.vts"

        if not ref_path.exists():
            log.warning(f"No reference solution found at {ref_path}")
            return {}

        # Load reference solution
        ref_mesh = pv.read(str(ref_path))
        ref_u = ref_mesh.point_data["u"]
        ref_v = ref_mesh.point_data["v"]

        # Get reference grid coordinates
        ref_points = ref_mesh.points
        ref_x = ref_points[:, 0]
        ref_y = ref_points[:, 1]

        # Evaluate computed solution at reference grid points
        # Subclasses can override _evaluate_at_points for native interpolation (e.g., spectral)
        curr_u_at_ref, curr_v_at_ref = self._evaluate_at_points(ref_x, ref_y)

        # Only compute norm on interior points (exclude boundaries where BCs differ)
        # Use small margin to exclude boundary points
        margin = 0.02
        interior = (
            (ref_x > margin) & (ref_x < self.params.Lx - margin) &
            (ref_y > margin) & (ref_y < self.params.Ly - margin)
        )
        valid = interior & ~(np.isnan(curr_u_at_ref) | np.isnan(curr_v_at_ref))
        n_valid = np.sum(valid)
        n_total = len(ref_u)

        if n_valid < n_total * 0.5:
            log.warning(f"Only {n_valid}/{n_total} valid points for validation")

        # Compute relative L2 errors on valid interior points
        u_error = np.linalg.norm(curr_u_at_ref[valid] - ref_u[valid]) / (
            np.linalg.norm(ref_u[valid]) + 1e-12
        )
        v_error = np.linalg.norm(curr_v_at_ref[valid] - ref_v[valid]) / (
            np.linalg.norm(ref_v[valid]) + 1e-12
        )

        log.info(f"Validation errors vs FV reference ({n_valid}/{n_total} pts): u_L2={u_error:.6e}, v_L2={v_error:.6e}")

        # Save error distribution plots
        if save_plots:
            self._save_validation_error_plots(
                ref_x, ref_y, ref_u, ref_v, curr_u_at_ref, curr_v_at_ref, valid
            )

        return {"u_L2_error": float(u_error), "v_L2_error": float(v_error)}

    def _save_validation_error_plots(
        self, ref_x, ref_y, ref_u, ref_v, curr_u, curr_v, valid_mask
    ):
        """Save error distribution plots as artifacts."""
        import matplotlib.pyplot as plt

        # Compute error fields
        u_diff = curr_u - ref_u
        v_diff = curr_v - ref_v

        # Get grid dimensions from reference mesh
        n_unique_x = len(np.unique(ref_x))
        n_unique_y = len(np.unique(ref_y))

        # Reshape for plotting (assumes structured grid)
        try:
            X = ref_x.reshape(n_unique_x, n_unique_y)
            Y = ref_y.reshape(n_unique_x, n_unique_y)
            U_diff = u_diff.reshape(n_unique_x, n_unique_y)
            V_diff = v_diff.reshape(n_unique_x, n_unique_y)
        except ValueError:
            log.warning("Could not reshape error field for plotting - skipping plots")
            return

        # Determine output directory
        output_dir = Path("outputs/validation_errors")
        output_dir.mkdir(parents=True, exist_ok=True)
        method_name = getattr(self.params, 'method', 'solver')
        Re = int(self.params.Re)

        # Create figure for u error
        fig, ax = plt.subplots(figsize=(8, 6))
        vmax = max(abs(U_diff).max(), 1e-10)
        im = ax.pcolormesh(X, Y, U_diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'u error (computed - reference), Re={Re}')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='u error')
        plt.tight_layout()

        u_path = output_dir / f"{method_name}_Re{Re}_u_error.png"
        fig.savefig(u_path, dpi=150)
        if mlflow.active_run():
            mlflow.log_artifact(str(u_path))
        log.info(f"Saved u error plot to {u_path}")
        plt.close(fig)

        # Create figure for v error
        fig, ax = plt.subplots(figsize=(8, 6))
        vmax = max(abs(V_diff).max(), 1e-10)
        im = ax.pcolormesh(X, Y, V_diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'v error (computed - reference), Re={Re}')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='v error')
        plt.tight_layout()

        v_path = output_dir / f"{method_name}_Re{Re}_v_error.png"
        fig.savefig(v_path, dpi=150)
        if mlflow.active_run():
            mlflow.log_artifact(str(v_path))
        log.info(f"Saved v error plot to {v_path}")
        plt.close(fig)

    def _evaluate_at_points(self, x: np.ndarray, y: np.ndarray) -> tuple:
        """Evaluate solution at arbitrary points.

        Default uses bilinear interpolation from stored fields.
        Spectral solver overrides with native spectral interpolation.

        Parameters
        ----------
        x, y : np.ndarray
            Coordinates to evaluate at

        Returns
        -------
        u, v : np.ndarray
            Velocity components at requested points
        """
        from scipy.interpolate import RegularGridInterpolator

        # Get unique sorted coordinates from stored fields
        x_unique = np.sort(np.unique(self.fields.x))
        y_unique = np.sort(np.unique(self.fields.y))
        nx, ny = len(x_unique), len(y_unique)

        # Reshape fields to 2D
        indices = np.lexsort((self.fields.x, self.fields.y))
        u_2d = self.fields.u[indices].reshape(ny, nx)
        v_2d = self.fields.v[indices].reshape(ny, nx)

        # Create interpolators (NaN for out-of-bounds, filtered later)
        u_interp = RegularGridInterpolator(
            (y_unique, x_unique), u_2d, method="linear", bounds_error=False, fill_value=np.nan
        )
        v_interp = RegularGridInterpolator(
            (y_unique, x_unique), v_2d, method="linear", bounds_error=False, fill_value=np.nan
        )

        # Evaluate at requested points
        points = np.column_stack([y, x])  # (y, x) order for interpolator
        return u_interp(points), v_interp(points)
