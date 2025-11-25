"""Abstract base solver for lid-driven cavity problem."""
# --- Force project root onto sys.path so "ldc" is importable when run directly ---
import sys
from pathlib import Path

# Path to: <project-root>/src
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # go from src/ldc/base_solver.py to project root
SRC_PATH = PROJECT_ROOT / "src"

# Insert only if not present (avoid duplicates)
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
# -------------------------------------------------------------------------------

from abc import ABC, abstractmethod
import numpy as np
from dataclasses import replace

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

    def _store_results(self, residual_history, final_iter_count, is_converged, energy, enstrophy, palinstropy):
        """Store solve results in self.fields, self.time_series, and self.metadata."""
        # Extract residuals
        u_residuals = [r["u"] for r in residual_history]
        v_residuals = [r["v"] for r in residual_history]
        combined_residual = [max(r["u"], r["v"]) for r in residual_history]

        # Create time series (same for all solvers)
        self.time_series = TimeSeries(
            iter_residual=combined_residual,
            u_residual=u_residuals,
            v_residual=v_residuals,
            continuity_residual=[],
            energy=energy,
            enstrophy=enstrophy,
            palinstropy=palinstropy,
        )

        # Update metadata with convergence info
        self.metadata = replace(
            self.config,
            iterations=final_iter_count,
            converged=is_converged,
            final_residual=combined_residual[-1] if combined_residual else float("inf"),
        )

    def solve(self, tolerance, max_iter):
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
        u_prev = self.fields.u.copy()
        v_prev = self.fields.v.copy()

        # Residual history
        residual_history = []
        # Quantities
        energy = []
        palinstropy = []
        enstrophy = []

        time_start = time.time()
        final_iter_count = 0
        is_converged = False

        for i in range(max_iter):
            final_iter_count = i + 1

            # Perform one iteration
            self.step()

            # Calculate normalized solution change: ||u^{n+1} - u^n||_2 / ||u^n||_2
            u_change_norm = np.linalg.norm(self.fields.u - u_prev)
            v_change_norm = np.linalg.norm(self.fields.v - v_prev)

            u_prev_norm = np.linalg.norm(u_prev) + 1e-12
            v_prev_norm = np.linalg.norm(v_prev) + 1e-12

            u_residual = u_change_norm / u_prev_norm
            v_residual = v_change_norm / v_prev_norm

            # Only store residual history after first 10 iterations
            if i >= 10:
                residual_history.append({"u": u_residual, "v": v_residual})

            # Update previous iteration
            u_prev = self.fields.u.copy()
            v_prev = self.fields.v.copy()

            # ========================================================
            # Calculate quantities
            # ========================================================
            energy.append(self._calculate_energy())
            enstrophy.append(self._calculate_enstrophy())
            palinstropy.append(self._calculate_palinstropy())

            # Check convergence (only after warmup period)
            if i >= 10:
                is_converged = (u_residual < tolerance) and (v_residual < tolerance)
            else:
                is_converged = False

            if i % 10 == 0 or is_converged:
                print(f"Iteration {i}: u_res={u_residual:.6e}, v_res={v_residual:.6e}")

            if is_converged:
                print(f"Converged at iteration {i}")
                break

        time_end = time.time()
        print(f"Solver finished in {time_end - time_start:.2f} seconds.")

        # Store results
        self._store_results(residual_history, final_iter_count, is_converged, energy, enstrophy, palinstropy)

    def save(self, filepath):
        """Save results to HDF5 file.

        Parameters
        ----------
        filepath : str or Path
            Output file path.
        """
        from dataclasses import asdict
        import h5py
        from pathlib import Path

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Convert dataclasses to dicts
        fields_dict = asdict(self.fields)
        time_series_dict = asdict(self.time_series)
        metadata_dict = asdict(self.metadata)

        with h5py.File(filepath, "w") as f:
            # Save metadata as root-level attributes
            for key, val in metadata_dict.items():
                # Skip None values and convert to appropriate types
                if val is None:
                    continue
                # Convert strings to bytes for HDF5 compatibility
                if isinstance(val, str):
                    f.attrs[key] = val
                else:
                    f.attrs[key] = val

            # Save fields in a fields group
            fields_grp = f.create_group("fields")
            for key, val in fields_dict.items():
                fields_grp.create_dataset(key, data=val)

            # Add velocity magnitude if u and v are present
            if "u" in fields_dict and "v" in fields_dict:
                import numpy as _np

                vel_mag = _np.sqrt(fields_dict["u"] ** 2 + fields_dict["v"] ** 2)
                fields_grp.create_dataset("velocity_magnitude", data=vel_mag)

            # Save grid_points at root level for compatibility
            if "grid_points" in fields_dict:
                f.create_dataset("grid_points", data=fields_dict["grid_points"])

            # Save time series in a group
            if time_series_dict:
                ts_grp = f.create_group("time_series")
                for key, val in time_series_dict.items():
                    if val is not None:
                        ts_grp.create_dataset(key, data=val)

    # ---------------------------------------------------------------------
    # Diagnostics using ONLY self.fields  (no self.arrays needed)
    # ---------------------------------------------------------------------
    def _energy_from_stored(self) -> float:
        """
        Kinetic energy:
            E = 0.5 * ∫ (u^2 + v^2) dA
        Uses self.fields.u and self.fields.v (both flattened 1D arrays).
        """
        fields = self.fields
        u = fields.u
        v = fields.v

        # area per cell (uniform grid)
        dx = float(getattr(self, "dx_min", 1.0))
        dy = float(getattr(self, "dy_min", 1.0))
        dA = dx * dy

        return 0.5 * float(np.dot(u, u) + np.dot(v, v)) * dA

    def _enstrophy_from_stored(self) -> float:
        """
        Enstrophy:
            𝓔 = 0.5 * ∫ ω² dA,   where ω = dv/dx - du/dy

        Derivatives use Dx, Dy stored on the solver.
        """
        fields = self.fields
        Dx = getattr(self, "Dx", None)
        Dy = getattr(self, "Dy", None)

        u = fields.u
        v = fields.v

        # If Dx/Dy not available, we cannot compute derivatives; return 0.0
        if Dx is None or Dy is None:
            return 0.0

        dv_dx = Dx @ v
        du_dy = Dy @ u
        omega = dv_dx - du_dy

        dx = float(getattr(self, "dx_min", 1.0))
        dy = float(getattr(self, "dy_min", 1.0))
        dA = dx * dy

        return 0.5 * float(np.dot(omega, omega)) * dA

    def _palinstrophy_from_stored(self) -> float:
        """
        Palinstrophy:
            𝓟 = ∫ (ω_x² + ω_y²) dA

        Uses Dx, Dy to compute ω_x, ω_y.
        """
        fields = self.fields
        Dx = getattr(self, "Dx", None)
        Dy = getattr(self, "Dy", None)

        u = fields.u
        v = fields.v

        if Dx is None or Dy is None:
            return 0.0

        dv_dx = Dx @ v
        du_dy = Dy @ u
        omega = dv_dx - du_dy

        omega_x = Dx @ omega
        omega_y = Dy @ omega

        dx = float(getattr(self, "dx_min", 1.0))
        dy = float(getattr(self, "dy_min", 1.0))
        dA = dx * dy

        return float(np.dot(omega_x, omega_x) + np.dot(omega_y, omega_y)) * dA

    # ---------------------------------------------------------------------
    # Compatibility wrappers expected by solve()
    # ---------------------------------------------------------------------
    def _calculate_energy(self) -> float:
        """
        Compatibility wrapper used by solve(). Prefer solver-specific
        _energy_from_stored() if available.
        """
        if hasattr(self, "_energy_from_stored"):
            return float(self._energy_from_stored())
        # allow subclass override
        if hasattr(self, "_calculate_energy_override"):
            return float(self._calculate_energy_override())
        raise NotImplementedError(
            "No energy diagnostic found. Implement _energy_from_stored() or _calculate_energy_override()."
        )

    def _calculate_enstrophy(self) -> float:
        """
        Compatibility wrapper used by solve(). Prefer solver-specific
        _enstrophy_from_stored() if available.
        """
        if hasattr(self, "_enstrophy_from_stored"):
            return float(self._enstrophy_from_stored())
        if hasattr(self, "_calculate_enstrophy_override"):
            return float(self._calculate_enstrophy_override())
        raise NotImplementedError(
            "No enstrophy diagnostic found. Implement _enstrophy_from_stored() or _calculate_enstrophy_override()."
        )

    def _calculate_palinstropy(self) -> float:
        """
        Compatibility wrapper used by solve(). The name 'palinstropy' (no 'h')
        is the symbol used in solve()/TimeSeries; accept the stored implementation.
        """
        if hasattr(self, "_palinstrophy_from_stored"):
            return float(self._palinstrophy_from_stored())
        if hasattr(self, "_palinstropy_from_stored"):
            return float(self._palinstropy_from_stored())
        if hasattr(self, "_calculate_palinstropy_override"):
            return float(self._calculate_palinstropy_override())
        raise NotImplementedError(
            "No palinstropy diagnostic found. Implement _palinstrophy_from_stored() "
            "or _palinstropy_from_stored() or _calculate_palinstropy_override()."
        )