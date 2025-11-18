"""Abstract base solver for lid-driven cavity problem."""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

from .datastructures import Fields, Meta, TimeSeries, Mesh


class LidDrivenCavitySolver(ABC):
    """Abstract base solver for lid-driven cavity problem.

    This base class handles:
    - Configuration management
    - Physics parameters (Re, viscosity, density)

    Subclasses must implement:
    - step() : Perform one iteration
    - _create_output_dataclasses() : Create result dataclasses

    Parameters
    ----------
    config : Config (or subclass like FVConfig, SpectralConfig)
        Configuration with physics (Re, Lx, Ly, lid_velocity) and numerics (nx, ny, etc).
    """
    def __init__(self, **kwargs):
        """Initialize solver with configuration.

        Parameters
        ----------
        **kwargs
            Configuration parameters passed to the Config class.
        """

        self.config = self.Config(**kwargs)

        self.mesh = Mesh(self.config)        

        self.fields = Fields(self.config)

        self.time_series = TimeSeries()

    @abstractmethod
    def step(self):
        """Perform one iteration/time step of the solver.

        This method should:
        1. Update the solution fields in self.fields (u, v, p)
        2. Return the updated fields as a tuple (u, v, p)

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
            self.config.tolerance = tolerance
        if max_iter is None:
            self.config.max_iterations = max_iter

        # Store previous iteration for residual calculation
        u_prev = self.fields.u.copy()
        v_prev = self.fields.v.copy()

        # Residual history
        residual_history = []

        time_start = time.time()

        for i in range(self.config.max_iterations):
            self.config.iterations = i + 1

            # Perform one iteration
            self.fields.u, self.fields.v, self.fields.p = self.step()

            # Calculate normalized solution change: ||u^{n+1} - u^n||_2 / ||u^n||_2
            u_change_norm = np.linalg.norm(self.fields.u - u_prev)
            v_change_norm = np.linalg.norm(self.fields.v - v_prev)

            u_prev_norm = np.linalg.norm(u_prev) + 1e-12
            v_prev_norm = np.linalg.norm(v_prev) + 1e-12

            u_residual = u_change_norm / u_prev_norm
            v_residual = v_change_norm / v_prev_norm

            # Only store residual history after first 10 iterations
            if i >= 10:
                residual_history.append({'u': u_residual, 'v': v_residual})
                self.time_series.

            # Update previous iteration
            u_prev = self.fields.u.copy()
            v_prev = self.fields.v.copy()

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

        # Create output dataclasses
        self.fields, self.time_series, self.metadata = self._create_output_dataclasses(
            residual_history, final_iter_count, is_converged
        )


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

        with h5py.File(filepath, 'w') as f:
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
            fields_grp = f.create_group('fields')
            for key, val in fields_dict.items():
                fields_grp.create_dataset(key, data=val)

            # Add velocity magnitude if u and v are present
            if 'u' in fields_dict and 'v' in fields_dict:
                import numpy as np
                vel_mag = np.sqrt(fields_dict['u']**2 + fields_dict['v']**2)
                fields_grp.create_dataset('velocity_magnitude', data=vel_mag)

            # Save grid_points at root level for compatibility
            if 'grid_points' in fields_dict:
                f.create_dataset('grid_points', data=fields_dict['grid_points'])

            # Save time series in a group
            if time_series_dict:
                ts_grp = f.create_group('time_series')
                for key, val in time_series_dict.items():
                    if val is not None:
                        ts_grp.create_dataset(key, data=val)
