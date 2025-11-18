"""Abstract base solver for lid-driven cavity problem."""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

from .datastructures import Fields, Meta, TimeSeries, Mesh


class LidDrivenCavitySolver(ABC):
    """Abstract base solver for lid-driven cavity problem.

    This base class handles:
    - Configuration management
    - Mesh and field initialization
    - Common solve loop with residual tracking

    Subclasses must implement:
    - step() : Perform one iteration

    Subclasses must define:
    - Config : The config dataclass (Meta or subclass)
    - MeshType : (optional) The mesh dataclass (Mesh or subclass, defaults to Mesh)
    - FieldsType : (optional) The fields dataclass (Fields or subclass, defaults to Fields)
    """

    # Subclasses override these
    Config = None
    MeshType = Mesh
    FieldsType = Fields

    def __init__(self, **kwargs):
        """Initialize solver with configuration.

        Parameters
        ----------
        **kwargs
            Configuration parameters passed to the Config class.
        """
        # Create config
        if self.Config is None:
            raise ValueError("Subclass must define Config class attribute")

        self.config = self.Config(**kwargs)

        # Compute fluid properties from Reynolds number
        self.rho = 1.0  # Normalized density
        self.mu = self.rho * self.config.lid_velocity * self.config.Lx / self.config.Re

        # Create mesh
        self.mesh = self.MeshType(
            nx=self.config.nx,
            ny=self.config.ny,
            Lx=self.config.Lx,
            Ly=self.config.Ly
        )

        # Create fields
        self.fields = self.FieldsType(n_cells=self.mesh.n_cells)

        # Initialize time series
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
            tolerance = self.config.tolerance
        if max_iter is None:
            max_iter = self.config.max_iterations

        # Store previous iteration for residual calculation
        self.fields.u_prev[:] = self.fields.u
        self.fields.v_prev[:] = self.fields.v

        time_start = time.time()
        is_converged = False

        for i in range(max_iter):
            # Perform one iteration
            self.fields.u, self.fields.v, self.fields.p = self.step()

            # Calculate normalized solution change: ||u^{n+1} - u^n||_2 / ||u^n||_2
            u_change_norm = np.linalg.norm(self.fields.u - self.fields.u_prev)
            v_change_norm = np.linalg.norm(self.fields.v - self.fields.v_prev)

            u_prev_norm = np.linalg.norm(self.fields.u_prev) + 1e-12
            v_prev_norm = np.linalg.norm(self.fields.v_prev) + 1e-12

            u_residual = u_change_norm / u_prev_norm
            v_residual = v_change_norm / v_prev_norm

            # Only store residual history after first 10 iterations
            if i >= 10:
                self.time_series.u_residuals.append(u_residual)
                self.time_series.v_residuals.append(v_residual)

            # Update previous iteration
            self.fields.u_prev[:] = self.fields.u
            self.fields.v_prev[:] = self.fields.v

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

        # Update config with convergence info
        self.config.iterations = i + 1
        self.config.converged = is_converged
        if self.time_series.u_residuals:
            self.config.final_residual = max(
                self.time_series.u_residuals[-1],
                self.time_series.v_residuals[-1]
            )

        # Print summary
        print(f"\nSolution Status:")
        print(f"  Converged: {self.config.converged}")
        print(f"  Iterations: {self.config.iterations}")
        print(f"  Final residual: {self.config.final_residual:.6e}")


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

        # Convert dataclasses to dicts (manually extract fields to avoid mesh pickling issue)
        fields_dict = {
            'u': self.fields.u,
            'v': self.fields.v,
            'p': self.fields.p,
            'u_prev': self.fields.u_prev,
            'v_prev': self.fields.v_prev,
        }

        # Extract grid information (different for Mesh vs FV mesh)
        if hasattr(self.mesh, 'x') and hasattr(self.mesh, 'y'):
            # Simple Mesh class
            fields_dict['x'] = self.mesh.x
            fields_dict['y'] = self.mesh.y
            fields_dict['grid_points'] = self.mesh.grid_points
        elif hasattr(self.mesh, 'cell_centers'):
            # FV mesh (MeshData2D)
            fields_dict['grid_points'] = self.mesh.cell_centers
            # Extract unique x and y coordinates
            fields_dict['x'] = np.unique(self.mesh.cell_centers[:, 0])
            fields_dict['y'] = np.unique(self.mesh.cell_centers[:, 1])

        # Add FV-specific fields if they exist
        if hasattr(self.fields, 'mdot'):
            fields_dict['mdot'] = self.fields.mdot

        time_series_dict = asdict(self.time_series)
        config_dict = asdict(self.config)

        with h5py.File(filepath, 'w') as f:
            # Save config as root-level attributes
            for key, val in config_dict.items():
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
                # Save all arrays
                if isinstance(val, np.ndarray):
                    fields_grp.create_dataset(key, data=val)

            # Add velocity magnitude if u and v are present
            if 'u' in fields_dict and 'v' in fields_dict:
                vel_mag = np.sqrt(fields_dict['u']**2 + fields_dict['v']**2)
                fields_grp.create_dataset('velocity_magnitude', data=vel_mag)

            # Save grid_points at root level for compatibility
            if 'grid_points' in fields_dict:
                f.create_dataset('grid_points', data=fields_dict['grid_points'])

            # Save time series in a group
            if time_series_dict:
                ts_grp = f.create_group('time_series')
                for key, val in time_series_dict.items():
                    if val is not None and len(val) > 0:
                        ts_grp.create_dataset(key, data=val)
