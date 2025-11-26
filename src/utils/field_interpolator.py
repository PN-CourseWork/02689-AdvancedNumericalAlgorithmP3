"""Unified field interpolator for FV and Spectral solvers.

This module provides a consistent interface for interpolating solution fields
from different grid types (collocated FV, Gauss-Lobatto spectral) to a unified
representation suitable for validation and visualization.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import RectBivariateSpline


class UnifiedFieldInterpolator:
    """Unified interpolator for FV and Spectral solution fields.

    Automatically detects grid type and creates high-quality interpolators
    that handle boundary conditions appropriately for each grid type.

    Parameters
    ----------
    h5_path : str or Path
        Path to HDF5 solution file

    Attributes
    ----------
    fields : pd.DataFrame
        Raw field data from HDF5 file
    grid_type : str
        Detected grid type: 'collocated_fv' or 'lobatto_spectral'
    interpolators : dict
        Dictionary of RectBivariateSpline interpolators for each field
    """

    def __init__(self, h5_path):
        """Initialize interpolator from HDF5 file."""
        self.h5_path = Path(h5_path)

        # Load fields
        self.fields = pd.read_hdf(self.h5_path, 'fields')
        self.metadata = pd.read_hdf(self.h5_path, 'metadata')

        # Detect grid type and create interpolators
        self._detect_grid_type()
        self._create_interpolators()

    def _detect_grid_type(self):
        """Detect whether this is FV collocated or spectral Lobatto grid."""
        x = self.fields['x'].values
        y = self.fields['y'].values

        x_unique = np.unique(x)
        y_unique = np.unique(y)

        # Check if grid points are at boundaries (0 and 1)
        has_boundary_x = (np.abs(x_unique.min()) < 1e-10 and np.abs(x_unique.max() - 1.0) < 1e-10)
        has_boundary_y = (np.abs(y_unique.min()) < 1e-10 and np.abs(y_unique.max() - 1.0) < 1e-10)

        if has_boundary_x and has_boundary_y:
            # Gauss-Lobatto points include boundaries
            self.grid_type = 'lobatto_spectral'
        else:
            # Collocated FV: cell centers don't touch boundaries
            self.grid_type = 'collocated_fv'

    def _create_interpolators(self):
        """Create bicubic spline interpolators for all fields."""
        x = self.fields['x'].values
        y = self.fields['y'].values

        # Get unique sorted coordinates
        x_unique = np.sort(np.unique(x))
        y_unique = np.sort(np.unique(y))

        # Store grid info
        self.x_grid = x_unique
        self.y_grid = y_unique

        # Reshape fields to 2D grids
        sort_indices = np.lexsort((x, y))

        self.interpolators = {}
        for field in ['u', 'v', 'p']:
            if field in self.fields.columns:
                field_values = self.fields[field].values[sort_indices]
                field_grid = field_values.reshape((len(y_unique), len(x_unique)))

                # Create bicubic interpolator
                self.interpolators[field] = RectBivariateSpline(
                    y_unique, x_unique, field_grid, kx=3, ky=3
                )

    def evaluate_at_points(self, x, y, field='u'):
        """Evaluate field at arbitrary points using interpolation.

        Parameters
        ----------
        x : array_like
            X coordinates
        y : array_like
            Y coordinates
        field : str
            Field name ('u', 'v', or 'p')

        Returns
        -------
        values : np.ndarray
            Interpolated field values at (x, y) points
        """
        if field not in self.interpolators:
            raise ValueError(f"Field '{field}' not available. Available: {list(self.interpolators.keys())}")

        return self.interpolators[field](y, x, grid=False)

    def get_uniform_grid(self, nx=200, ny=200, include_boundaries=True):
        """Get fields interpolated onto a uniform grid.

        Parameters
        ----------
        nx, ny : int
            Number of points in x and y directions
        include_boundaries : bool
            If True, grid extends to boundaries (0, 1). If False, stays
            within the original data extent (useful for FV collocated grids).

        Returns
        -------
        dict
            Dictionary with keys:
            - 'x': 2D array of x coordinates
            - 'y': 2D array of y coordinates
            - 'u', 'v', 'p': 2D arrays of interpolated field values
        """
        if include_boundaries or self.grid_type == 'lobatto_spectral':
            # Include full domain boundaries
            x_uniform = np.linspace(0, 1, nx)
            y_uniform = np.linspace(0, 1, ny)
        else:
            # Stay within data extent (for FV collocated)
            x_uniform = np.linspace(self.x_grid.min(), self.x_grid.max(), nx)
            y_uniform = np.linspace(self.y_grid.min(), self.y_grid.max(), ny)

        # Create meshgrid
        X, Y = np.meshgrid(x_uniform, y_uniform, indexing='xy')

        # Interpolate all fields
        result = {'x': X, 'y': Y}
        for field_name, interp in self.interpolators.items():
            result[field_name] = interp(y_uniform, x_uniform, grid=True)

        return result

    def extract_centerline(self, field='u', axis='y', n_points=200):
        """Extract field values along a centerline.

        Parameters
        ----------
        field : str
            Field name ('u', 'v', or 'p')
        axis : str
            Axis along which to extract:
            - 'y': vertical centerline at x=0.5 (returns position, field)
            - 'x': horizontal centerline at y=0.5 (returns position, field)
        n_points : int
            Number of points along centerline

        Returns
        -------
        position : np.ndarray
            Position coordinates along centerline
        values : np.ndarray
            Field values along centerline
        """
        if field not in self.interpolators:
            raise ValueError(f"Field '{field}' not available")

        interp = self.interpolators[field]

        # Always extract from 0 to 1 for centerlines
        position = np.linspace(0, 1, n_points)

        if axis == 'y':
            # Vertical centerline at x=0.5
            values = interp(position, 0.5, grid=False)
        elif axis == 'x':
            # Horizontal centerline at y=0.5
            values = interp(0.5, position, grid=False)
        else:
            raise ValueError(f"axis must be 'x' or 'y', got '{axis}'")

        return position, values

    def get_info(self):
        """Get information about the grid and interpolator.

        Returns
        -------
        dict
            Dictionary with grid information
        """
        return {
            'grid_type': self.grid_type,
            'nx': len(self.x_grid),
            'ny': len(self.y_grid),
            'x_range': (self.x_grid.min(), self.x_grid.max()),
            'y_range': (self.y_grid.min(), self.y_grid.max()),
            'available_fields': list(self.interpolators.keys()),
            'file': self.h5_path.name
        }

    def __repr__(self):
        """String representation."""
        info = self.get_info()
        return (f"UnifiedFieldInterpolator(\n"
                f"  file='{info['file']}',\n"
                f"  grid_type='{info['grid_type']}',\n"
                f"  resolution={info['nx']}×{info['ny']},\n"
                f"  fields={info['available_fields']}\n"
                f")")
