"""Field data structures for solver results."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Fields:
    """Base spatial solution fields.

    Parameters
    ----------
    u : np.ndarray
        x-velocity component field.
    v : np.ndarray
        y-velocity component field.
    p : np.ndarray
        Pressure field.
    x : np.ndarray
        x-coordinates of grid points.
    y : np.ndarray
        y-coordinates of grid points.
    grid_points : np.ndarray
        Flattened array of grid point coordinates, shape (N, 2).
    """
    u: np.ndarray
    v: np.ndarray
    p: np.ndarray
    x: np.ndarray
    y: np.ndarray
    grid_points: np.ndarray


@dataclass
class FVFields(Fields):
    """FV-specific fields with mass flux.

    Inherits all fields from Fields and adds mass flux.

    Parameters
    ----------
    mdot : np.ndarray, optional
        Mass flux field at cell faces. Default is None.
    """
    mdot: np.ndarray = None
