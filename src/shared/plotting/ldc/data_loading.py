"""
Data Loading and Transformation for LDC Plotting.

Handles loading solution from VTS files and converting to formats
needed by plotting functions.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv


def load_solution(artifact_dir: Path) -> pv.StructuredGrid:
    """Load solution as PyVista StructuredGrid.

    The VTS file contains all fields (u, v, p, vorticity, velocity_magnitude)
    and metadata (Re, N, solver name).

    Parameters
    ----------
    artifact_dir : Path
        Directory containing solution.vts

    Returns
    -------
    pv.StructuredGrid
        Solution grid with all fields
    """
    vtk_path = artifact_dir / "solution.vts"

    if not vtk_path.exists():
        # Fallback to legacy location
        vtk_path = artifact_dir / "fields" / "solution.vts"

    if not vtk_path.exists():
        raise FileNotFoundError(f"VTK file not found in {artifact_dir}")

    return pv.read(vtk_path)


def load_fields_from_zarr(artifact_dir: Path) -> dict:
    """Load solution fields from VTK artifact.

    Note: Legacy function name kept for backwards compatibility.
    Use load_solution() for new code.
    """
    grid = load_solution(artifact_dir)

    # Extract coordinates from structured grid
    points = grid.points
    x = points[:, 0]
    y = points[:, 1]

    # Extract fields
    fields = {
        "x": x,
        "y": y,
        "u": grid["u"],
        "v": grid["v"],
        "p": grid["pressure"],
    }

    # Include derived fields if available
    if "vorticity" in grid.array_names:
        fields["vorticity"] = grid["vorticity"]
    if "velocity_magnitude" in grid.array_names:
        fields["velocity_magnitude"] = grid["velocity_magnitude"]

    return fields


def restructure_fields(
    fields: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert potentially flattened fields to structured 2D arrays.

    Returns (x_unique, y_unique, U_2d, V_2d, P_2d) where U, V, P are 2D arrays
    with shape (ny, nx) suitable for RectBivariateSpline.
    """
    x, y = fields["x"], fields["y"]
    u, v, p = fields["u"], fields["v"], fields["p"]

    # Get unique coordinates
    x_unique = np.sort(np.unique(x))
    y_unique = np.sort(np.unique(y))
    nx, ny = len(x_unique), len(y_unique)

    # If already 2D, just return
    if u.ndim == 2:
        return x_unique, y_unique, u, v, p

    # Reshape from flattened to 2D
    # Create mapping from coordinates to indices
    x_to_idx = {val: i for i, val in enumerate(x_unique)}
    y_to_idx = {val: i for i, val in enumerate(y_unique)}

    U_2d = np.zeros((ny, nx))
    V_2d = np.zeros((ny, nx))
    P_2d = np.zeros((ny, nx))

    for k in range(len(x)):
        i = x_to_idx[x[k]]
        j = y_to_idx[y[k]]
        U_2d[j, i] = u[k]
        V_2d[j, i] = v[k]
        P_2d[j, i] = p[k]

    return x_unique, y_unique, U_2d, V_2d, P_2d


def fields_to_dataframe(fields: dict) -> pd.DataFrame:
    """Convert fields dict to DataFrame format for plotting.

    Handles two storage formats:
    1. Flattened: x, y, u, v, p all 1D with same length (already meshgrid coords)
    2. Structured: x, y are 1D coordinate arrays, u, v, p are 2D fields
    """
    x, y = fields["x"], fields["y"]
    u, v, p = fields["u"], fields["v"], fields["p"]

    # Check if already flattened (all same length, 1D)
    if x.ndim == 1 and len(x) == len(u.ravel()):
        # Already flattened meshgrid coordinates
        return pd.DataFrame(
            {
                "x": x,
                "y": y,
                "u": u.ravel(),
                "v": v.ravel(),
                "p": p.ravel(),
            }
        )

    # Need to create meshgrid
    if x.ndim == 1 and y.ndim == 1:
        X, Y = np.meshgrid(x, y)
    else:
        X, Y = x, y

    return pd.DataFrame(
        {
            "x": X.ravel(),
            "y": Y.ravel(),
            "u": u.ravel(),
            "v": v.ravel(),
            "p": p.ravel(),
        }
    )
