"""Data I/O utilities for HDF5 and pandas integration.

This module provides helpers for loading solver results from HDF5 files
into pandas DataFrames for analysis and plotting.
"""

from pathlib import Path
from typing import Dict, Any, List, Union
import numpy as np
import pandas as pd
import h5py


def load_run_data(path: Union[str, Path]) -> pd.DataFrame:
    """Load HDF5 run data as DataFrame for plotting.

    Loads time-series data (residuals) along with metadata as columns.
    This format is optimized for seaborn plotting with `hue` parameter.

    Parameters
    ----------
    path : str or Path
        Path to HDF5 file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - iteration: Iteration number (0, 1, 2, ...)
        - residual: Residual value at each iteration
        - u_residual, v_residual, continuity_residual: Component residuals (if available)
        - Re: Reynolds number (from metadata)
        - converged: Whether solver converged (from metadata)
        - All other metadata fields as additional columns

    Examples
    --------
    >>> df = load_run_data('run.h5')
    >>> df.head()
       iteration  residual  Re  converged  mesh_path  ...
    0          0  1.000000  100       True  fine.msh  ...
    1          1  0.500000  100       True  fine.msh  ...

    >>> # Plot multiple runs
    >>> import seaborn as sns
    >>> df1 = load_run_data('run1.h5').assign(run='Run 1')
    >>> df2 = load_run_data('run2.h5').assign(run='Run 2')
    >>> df = pd.concat([df1, df2])
    >>> sns.lineplot(data=df, x='iteration', y='residual', hue='run')
    """
    path = Path(path)

    with h5py.File(path, "r") as f:
        # Load time-series data
        residual = f["time_series/iter_residual"][:]
        n_iter = len(residual)

        # Start building DataFrame with time-series
        data = {
            "iteration": np.arange(n_iter),
            "residual": residual,
        }

        # Add other time-series if available
        ts_group = f["time_series"]
        for key in ts_group.keys():
            if key != "iter_residual":  # Already added
                data[key] = ts_group[key][:]

        # Load metadata and broadcast to all rows
        metadata = dict(f.attrs)

        # Create DataFrame
        df = pd.DataFrame(data)

        # Add metadata as columns (broadcast to all rows)
        for key, value in metadata.items():
            df[key] = value

    return df


def load_fields(path: Union[str, Path]) -> pd.DataFrame:
    """Load spatial fields as DataFrame.

    Parameters
    ----------
    path : str or Path
        Path to HDF5 file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - x, y: Spatial coordinates
        - u, v, p: Velocity and pressure fields
        - velocity_magnitude: Magnitude of velocity

    Examples
    --------
    >>> df = load_fields('run.h5')
    >>> df.head()
              x         y         u         v         p  velocity_magnitude
    0  0.000000  0.000000  0.000000  0.000000  0.500000            0.000000
    1  0.031250  0.000000  0.000000  0.000000  0.490000            0.000000
    """
    path = Path(path)

    with h5py.File(path, "r") as f:
        # Load grid points
        grid_points = f["grid_points"][:]

        # Load fields
        u = f["fields/u"][:]
        v = f["fields/v"][:]
        p = f["fields/p"][:]
        vel_mag = f["fields/velocity_magnitude"][:]

        # Create DataFrame
        df = pd.DataFrame(
            {
                "x": grid_points[:, 0],
                "y": grid_points[:, 1],
                "u": u,
                "v": v,
                "p": p,
                "velocity_magnitude": vel_mag,
            }
        )

    return df


def load_metadata(path: Union[str, Path]) -> Dict[str, Any]:
    """Load only metadata from HDF5 file.

    Parameters
    ----------
    path : str or Path
        Path to HDF5 file.

    Returns
    -------
    dict
        Metadata dictionary containing solver config and convergence info.

    Examples
    --------
    >>> metadata = load_metadata('run.h5')
    >>> print(f"Re={metadata['Re']}, converged={metadata['converged']}")
    Re=100.0, converged=True
    """
    path = Path(path)

    with h5py.File(path, "r") as f:
        metadata = dict(f.attrs)

    return metadata


def load_multiple_runs(
    paths: List[Union[str, Path]], labels: List[str] = None
) -> pd.DataFrame:
    """Load multiple runs into single DataFrame for comparison.

    Parameters
    ----------
    paths : list of str or Path
        Paths to HDF5 files.
    labels : list of str, optional
        Labels for each run. If None, uses filenames.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with 'run' column for distinguishing runs.

    Examples
    --------
    >>> df = load_multiple_runs(
    ...     ['run1.h5', 'run2.h5'],
    ...     labels=['32x32', '64x64']
    ... )
    >>> sns.lineplot(data=df, x='iteration', y='residual', hue='run')
    """
    if labels is None:
        labels = [Path(p).stem for p in paths]

    dfs = []
    for path, label in zip(paths, labels):
        df = load_run_data(path)
        df["run"] = label
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)
