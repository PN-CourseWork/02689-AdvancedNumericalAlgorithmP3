"""Data structures for solver configuration and results.

This module defines the configuration and result data structures
for lid-driven cavity solvers (both FV and spectral).
"""

from dataclasses import dataclass, asdict
from typing import Optional, List

import numpy as np
import pandas as pd

# ========================================================
# Shared Data Classes
# =======================================================


@dataclass
class Fields:
    """Base spatial solution fields."""

    u: np.ndarray
    v: np.ndarray
    p: np.ndarray
    x: np.ndarray
    y: np.ndarray
    grid_points: np.ndarray
    # Previous iteration (for under-relaxation)
    u_prev: np.ndarray
    v_prev: np.ndarray

    def to_dataframe(self) -> pd.DataFrame:
        """Convert fields to DataFrame for analysis and plotting.

        Returns
        -------
        pd.DataFrame
            Wide-format DataFrame with columns: x, y, u, v, p.
            Each row represents one grid point.
            Excludes grid_points, u_prev, v_prev since they're not needed for analysis.
        """
        data = asdict(self)
        # Remove grid_points and previous iteration fields
        data.pop('grid_points', None)
        data.pop('u_prev', None)
        data.pop('v_prev', None)
        data.pop('mdot', None)  # May not exist in all subclasses
        return pd.DataFrame(data)


@dataclass
class TimeSeries:
    """Time series data common to all solvers.

    Attributes
    ----------
    rel_iter_residual : List[float]
        Relative iteration residual: max(||u^{n+1}-u^n||/||u^n||, ||v^{n+1}-v^n||/||v^n||)
    u_residual : List[float]
        U-velocity relative iteration residual: ||u^{n+1}-u^n||/||u^n||
    v_residual : List[float]
        V-velocity relative iteration residual: ||v^{n+1}-v^n||/||v^n||
    continuity_residual : Optional[List[float]]
        Continuity equation residual (solver-specific, may be None)
    """

    rel_iter_residual: List[float]
    u_residual: List[float]
    v_residual: List[float]
    continuity_residual: Optional[List[float]]
    # TODO: Add the quantities stuff from the paper

    def to_dataframe(self) -> pd.DataFrame:
        """Convert time series to DataFrame for analysis and plotting.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns for each residual type.
            Index represents iteration number.
        """
        # Filter out None values to avoid object dtype
        data = {k: v for k, v in asdict(self).items() if v is not None}
        return pd.DataFrame(data)


@dataclass
class MetaConfig:
    """Base solver metadata, config and convergence info."""

    # Physics parameters (required)
    Re: float = 100

    # Grid parameters (with defaults)
    nx: int = 16
    ny: int = 16

    # Physics parameters (with defaults)
    lid_velocity: float = 1
    Lx: float = 1
    Ly: float = 1

    # Solver config
    max_iterations: int = 500
    tolerance: float = 1e-4
    method: str = ""

    # Convergence info
    iterations: int = 0
    converged: bool = False
    final_residual: float = 10000

    def to_dataframe(self) -> pd.DataFrame:
        """Convert config/metadata to single-row DataFrame.

        Returns
        -------
        pd.DataFrame
            Single-row DataFrame with all configuration and metadata fields.
        """
        return pd.DataFrame([asdict(self)])


# =============================================================
# Finite Volume specific data classes
# ============================================================


@dataclass
class FVinfo(MetaConfig):
    """FV-specific metadata with discretization parameters."""
    convection_scheme: str = "Upwind"
    limiter: str = "MUSCL"
    alpha_uv: float = 0.6
    alpha_p: float = 0.4


@dataclass
class FVResultFields(Fields):
    """FV result fields returned to user after solving."""
    mdot: np.ndarray = None


@dataclass
class FVSolverFields:
    """Internal FV solver arrays - current state, previous iteration, and work buffers."""
    # Current solution state
    u: np.ndarray
    v: np.ndarray
    p: np.ndarray
    mdot: np.ndarray

    # Previous iteration (for under-relaxation)
    u_prev: np.ndarray
    v_prev: np.ndarray

    # Gradient buffers
    grad_p: np.ndarray
    grad_u: np.ndarray
    grad_v: np.ndarray
    grad_p_prime: np.ndarray

    # Face interpolation buffers
    grad_p_bar: np.ndarray
    bold_D: np.ndarray
    bold_D_bar: np.ndarray

    # Velocity and flux work buffers
    U_star_rc: np.ndarray
    U_prime_face: np.ndarray
    u_prime: np.ndarray
    v_prime: np.ndarray
    mdot_star: np.ndarray
    mdot_prime: np.ndarray

    @classmethod
    def allocate(cls, n_cells: int, n_faces: int):
        """Allocate all arrays with proper sizes."""
        return cls(
            # Current solution
            u=np.zeros(n_cells),
            v=np.zeros(n_cells),
            p=np.zeros(n_cells),
            mdot=np.zeros(n_faces),
            # Previous iteration
            u_prev=np.zeros(n_cells),
            v_prev=np.zeros(n_cells),
            # Gradient buffers
            grad_p=np.zeros((n_cells, 2)),
            grad_u=np.zeros((n_cells, 2)),
            grad_v=np.zeros((n_cells, 2)),
            grad_p_prime=np.zeros((n_cells, 2)),
            # Face interpolation buffers
            grad_p_bar=np.zeros((n_faces, 2)),
            bold_D=np.zeros((n_cells, 2)),
            bold_D_bar=np.zeros((n_faces, 2)),
            # Velocity and flux work buffers
            U_star_rc=np.zeros((n_faces, 2)),
            U_prime_face=np.zeros((n_faces, 2)),
            u_prime=np.zeros(n_cells),
            v_prime=np.zeros(n_cells),
            mdot_star=np.zeros(n_faces),
            mdot_prime=np.zeros(n_faces),
        )


#=====================================================
# Spectral Data Classes
# =====================================================


@dataclass
class SpectralInfo(MetaConfig):
    """Spectral-specific metadata with discretization parameters."""

    Nx: int = 64
    Ny: int = 64
    basis_type: str = "legendre"  # 'legendre' or 'chebyshev'
    differentiation_method: str = "fft"  # 'fft', 'chebyshev', 'matrix'
    time_scheme: str = "rk4"
    dt: float = 0.001
    dealiasing: bool = True
    multigrid: bool = False
    mg_levels: int = 3
    CFL: float = 0.1
    beta_squared: float = 5.0
    corner_smoothing: float = 0.15


@dataclass
class SpectralResultFields(Fields):
    """Spectral result fields returned to user after solving."""
    pass


@dataclass
class SpectralSolverFields:
    """Internal spectral solver arrays - current state and work buffers.

    Following the PN-PN-2 method:
    - Velocities (u, v) live on full (Nx+1) × (Ny+1) grid
    - Pressure (p) lives ONLY on inner (Nx-1) × (Ny-1) grid
    """
    # Current solution state - velocities on full grid
    u: np.ndarray
    v: np.ndarray

    # Pressure on INNER grid only (PN-PN-2)
    p: np.ndarray

    # Previous iteration (for convergence check)
    u_prev: np.ndarray
    v_prev: np.ndarray

    # RK4 stage buffers
    u_stage: np.ndarray
    v_stage: np.ndarray
    p_stage: np.ndarray  # Inner grid

    # Residuals
    R_u: np.ndarray  # Full grid
    R_v: np.ndarray  # Full grid
    R_p: np.ndarray  # Inner grid

    # Derivative buffers (full grid)
    du_dx: np.ndarray
    du_dy: np.ndarray
    dv_dx: np.ndarray
    dv_dy: np.ndarray
    lap_u: np.ndarray
    lap_v: np.ndarray

    # Pressure gradients interpolated to full grid
    dp_dx: np.ndarray  # Full grid
    dp_dy: np.ndarray  # Full grid

    # Pressure gradients on inner grid (before interpolation)
    dp_dx_inner: np.ndarray  # Inner grid
    dp_dy_inner: np.ndarray  # Inner grid

    @classmethod
    def allocate(cls, n_nodes_full: int, n_nodes_inner: int):
        """Allocate all arrays with proper sizes.

        Parameters
        ----------
        n_nodes_full : int
            Number of nodes on full (Nx+1) × (Ny+1) grid
        n_nodes_inner : int
            Number of nodes on inner (Nx-1) × (Ny-1) grid
        """
        return cls(
            # Current solution - velocities on full grid
            u=np.zeros(n_nodes_full),
            v=np.zeros(n_nodes_full),
            # Pressure on INNER grid only (PN-PN-2)
            p=np.zeros(n_nodes_inner),
            # Previous iteration
            u_prev=np.zeros(n_nodes_full),
            v_prev=np.zeros(n_nodes_full),
            # RK4 stage buffers
            u_stage=np.zeros(n_nodes_full),
            v_stage=np.zeros(n_nodes_full),
            p_stage=np.zeros(n_nodes_inner),  # Inner grid!
            # Residuals
            R_u=np.zeros(n_nodes_full),
            R_v=np.zeros(n_nodes_full),
            R_p=np.zeros(n_nodes_inner),  # Inner grid!
            # Derivative buffers (full grid)
            du_dx=np.zeros(n_nodes_full),
            du_dy=np.zeros(n_nodes_full),
            dv_dx=np.zeros(n_nodes_full),
            dv_dy=np.zeros(n_nodes_full),
            lap_u=np.zeros(n_nodes_full),
            lap_v=np.zeros(n_nodes_full),
            # Pressure gradients on full grid (interpolated)
            dp_dx=np.zeros(n_nodes_full),
            dp_dy=np.zeros(n_nodes_full),
            # Pressure gradients on inner grid (before interpolation)
            dp_dx_inner=np.zeros(n_nodes_inner),
            dp_dy_inner=np.zeros(n_nodes_inner),
        )
