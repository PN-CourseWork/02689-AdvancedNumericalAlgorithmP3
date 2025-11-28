"""Data structures for solver configuration and results.

This module defines the configuration and result data structures
for lid-driven cavity solvers (both FV and spectral).

Structure:
- Parameters: Input configuration (logged to MLflow at start)
- Metrics: Output results (logged to MLflow at end)
- Fields: Spatial solution data
- TimeSeries: Convergence history
"""

from dataclasses import dataclass, asdict
from typing import Optional, List

import numpy as np
import pandas as pd


# ========================================================
# Parameters (Input Configuration)
# ========================================================


@dataclass
class Parameters:
    """Base solver parameters - input configuration for all solvers."""

    Re: float = 100
    lid_velocity: float = 1.0
    Lx: float = 1.0
    Ly: float = 1.0
    nx: int = 64
    ny: int = 64
    max_iterations: int = 500
    tolerance: float = 1e-4
    method: str = ""

    def to_dataframe(self):
        return pd.DataFrame([asdict(self)])


# ========================================================
# Metrics (Output Results)
# ========================================================


@dataclass
class Metrics:
    """Solver metrics - output results computed during/after solving."""

    iterations: int = 0
    converged: bool = False
    final_residual: float = float("inf")
    wall_time_seconds: float = 0.0
    u_momentum_residual: float = 0.0
    v_momentum_residual: float = 0.0
    continuity_residual: float = 0.0
    final_energy: float = 0.0
    final_enstrophy: float = 0.0
    final_palinstrophy: float = 0.0

    def to_dataframe(self):
        return pd.DataFrame([asdict(self)])


# ========================================================
# Fields (Spatial Solution Data)
# ========================================================


@dataclass
class Fields:
    """Spatial solution fields (u, v, p) on grid (x, y)."""

    u: np.ndarray
    v: np.ndarray
    p: np.ndarray
    x: np.ndarray
    y: np.ndarray

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with one row per grid point."""
        return pd.DataFrame(asdict(self))


# ========================================================
# Time Series (Convergence History)
# ========================================================


@dataclass
class TimeSeries:
    """Convergence history (one value per iteration)."""

    rel_iter_residual: List[float]
    u_residual: List[float]
    v_residual: List[float]
    continuity_residual: Optional[List[float]]
    energy: Optional[List[float]] = None
    enstrophy: Optional[List[float]] = None
    palinstrophy: Optional[List[float]] = None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with one row per iteration."""
        return pd.DataFrame({k: v for k, v in asdict(self).items() if v is not None})


# =============================================================
# Finite Volume Specific
# ============================================================


@dataclass
class FVParameters(Parameters):
    """FV solver parameters (extends Parameters with SIMPLE-specific settings)."""

    convection_scheme: str = "Upwind"
    limiter: str = "MUSCL"
    alpha_uv: float = 0.6  # velocity under-relaxation
    alpha_p: float = 0.4  # pressure under-relaxation
    linear_solver_tol: float = 1e-6  # PETSc linear solver tolerance
    method: str = "FV-SIMPLE"


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

    # PETSc KSP objects for solver reuse
    ksp_u: object = None
    ksp_v: object = None
    ksp_p: object = None

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


# =====================================================
# Spectral Specific
# =====================================================


@dataclass
class SpectralParameters(Parameters):
    """Spectral solver parameters (nx/ny = polynomial order N, giving N+1 nodes)."""

    basis_type: str = "legendre"  # "legendre" or "chebyshev"
    CFL: float = 0.1
    beta_squared: float = 5.0  # artificial compressibility
    corner_smoothing: float = 0.15
    method: str = "Spectral-AC"


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
