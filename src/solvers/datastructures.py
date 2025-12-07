"""Data structures for solver configuration and results.

Architecture: Params vs Metrics (following TEMPLATE pattern)

             Params (input/config)         Metrics (output/results)
             ─────────────────────         ────────────────────────
Global       Parameters                    Metrics
             Re, nx, ny, tolerance...      wall_time, converged, iterations...

Timeseries   -                             TimeSeries
                                           residual_history[], energy[]...

Spatial      -                             Fields
                                           u, v, p arrays on grid
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd


# ============================================================================
# Parameters (Input Configuration) - logged to MLflow as params
# ============================================================================


@dataclass
class Parameters:
    """Base solver parameters - input configuration for all solvers."""

    name: str = ""
    Re: float = 100
    lid_velocity: float = 1.0
    Lx: float = 1.0
    Ly: float = 1.0
    nx: int = 64
    ny: int = 64
    max_iterations: int = 500
    tolerance: float = 1e-4
    method: str = ""

    def to_mlflow(self) -> dict:
        """Convert to MLflow-compatible params dict."""
        return {
            k: (int(v) if isinstance(v, bool) else v) for k, v in self.__dict__.items()
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.to_mlflow()])


# ============================================================================
# Metrics (Output Results) - logged to MLflow as metrics
# ============================================================================


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

    def to_mlflow(self) -> dict:
        """Convert to MLflow-compatible dict (bools as int, skip inf)."""
        return {
            k: (int(v) if isinstance(v, bool) else v)
            for k, v in self.__dict__.items()
            if v != float("inf")  # Skip unset values
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.to_mlflow()])


# ============================================================================
# TimeSeries (Convergence History) - logged to MLflow as step metrics
# ============================================================================


@dataclass
class TimeSeries:
    """Convergence history (one value per iteration)."""

    rel_iter_residual: List[float] = field(default_factory=list)
    u_residual: List[float] = field(default_factory=list)
    v_residual: List[float] = field(default_factory=list)
    continuity_residual: List[float] = field(default_factory=list)
    energy: List[float] = field(default_factory=list)
    enstrophy: List[float] = field(default_factory=list)
    palinstrophy: List[float] = field(default_factory=list)

    def to_mlflow_batch(self) -> list:
        """Convert timeseries to MLflow Metric objects for batch logging."""
        from mlflow.entities import Metric

        return [
            Metric(key=name, value=value, timestamp=0, step=step)
            for name, values in self.__dict__.items()
            if values  # Skip empty lists
            for step, value in enumerate(values)
            if value is not None
        ]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with one row per iteration."""
        return pd.DataFrame({k: v for k, v in self.__dict__.items() if v})


# ============================================================================
# Fields (Spatial Solution Data) - saved to HDF5 artifact
# ============================================================================


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
        return pd.DataFrame(
            {"x": self.x, "y": self.y, "u": self.u, "v": self.v, "p": self.p}
        )


# ============================================================================
# Finite Volume Specific
# ============================================================================


@dataclass
class FVParameters(Parameters):
    """FV solver parameters (extends Parameters with SIMPLE-specific settings)."""

    convection_scheme: str = "Upwind"
    limiter: str = "MUSCL"
    alpha_uv: float = 0.6
    alpha_p: float = 0.4
    linear_solver_tol: float = 1e-6
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

    # Scipy preconditioners for solver reuse
    M_u: object = None
    M_v: object = None
    M_p: object = None

    @classmethod
    def allocate(cls, n_cells: int, n_faces: int):
        """Allocate all arrays with proper sizes."""
        return cls(
            u=np.zeros(n_cells),
            v=np.zeros(n_cells),
            p=np.zeros(n_cells),
            mdot=np.zeros(n_faces),
            u_prev=np.zeros(n_cells),
            v_prev=np.zeros(n_cells),
            grad_p=np.zeros((n_cells, 2)),
            grad_u=np.zeros((n_cells, 2)),
            grad_v=np.zeros((n_cells, 2)),
            grad_p_prime=np.zeros((n_cells, 2)),
            grad_p_bar=np.zeros((n_faces, 2)),
            bold_D=np.zeros((n_cells, 2)),
            bold_D_bar=np.zeros((n_faces, 2)),
            U_star_rc=np.zeros((n_faces, 2)),
            U_prime_face=np.zeros((n_faces, 2)),
            u_prime=np.zeros(n_cells),
            v_prime=np.zeros(n_cells),
            mdot_star=np.zeros(n_faces),
            mdot_prime=np.zeros(n_faces),
        )


# ============================================================================
# Spectral Specific
# ============================================================================


@dataclass
class SpectralParameters(Parameters):
    """Spectral solver parameters (nx/ny = polynomial order N, giving N+1 nodes)."""

    basis_type: str = "legendre"
    CFL: float = 0.1
    beta_squared: float = 5.0
    method: str = "Spectral-AC"

    # Corner singularity treatment
    # Options: "smoothing" (simple cosine smoothing) or "subtraction" (Botella & Peyret 1998)
    corner_treatment: str = "smoothing"
    corner_smoothing: float = 0.15  # smoothing width for smoothing method

    # Multigrid settings
    multigrid: str = "none"  # "none", "fsg", "vmg", "fmg"
    n_levels: int = 3
    coarse_tolerance_factor: float = 10.0

    # Transfer operators (prolongation/restriction) for multigrid
    # Options: "fft" (Zhang & Xi 2010 paper) or "polynomial"/"injection"
    prolongation_method: str = "fft"
    restriction_method: str = "fft"

    # VMG-specific: smoothing iterations per level (coarse-to-fine order)
    # Default is None, which uses [n_levels, n_levels-1, ..., 1]
    pre_smoothing: list = None
    post_smoothing: list = None

    # VMG coarse correction damping (0 = no correction, 1 = full correction)
    # With proper FAS tau correction, use full correction (1.0)
    correction_damping: float = 1.0


@dataclass
class SpectralSolverFields:
    """Internal spectral solver arrays - current state and work buffers.

    Following the PN-PN-2 method:
    - Velocities (u, v) live on full (Nx+1) x (Ny+1) grid
    - Pressure (p) lives ONLY on inner (Nx-1) x (Ny-1) grid
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
    p_stage: np.ndarray

    # Residuals
    R_u: np.ndarray
    R_v: np.ndarray
    R_p: np.ndarray

    # Derivative buffers (full grid)
    du_dx: np.ndarray
    du_dy: np.ndarray
    dv_dx: np.ndarray
    dv_dy: np.ndarray
    lap_u: np.ndarray
    lap_v: np.ndarray

    # Pressure gradients interpolated to full grid
    dp_dx: np.ndarray
    dp_dy: np.ndarray

    # Pressure gradients on inner grid (before interpolation)
    dp_dx_inner: np.ndarray
    dp_dy_inner: np.ndarray

    @classmethod
    def allocate(cls, n_nodes_full: int, n_nodes_inner: int):
        """Allocate all arrays with proper sizes."""
        return cls(
            u=np.zeros(n_nodes_full),
            v=np.zeros(n_nodes_full),
            p=np.zeros(n_nodes_inner),
            u_prev=np.zeros(n_nodes_full),
            v_prev=np.zeros(n_nodes_full),
            u_stage=np.zeros(n_nodes_full),
            v_stage=np.zeros(n_nodes_full),
            p_stage=np.zeros(n_nodes_inner),
            R_u=np.zeros(n_nodes_full),
            R_v=np.zeros(n_nodes_full),
            R_p=np.zeros(n_nodes_inner),
            du_dx=np.zeros(n_nodes_full),
            du_dy=np.zeros(n_nodes_full),
            dv_dx=np.zeros(n_nodes_full),
            dv_dy=np.zeros(n_nodes_full),
            lap_u=np.zeros(n_nodes_full),
            lap_v=np.zeros(n_nodes_full),
            dp_dx=np.zeros(n_nodes_full),
            dp_dy=np.zeros(n_nodes_full),
            dp_dx_inner=np.zeros(n_nodes_inner),
            dp_dy_inner=np.zeros(n_nodes_inner),
        )
