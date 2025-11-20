"""Data structures for solver configuration and results.

This module defines the configuration and result data structures
for lid-driven cavity solvers (both FV and spectral).
"""
from dataclasses import dataclass
from typing import Optional, List

import numpy as np

#========================================================
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
    




@dataclass
class TimeSeries:
    """Time series data common to all solvers."""
    iter_residual: List[float]
    u_residual: List[float]
    v_residual: List[float]
    continuity_residual: Optional[List[float]]
    #TODO: Add the quantities stuff from the paper


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


#=============================================================
# Finite Volume specific data classes
# ============================================================

@dataclass
class FVinfo(MetaConfig):
    """FV-specific metadata with discretization parameters."""
    convection_scheme: str = ""
    alpha_uv: float = 0.7
    alpha_p: float = 0.3


@dataclass
class FVFields(Fields):
    """Internal FV solver arrays - current state, previous iteration, and work buffers."""
    # Current solution state
    mdot: np.ndarray

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
            x=np.zeros(n_cells),
            y=np.zeros(n_cells),
            grid_points=np.zeros((n_cells, 2)),
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
#=====================================================

@dataclass
class SpectralInfo(MetaConfig):
    """Spectral-specific metadata with discretization parameters."""
    Nx: int = 64
    Ny: int = 64
    differentiation_method: str = "fft"  # 'fft', 'chebyshev', 'matrix'
    time_scheme: str = "rk4"
    dt: float = 0.001
    dealiasing: bool = True
    multigrid: bool = False
    mg_levels: int = 3


