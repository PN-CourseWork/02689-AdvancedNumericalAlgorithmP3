"""Data structures for solver configuration and results.

This module defines the configuration and result data structures
for lid-driven cavity solvers (both FV and spectral).
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import numpy as np

#========================================================
# Shared Data Classes 
# =======================================================

@dataclass
class Mesh: 




@dataclass
class Fields:
    """Base spatial solution fields."""
    u: np.ndarray
    u_prev: np.ndarray
    v: np.ndarray
    v_prev: np.ndarray
    p: np.ndarray
    x: np.ndarray
    y: np.ndarray
    grid_points: np.ndarray


@dataclass
class TimeSeries:
    """Time series data common to all solvers."""
    rel_residual: List[float]
    u_residual: List[float] 
    v_residual: List[float] 
    #TODO: Add the quantities stuff from the paper


@dataclass
class Meta:
    """Base solver metadata, config and convergence info."""
    # Physics parameters (required)
    Re: float

    # Grid parameters (with defaults)
    nx: int = 64
    ny: int = 64

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
    final_residual: float = 0.0


#=============================================================
# Finite Volume specific data classes
# ============================================================

@dataclass
class FVmeta(Meta):
    """FV-specific metadata with discretization parameters."""
    convection_scheme: str = "Upwind"
    limiter: str = "MUSCL"
    alpha_uv: float = 0.6
    alpha_p: float = 0.4



