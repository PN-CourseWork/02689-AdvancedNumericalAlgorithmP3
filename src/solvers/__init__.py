"""Lid-driven cavity solver framework.

This module provides solvers for comparing finite volume and spectral methods.

Solver Hierarchy:
-----------------
LidDrivenCavitySolver (abstract base - defines problem)
├── FVSolver (finite volume with SIMPLE algorithm)
└── SpectralSolver (spectral methods with basic implementation)
    └── MultigridSpectralSolver (extends with multigrid acceleration)
"""

from .base_solver import LidDrivenCavitySolver
from .datastructures import (
    # Base classes (shared by all solvers)
    Parameters,
    Metrics,
    Fields,
    TimeSeries,
    # FV-specific
    FVParameters,
    FVSolverFields,
    # Spectral-specific
    SpectralParameters,
    SpectralSolverFields,
)
from .spectral_solver import SpectralSolver
from .fv_solver import FVSolver


__all__ = [
    # Base solver
    "LidDrivenCavitySolver",
    # Shared data structures
    "Parameters",
    "Metrics",
    "Fields",
    "TimeSeries",
    # FV solver
    "FVSolver",
    "FVParameters",
    "FVSolverFields",
    # Spectral solver
    "SpectralSolver",
    "SpectralParameters",
    "SpectralSolverFields",
]
