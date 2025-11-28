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

# FVSolver requires petsc4py - make import optional
try:
    from .fv_solver import FVSolver
except ImportError as e:
    _fv_import_error = e

    def FVSolver(*args, **kwargs):
        raise ImportError(
            f"FVSolver requires petsc4py which is not installed: {_fv_import_error}"
        )


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
