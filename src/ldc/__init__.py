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
    MetaConfig,
    FVinfo,
    SpectralInfo,
    FVFields,
    TimeSeries,
)
from .fv_solver import FVSolver

__all__ = [
    # Base classes
    "LidDrivenCavitySolver",
    # Configurations
    "MetaConfig",
    "FVinfo",
    "SpectralInfo",
    # Data structures
    "FVFields",
    "TimeSeries",
    # Concrete solvers
    "FVSolver",
]
