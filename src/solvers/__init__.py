"""Lid-driven cavity solver framework.

This module provides solvers for comparing finite volume and spectral methods.

Solver Hierarchy:
-----------------
LidDrivenCavitySolver (abstract base - defines problem)
├── FVSolver (finite volume with SIMPLE algorithm)
└── SGSolver (spectral single grid base)
    ├── FSGSolver (Full Single Grid multigrid)
    ├── VMGSolver (V-cycle Multigrid)
    └── FMGSolver (Full MultiGrid)
"""

from .base import LidDrivenCavitySolver
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
from solvers.fv.solver import FVSolver
from solvers.spectral.sg import SGSolver
from solvers.spectral.fsg import FSGSolver
from solvers.spectral.vmg import VMGSolver
from solvers.spectral.fmg import FMGSolver


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
    # Spectral solvers
    "SGSolver",
    "FSGSolver",
    "VMGSolver",
    "FMGSolver",
    "SpectralParameters",
    "SpectralSolverFields",
]
