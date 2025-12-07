"""Spectral solver package."""

from solvers.spectral.sg import SGSolver
from solvers.spectral.fsg import FSGSolver
from solvers.spectral.vmg import VMGSolver

__all__ = ["SGSolver", "FSGSolver", "VMGSolver"]
