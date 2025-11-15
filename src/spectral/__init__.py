"""Spectral methods for Navier-Stokes solver."""

from .spectral import (
    LegendreLobattoBasis,
    legendre_diff_matrix,
)
from .utils.plotting import get_repo_root

__all__ = [
    # Spectral bases
    "LegendreLobattoBasis",
    "legendre_diff_matrix",
    # Utilities
    "get_repo_root",
]
