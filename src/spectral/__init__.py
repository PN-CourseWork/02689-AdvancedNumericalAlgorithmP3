"""Spectral methods for Navier-Stokes solver."""

from .spectral import (
    ChebyshevLobattoBasis,
    FourierEquispacedBasis,
    LegendreLobattoBasis,
    chebyshev_diff_matrix,
    chebyshev_gauss_lobatto_nodes,
    fourier_diff_matrix_complex,
    fourier_diff_matrix_cotangent,
    fourier_diff_matrix_on_interval,
    legendre_diff_matrix,
    legendre_mass_matrix,
)
from .utils.plotting import get_repo_root

__all__ = [
    # Spectral bases
    "LegendreLobattoBasis",
    "ChebyshevLobattoBasis",
    "FourierEquispacedBasis",
    # Differentiation matrices
    "legendre_diff_matrix",
    "legendre_mass_matrix",
    "chebyshev_diff_matrix",
    "chebyshev_gauss_lobatto_nodes",
    "fourier_diff_matrix_cotangent",
    "fourier_diff_matrix_complex",
    "fourier_diff_matrix_on_interval",
    # Utilities
    "get_repo_root",
]
