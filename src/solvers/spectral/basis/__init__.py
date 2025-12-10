"""Spectral basis utilities."""

from solvers.spectral.basis.spectral import (
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
from solvers.spectral.basis.polynomial import (
    spectral_interpolate,
    spectral_interpolate_2d,
    spectral_interpolate_line,
)

__all__ = [
    "LegendreLobattoBasis",
    "ChebyshevLobattoBasis",
    "FourierEquispacedBasis",
    "legendre_diff_matrix",
    "legendre_mass_matrix",
    "chebyshev_diff_matrix",
    "chebyshev_gauss_lobatto_nodes",
    "fourier_diff_matrix_cotangent",
    "fourier_diff_matrix_complex",
    "fourier_diff_matrix_on_interval",
    "spectral_interpolate",
    "spectral_interpolate_2d",
    "spectral_interpolate_line",
]
