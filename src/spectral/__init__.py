"""Compatibility layer for spectral utilities (forwarded to new locations)."""

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
from solvers.spectral.basis.polynomial import spectral_interpolate
from solvers.spectral.operators.transfer_operators import (
    FFTProlongation,
    FFTRestriction,
    InjectionRestriction,
    PolynomialProlongation,
    TransferOperators,
    create_transfer_operators,
)
from solvers.spectral.operators.corner import (
    CornerTreatment,
    SmoothingTreatment,
    SubtractionTreatment,
    create_corner_treatment,
)

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
    # Interpolation
    "spectral_interpolate",
    # Transfer operators (multigrid)
    "TransferOperators",
    "create_transfer_operators",
    "FFTProlongation",
    "FFTRestriction",
    "PolynomialProlongation",
    "InjectionRestriction",
    # Corner singularity treatment
    "CornerTreatment",
    "SmoothingTreatment",
    "SubtractionTreatment",
    "create_corner_treatment",
]
