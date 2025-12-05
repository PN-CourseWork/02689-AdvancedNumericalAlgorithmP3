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
from .polynomial import spectral_interpolate
from .transfer_operators import (
    TransferOperators,
    create_transfer_operators,
    FFTProlongation,
    FFTRestriction,
    PolynomialProlongation,
    InjectionRestriction,
)
from .corner_singularity import (
    CornerTreatment,
    SmoothingTreatment,
    SubtractionTreatment,
    create_corner_treatment,
)

# Note: CornerTreatmentMethod enum removed for simplicity - use string method names
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
    # Utilities
    "get_repo_root",
]
