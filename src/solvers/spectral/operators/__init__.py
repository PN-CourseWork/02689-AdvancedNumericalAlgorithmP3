"""Spectral operators (corner treatment, transfer operators)."""

from solvers.spectral.operators.corner import (
    CornerTreatment,
    SmoothingTreatment,
    SaadTreatment,
    PolynomialTreatment,
    create_corner_treatment,
)
from solvers.spectral.operators.transfer_operators import (
    FFTProlongation,
    FFTRestriction,
    InjectionRestriction,
    PolynomialProlongation,
    TransferOperators,
    create_transfer_operators,
)

__all__ = [
    "CornerTreatment",
    "SmoothingTreatment",
    "SaadTreatment",
    "PolynomialTreatment",
    "create_corner_treatment",
    "TransferOperators",
    "create_transfer_operators",
    "FFTProlongation",
    "FFTRestriction",
    "PolynomialProlongation",
    "InjectionRestriction",
]
