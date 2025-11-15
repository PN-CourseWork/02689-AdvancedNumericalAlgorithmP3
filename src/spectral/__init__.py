"""Public API for Assignment 2 utilities."""

from .bvp import (
    BvpProblem,
    solve_legendre_collocation,
    solve_legendre_tau,
    solve_polar_bvp,
    solve_bvp,
)
from .spectral import (
    FourierEquispacedBasis,
    LegendreLobattoBasis,
    fourier_diff_matrix_cotangent,
    fourier_diff_matrix_complex,
    fourier_diff_matrix_on_interval,
    legendre_diff_matrix,
    legendre_mass_matrix,
)
from .tdp import (
    KdVSolver,
    soliton,
    two_soliton_initial,
    ManufacturedSolution,
    TimeIntegrator,
    get_time_integrator,
    RK3,
    RK4,
)
from .utils.plotting import get_repo_root

__all__ = [
    # BVP solvers
    "BvpProblem",
    "solve_legendre_collocation",
    "solve_legendre_tau",
    "solve_polar_bvp",
    "solve_bvp",
    # Spectral bases
    "LegendreLobattoBasis",
    "FourierEquispacedBasis",
    "legendre_diff_matrix",
    "legendre_mass_matrix",
    "fourier_diff_matrix_cotangent",
    "fourier_diff_matrix_complex",
    "fourier_diff_matrix_on_interval",
    # Time integrators and PDE solvers
    "TimeIntegrator",
    "get_time_integrator",
    "KdVSolver",
    "soliton",
    "two_soliton_initial",
    "ManufacturedSolution",
    "RK3",
    "RK4",
    # Utilities
    "get_repo_root",
]
