"""Pytest configuration and fixtures for spectral solver tests."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def chebyshev_basis():
    """Chebyshev-Gauss-Lobatto basis on [0, 1]."""
    from solvers.spectral.basis.spectral import ChebyshevLobattoBasis

    return ChebyshevLobattoBasis(domain=(0.0, 1.0))


@pytest.fixture
def legendre_basis():
    """Legendre-Gauss-Lobatto basis on [0, 1]."""
    from solvers.spectral.basis.spectral import LegendreLobattoBasis

    return LegendreLobattoBasis(domain=(0.0, 1.0))


@pytest.fixture
def small_grid_params():
    """Parameters for a small 8x8 test grid."""
    return {
        "Re": 100,
        "nx": 8,
        "ny": 8,
        "tolerance": 1e-6,
        "max_iterations": 1000,
        "lid_velocity": 1.0,
        "Lx": 1.0,
        "Ly": 1.0,
        "CFL": 0.5,
        "beta_squared": 5.0,
        "basis_type": "chebyshev",
        "corner_treatment": "smoothing",
    }


@pytest.fixture
def medium_grid_params():
    """Parameters for a medium 15x15 grid (used for Ghia validation)."""
    return {
        "Re": 100,
        "nx": 15,
        "ny": 15,
        "tolerance": 1e-6,
        "max_iterations": 100000,
        "lid_velocity": 1.0,
        "Lx": 1.0,
        "Ly": 1.0,
        "CFL": 0.5,
        "beta_squared": 5.0,
        "basis_type": "chebyshev",
        "corner_treatment": "smoothing",
    }
