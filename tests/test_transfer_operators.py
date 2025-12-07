"""Tests for FFT transfer operators (prolongation and restriction).

Tests prolongation and restriction operators for multigrid:
- 1D and 2D prolongation (coarse -> fine)
- 1D and 2D restriction (fine -> coarse)
- Round-trip accuracy
- Multigrid hierarchy compatibility
"""

import numpy as np
import pytest

from solvers.spectral.operators.transfer_operators import (
    FFTProlongation,
    FFTRestriction,
    PolynomialProlongation,
    InjectionRestriction,
)


def chebyshev_lobatto_nodes(N):
    """Chebyshev-Gauss-Lobatto nodes: x_j = cos(πj/N) for j=0,...,N."""
    j = np.arange(N + 1)
    return np.cos(np.pi * j / N)


# Test functions
def poly4(x):
    """Polynomial: x^4 + 2x^2 + 1 (degree 4, exact for N>=4)"""
    return x**4 + 2 * x**2 + 1


def poly2(x):
    """Polynomial: x^2 + x + 1 (degree 2, exact for N>=2)"""
    return x**2 + x + 1


def smooth_func(x):
    """Smooth function: sin(πx) (not polynomial, tests interpolation accuracy)"""
    return np.sin(np.pi * x)


class TestProlongation1D:
    """Tests for 1D prolongation operators."""

    @pytest.fixture
    def fft_prolong(self):
        return FFTProlongation()

    @pytest.fixture
    def poly_prolong(self):
        return PolynomialProlongation()

    @pytest.mark.parametrize("N_c,N_f", [(4, 8), (8, 16), (16, 32)])
    def test_polynomial_exact(self, fft_prolong, poly_prolong, N_c, N_f):
        """Polynomial prolongation should be exact to machine precision."""
        x_c = chebyshev_lobatto_nodes(N_c)
        x_f = chebyshev_lobatto_nodes(N_f)

        f_c = poly4(x_c)
        f_f_exact = poly4(x_f)

        f_fft = fft_prolong.prolongate_1d(f_c, N_f + 1)
        f_poly = poly_prolong.prolongate_1d(f_c, N_f + 1)

        assert np.allclose(f_fft, f_f_exact, atol=1e-12)
        assert np.allclose(f_poly, f_f_exact, atol=1e-12)

    @pytest.mark.parametrize("N_c,N_f", [(8, 16), (16, 32)])
    def test_smooth_function_spectral_accuracy(self, fft_prolong, N_c, N_f):
        """Smooth function prolongation should have spectral accuracy."""
        x_c = chebyshev_lobatto_nodes(N_c)
        x_f = chebyshev_lobatto_nodes(N_f)

        f_c = smooth_func(x_c)
        f_f_exact = smooth_func(x_f)

        f_fft = fft_prolong.prolongate_1d(f_c, N_f + 1)
        err = np.max(np.abs(f_fft - f_f_exact))

        # Spectral accuracy: error should be small for smooth functions
        assert err < 1e-3


class TestRestriction1D:
    """Tests for 1D restriction operators."""

    @pytest.fixture
    def fft_restrict(self):
        return FFTRestriction()

    @pytest.fixture
    def inj_restrict(self):
        return InjectionRestriction()

    @pytest.mark.parametrize("N_f,N_c", [(8, 4), (16, 8), (32, 16)])
    def test_polynomial_exact(self, fft_restrict, inj_restrict, N_f, N_c):
        """Polynomial restriction should be exact to machine precision."""
        x_f = chebyshev_lobatto_nodes(N_f)
        x_c = chebyshev_lobatto_nodes(N_c)

        f_f = poly4(x_f)
        f_c_exact = poly4(x_c)

        f_fft = fft_restrict.restrict_1d(f_f, N_c + 1)
        f_inj = inj_restrict.restrict_1d(f_f, N_c + 1)

        assert np.allclose(f_fft, f_c_exact, atol=1e-12)
        assert np.allclose(f_inj, f_c_exact, atol=1e-12)


class TestRoundTrip:
    """Tests for round-trip: prolongate then restrict."""

    def test_roundtrip_preserves_polynomial(self):
        """Round-trip (coarse -> fine -> coarse) should preserve polynomial."""
        fft_prolong = FFTProlongation()
        fft_restrict = FFTRestriction()

        for N_c in [4, 8, 16]:
            N_f = 2 * N_c
            x_c = chebyshev_lobatto_nodes(N_c)

            f_c = poly4(x_c)

            # Prolongate to fine
            f_f = fft_prolong.prolongate_1d(f_c, N_f + 1)

            # Restrict back to coarse
            f_c_back = fft_restrict.restrict_1d(f_f, N_c + 1)

            assert np.allclose(f_c_back, f_c, atol=1e-12)


class TestProlongation2D:
    """Tests for 2D prolongation operators."""

    @pytest.fixture
    def fft_prolong(self):
        return FFTProlongation()

    @pytest.fixture
    def poly_prolong(self):
        return PolynomialProlongation()

    @pytest.mark.parametrize("N_c", [4, 8])
    def test_2d_polynomial_exact(self, fft_prolong, poly_prolong, N_c):
        """2D polynomial prolongation should be exact."""
        N_f = 2 * N_c

        x_c = chebyshev_lobatto_nodes(N_c)
        x_f = chebyshev_lobatto_nodes(N_f)

        X_c, Y_c = np.meshgrid(x_c, x_c, indexing="ij")
        X_f, Y_f = np.meshgrid(x_f, x_f, indexing="ij")

        f_c = X_c**2 + Y_c**2 + X_c * Y_c + 1
        f_f_exact = X_f**2 + Y_f**2 + X_f * Y_f + 1

        f_fft = fft_prolong.prolongate_2d(f_c, (N_f + 1, N_f + 1))
        f_poly = poly_prolong.prolongate_2d(f_c, (N_f + 1, N_f + 1))

        assert np.allclose(f_fft, f_f_exact, atol=1e-12)
        assert np.allclose(f_poly, f_f_exact, atol=1e-12)


class TestRestriction2D:
    """Tests for 2D restriction operators."""

    @pytest.fixture
    def fft_restrict(self):
        return FFTRestriction()

    @pytest.fixture
    def inj_restrict(self):
        return InjectionRestriction()

    @pytest.mark.parametrize("N_f", [8, 16])
    def test_2d_polynomial_exact(self, fft_restrict, inj_restrict, N_f):
        """2D polynomial restriction should be exact."""
        N_c = N_f // 2

        x_f = chebyshev_lobatto_nodes(N_f)
        x_c = chebyshev_lobatto_nodes(N_c)

        X_f, Y_f = np.meshgrid(x_f, x_f, indexing="ij")
        X_c, Y_c = np.meshgrid(x_c, x_c, indexing="ij")

        f_f = X_f**2 + Y_f**2 + X_f * Y_f + 1
        f_c_exact = X_c**2 + Y_c**2 + X_c * Y_c + 1

        f_fft = fft_restrict.restrict_2d(f_f, (N_c + 1, N_c + 1))
        f_inj = inj_restrict.restrict_2d(f_f, (N_c + 1, N_c + 1))

        assert np.allclose(f_fft, f_c_exact, atol=1e-12)
        assert np.allclose(f_inj, f_c_exact, atol=1e-12)


class TestMultigridHierarchy:
    """Tests for multigrid hierarchy grid sequences."""

    @pytest.mark.parametrize(
        "levels",
        [
            [4, 8, 16],
            [8, 16, 32],
            [12, 24, 48],
            [16, 32, 64],
        ],
    )
    def test_hierarchy_prolongation(self, levels):
        """Test prolongation through multigrid hierarchy."""
        fft_prolong = FFTProlongation()

        for i in range(len(levels) - 1):
            N_c, N_f = levels[i], levels[i + 1]
            x_c = chebyshev_lobatto_nodes(N_c)
            x_f = chebyshev_lobatto_nodes(N_f)

            f_c = poly4(x_c)
            f_f_exact = poly4(x_f)

            f_f = fft_prolong.prolongate_1d(f_c, N_f + 1)

            assert np.allclose(f_f, f_f_exact, atol=1e-12)

    @pytest.mark.parametrize(
        "levels",
        [
            [4, 8, 16],
            [8, 16, 32],
            [12, 24, 48],
            [16, 32, 64],
        ],
    )
    def test_hierarchy_restriction(self, levels):
        """Test restriction through multigrid hierarchy."""
        fft_restrict = FFTRestriction()

        for i in range(len(levels) - 1, 0, -1):
            N_f, N_c = levels[i], levels[i - 1]
            x_f = chebyshev_lobatto_nodes(N_f)
            x_c = chebyshev_lobatto_nodes(N_c)

            f_f = poly4(x_f)
            f_c_exact = poly4(x_c)

            f_c = fft_restrict.restrict_1d(f_f, N_c + 1)

            assert np.allclose(f_c, f_c_exact, atol=1e-12)
