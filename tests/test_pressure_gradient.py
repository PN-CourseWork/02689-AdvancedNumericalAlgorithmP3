"""Tests for pressure gradient computation in PN-PN-2 formulation.

The key insight is that for PN-PN-2 (velocities on full grid, pressure on inner grid),
we must interpolate pressure to the full grid BEFORE computing gradients.
Using a differentiation matrix built on inner nodes gives incorrect results because
the inner nodes are not proper Gauss-Lobatto points.
"""

import numpy as np
import pytest
from numpy.polynomial.chebyshev import chebvander


class TestInterpolationMatrix:
    """Tests for spectral interpolation from inner to full grid."""

    def build_interpolation_matrix(self, nodes_inner, nodes_full):
        """Build Chebyshev interpolation matrix from inner to full grid."""
        n_inner = len(nodes_inner)
        a, b = nodes_full[0], nodes_full[-1]
        xi_inner = 2 * (nodes_inner - a) / (b - a) - 1
        xi_full = 2 * (nodes_full - a) / (b - a) - 1

        V_inner = chebvander(xi_inner, n_inner - 1)
        V_full = chebvander(xi_full, n_inner - 1)
        Interp = V_full @ np.linalg.solve(V_inner, np.eye(n_inner))
        return Interp

    def test_interpolation_polynomial(self, chebyshev_basis):
        """Interpolation is exact for polynomials of appropriate degree."""
        N = 15
        nodes_full = chebyshev_basis.nodes(N + 1)
        nodes_inner = nodes_full[1:-1]

        Interp = self.build_interpolation_matrix(nodes_inner, nodes_full)

        # p(x) = x^2 on inner grid
        p_inner = nodes_inner**2
        p_full_interp = Interp @ p_inner
        p_full_exact = nodes_full**2

        # Should be exact (up to machine precision)
        assert np.allclose(p_full_interp, p_full_exact, atol=1e-12)

    def test_interpolation_trigonometric(self, chebyshev_basis):
        """Interpolation is spectrally accurate for smooth functions."""
        N = 15
        nodes_full = chebyshev_basis.nodes(N + 1)
        nodes_inner = nodes_full[1:-1]

        Interp = self.build_interpolation_matrix(nodes_inner, nodes_full)

        # p(x) = sin(pi*x) on inner grid
        p_inner = np.sin(np.pi * nodes_inner)
        p_full_interp = Interp @ p_inner
        p_full_exact = np.sin(np.pi * nodes_full)

        # Interior values should be exact (these are the original inner values)
        assert np.allclose(p_full_interp[1:-1], p_full_exact[1:-1], atol=1e-14)

        # Boundary extrapolation should be spectrally accurate (O(1e-11) for N=15)
        # For sin(pi*x), the exact boundary values are 0
        assert np.allclose(p_full_interp, p_full_exact, atol=1e-10)

    def test_linear_extrapolation_is_inaccurate(self, chebyshev_basis):
        """Demonstrate that linear extrapolation is NOT spectrally accurate."""
        N = 15
        nodes_full = chebyshev_basis.nodes(N + 1)
        nodes_inner = nodes_full[1:-1]

        # Linear extrapolation (the wrong approach)
        def linear_extrapolate(inner_vals):
            full = np.zeros(len(nodes_full))
            full[1:-1] = inner_vals
            full[0] = 2 * full[1] - full[2]  # Linear extrapolation at left
            full[-1] = 2 * full[-2] - full[-3]  # Linear extrapolation at right
            return full

        # Test with x^2
        p_inner = nodes_inner**2
        p_full_linear = linear_extrapolate(p_inner)
        p_full_exact = nodes_full**2

        # Linear extrapolation has O(1) error at boundaries
        boundary_error = max(
            abs(p_full_linear[0] - p_full_exact[0]),
            abs(p_full_linear[-1] - p_full_exact[-1]),
        )
        assert boundary_error > 0.01, "Linear extrapolation should have significant error"


class TestPressureGradient:
    """Tests for pressure gradient computation."""

    def test_pressure_gradient_spectral(self, chebyshev_basis):
        """Pressure gradient with spectral interpolation is accurate."""
        N = 15
        nodes_full = chebyshev_basis.nodes(N + 1)
        nodes_inner = nodes_full[1:-1]

        D_full = chebyshev_basis.diff_matrix(nodes_full)

        # Build interpolation matrix
        n_inner = len(nodes_inner)
        a, b = nodes_full[0], nodes_full[-1]
        xi_inner = 2 * (nodes_inner - a) / (b - a) - 1
        xi_full = 2 * (nodes_full - a) / (b - a) - 1
        V_inner = chebvander(xi_inner, n_inner - 1)
        V_full = chebvander(xi_full, n_inner - 1)
        Interp = V_full @ np.linalg.solve(V_inner, np.eye(n_inner))

        # Correct approach: interpolate p to full grid, then differentiate
        p_inner = np.sin(np.pi * nodes_inner)
        p_full = Interp @ p_inner
        dp_dx = D_full @ p_full

        dp_dx_exact = np.pi * np.cos(np.pi * nodes_full)

        # Should be spectrally accurate
        assert np.allclose(dp_dx, dp_dx_exact, atol=1e-10)

    def test_inner_grid_diff_is_wrong(self, chebyshev_basis):
        """Differentiation directly on inner grid is WRONG."""
        N = 15
        nodes_full = chebyshev_basis.nodes(N + 1)
        nodes_inner = nodes_full[1:-1]

        # WRONG: Build diff matrix on inner nodes
        D_inner = chebyshev_basis.diff_matrix(nodes_inner)

        # Test d/dx[x^2] = 2x
        f_inner = nodes_inner**2
        df_inner = D_inner @ f_inner
        df_exact = 2 * nodes_inner

        # This has O(1) error because inner nodes are NOT proper Gauss-Lobatto points
        max_error = np.max(np.abs(df_inner - df_exact))
        assert max_error > 1.0, f"Expected large error, got {max_error}"

    def test_2d_pressure_gradient(self, chebyshev_basis):
        """2D pressure gradient using tensor product interpolation."""
        N = 8
        nodes_full = chebyshev_basis.nodes(N + 1)
        nodes_inner = nodes_full[1:-1]

        D_full = chebyshev_basis.diff_matrix(nodes_full)

        # Build interpolation matrices
        n_inner = len(nodes_inner)
        a, b = nodes_full[0], nodes_full[-1]
        xi_inner = 2 * (nodes_inner - a) / (b - a) - 1
        xi_full = 2 * (nodes_full - a) / (b - a) - 1
        V_inner = chebvander(xi_inner, n_inner - 1)
        V_full = chebvander(xi_full, n_inner - 1)
        Interp_1d = V_full @ np.linalg.solve(V_inner, np.eye(n_inner))

        # Build 2D operators using Kronecker products
        I_full = np.eye(N + 1)
        Dx = np.kron(D_full, I_full)
        Dy = np.kron(I_full, D_full)

        # Create inner and full 2D grids
        X_inner, Y_inner = np.meshgrid(nodes_inner, nodes_inner, indexing="ij")
        X_full, Y_full = np.meshgrid(nodes_full, nodes_full, indexing="ij")

        # p(x,y) = x^2 + y^2 on inner grid
        p_inner_2d = X_inner**2 + Y_inner**2

        # 2D interpolation: p_full = Interp_x @ p_inner @ Interp_y.T
        p_full_2d = Interp_1d @ p_inner_2d @ Interp_1d.T
        p_full = p_full_2d.ravel()

        # Compute gradients
        dp_dx = (Dx @ p_full).reshape(X_full.shape)
        dp_dy = (Dy @ p_full).reshape(Y_full.shape)

        # Exact gradients
        dp_dx_exact = 2 * X_full
        dp_dy_exact = 2 * Y_full

        assert np.allclose(dp_dx, dp_dx_exact, atol=1e-10)
        assert np.allclose(dp_dy, dp_dy_exact, atol=1e-10)
