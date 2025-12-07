"""Tests for spectral basis functions: nodes and differentiation matrices."""

import numpy as np
import pytest


class TestChebyshevNodes:
    """Tests for Chebyshev-Gauss-Lobatto nodes."""

    def test_nodes_correct_count(self, chebyshev_basis):
        """Nodes array has correct size."""
        for N in [4, 8, 16, 32]:
            nodes = chebyshev_basis.nodes(N)
            assert len(nodes) == N

    def test_nodes_endpoints(self, chebyshev_basis):
        """Nodes include domain endpoints [0, 1]."""
        nodes = chebyshev_basis.nodes(9)
        assert np.isclose(nodes[0], 0.0, atol=1e-14)
        assert np.isclose(nodes[-1], 1.0, atol=1e-14)

    def test_nodes_monotonic(self, chebyshev_basis):
        """Nodes are strictly increasing."""
        nodes = chebyshev_basis.nodes(17)
        diffs = np.diff(nodes)
        assert np.all(diffs > 0)

    def test_nodes_symmetric(self, chebyshev_basis):
        """Nodes are symmetric about domain center."""
        # Use even number of nodes to avoid node exactly at center
        nodes = chebyshev_basis.nodes(16)
        center = 0.5
        left = nodes[nodes < center]
        right = nodes[nodes > center][::-1]  # reverse right side
        assert len(left) == len(right)
        assert np.allclose(center - left, right - center, atol=1e-14)


class TestDifferentiationMatrix:
    """Tests for Chebyshev differentiation matrix."""

    def test_diff_matrix_shape(self, chebyshev_basis):
        """Differentiation matrix has correct shape."""
        N = 9
        nodes = chebyshev_basis.nodes(N)
        D = chebyshev_basis.diff_matrix(nodes)
        assert D.shape == (N, N)

    def test_diff_matrix_row_sum(self, chebyshev_basis):
        """Row sums are zero (derivative of constant is zero)."""
        N = 17
        nodes = chebyshev_basis.nodes(N)
        D = chebyshev_basis.diff_matrix(nodes)
        row_sums = np.sum(D, axis=1)
        assert np.allclose(row_sums, 0, atol=1e-12)

    @pytest.mark.parametrize("N", [8, 16, 32])
    def test_diff_polynomial(self, chebyshev_basis, N):
        """Exact differentiation of low-degree polynomials."""
        nodes = chebyshev_basis.nodes(N + 1)
        D = chebyshev_basis.diff_matrix(nodes)

        # d/dx[x^2] = 2x
        f = nodes**2
        df_exact = 2 * nodes
        df_spectral = D @ f
        assert np.allclose(df_spectral, df_exact, atol=1e-10)

        # d/dx[x^3] = 3x^2
        f = nodes**3
        df_exact = 3 * nodes**2
        df_spectral = D @ f
        assert np.allclose(df_spectral, df_exact, atol=1e-10)

    @pytest.mark.parametrize("N", [8, 16, 32])
    def test_diff_trigonometric(self, chebyshev_basis, N):
        """Spectral accuracy for smooth functions."""
        nodes = chebyshev_basis.nodes(N + 1)
        D = chebyshev_basis.diff_matrix(nodes)

        # d/dx[sin(pi*x)] = pi*cos(pi*x)
        f = np.sin(np.pi * nodes)
        df_exact = np.pi * np.cos(np.pi * nodes)
        df_spectral = D @ f

        # Error should decrease exponentially with N
        error = np.max(np.abs(df_spectral - df_exact))
        if N >= 16:
            assert error < 1e-8, f"Error {error} too high for N={N}"
        if N >= 32:
            assert error < 1e-12, f"Error {error} too high for N={N}"

    def test_second_derivative(self, chebyshev_basis):
        """Second derivative D^2 is accurate."""
        N = 17
        nodes = chebyshev_basis.nodes(N)
        D = chebyshev_basis.diff_matrix(nodes)
        D2 = D @ D

        # d^2/dx^2[sin(2*pi*x)] = -4*pi^2*sin(2*pi*x)
        f = np.sin(2 * np.pi * nodes)
        d2f_exact = -4 * np.pi**2 * np.sin(2 * np.pi * nodes)
        d2f_spectral = D2 @ f

        assert np.allclose(d2f_spectral, d2f_exact, atol=1e-6)


class TestKroneckerProducts:
    """Tests for 2D differentiation via Kronecker products."""

    def test_2d_gradient(self, chebyshev_basis):
        """2D gradient using Kronecker products."""
        N = 8
        nodes = chebyshev_basis.nodes(N + 1)
        D = chebyshev_basis.diff_matrix(nodes)

        # Build 2D operators
        I = np.eye(N + 1)
        Dx = np.kron(D, I)  # d/dx acts on first index
        Dy = np.kron(I, D)  # d/dy acts on second index

        # Create 2D grid
        X, Y = np.meshgrid(nodes, nodes, indexing="ij")

        # f(x,y) = x^2 + y^2
        f_2d = X**2 + Y**2
        f_flat = f_2d.ravel()

        df_dx_exact = 2 * X
        df_dy_exact = 2 * Y

        df_dx = (Dx @ f_flat).reshape(X.shape)
        df_dy = (Dy @ f_flat).reshape(Y.shape)

        assert np.allclose(df_dx, df_dx_exact, atol=1e-10)
        assert np.allclose(df_dy, df_dy_exact, atol=1e-10)

    def test_2d_laplacian(self, chebyshev_basis):
        """2D Laplacian using Kronecker products."""
        N = 8
        nodes = chebyshev_basis.nodes(N + 1)
        D = chebyshev_basis.diff_matrix(nodes)
        D2 = D @ D

        # Build 2D Laplacian
        I = np.eye(N + 1)
        Dxx = np.kron(D2, I)
        Dyy = np.kron(I, D2)
        Laplacian = Dxx + Dyy

        # Create 2D grid
        X, Y = np.meshgrid(nodes, nodes, indexing="ij")

        # f(x,y) = x^2 + y^2 => nabla^2 f = 4
        f_2d = X**2 + Y**2
        lap_exact = 4.0

        lap = (Laplacian @ f_2d.ravel()).reshape(X.shape)

        assert np.allclose(lap, lap_exact, atol=1e-10)
