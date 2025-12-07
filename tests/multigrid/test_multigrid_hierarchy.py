"""Tests for multigrid hierarchy and level operations.

Tests the core multigrid infrastructure:
- SpectralLevel creation and structure
- build_hierarchy function
- Solution transfer between levels (prolongate/restrict)
"""

import numpy as np
import pytest

from solvers.spectral.basis.spectral import ChebyshevLobattoBasis
from solvers.spectral.multigrid.fsg import (
    SpectralLevel,
    build_spectral_level,
    build_hierarchy,
    prolongate_solution,
    restrict_solution,
)
from solvers.spectral.operators.transfer_operators import create_transfer_operators


@pytest.fixture
def chebyshev_basis():
    """Chebyshev-Gauss-Lobatto basis on [0, 1]."""
    return ChebyshevLobattoBasis(domain=(0.0, 1.0))


@pytest.fixture
def transfer_ops():
    """Default transfer operators (FFT-based)."""
    return create_transfer_operators(
        prolongation_method="fft",
        restriction_method="fft",
    )


class TestSpectralLevel:
    """Tests for SpectralLevel data structure."""

    def test_build_level_shapes(self, chebyshev_basis):
        """Level has correct grid shapes."""
        n = 8
        level = build_spectral_level(
            n=n,
            level_idx=0,
            basis_x=chebyshev_basis,
            basis_y=chebyshev_basis,
        )

        assert level.n == n
        assert level.shape_full == (n + 1, n + 1)
        assert level.shape_inner == (n - 1, n - 1)
        assert level.n_nodes_full == (n + 1) ** 2
        assert level.n_nodes_inner == (n - 1) ** 2

    def test_build_level_arrays(self, chebyshev_basis):
        """Level has correctly sized arrays."""
        n = 8
        level = build_spectral_level(
            n=n,
            level_idx=0,
            basis_x=chebyshev_basis,
            basis_y=chebyshev_basis,
        )

        # Velocity arrays on full grid
        assert level.u.shape == ((n + 1) ** 2,)
        assert level.v.shape == ((n + 1) ** 2,)

        # Pressure on inner grid
        assert level.p.shape == ((n - 1) ** 2,)

        # Differentiation matrices
        n_full = (n + 1) ** 2
        assert level.Dx.shape == (n_full, n_full)
        assert level.Dy.shape == (n_full, n_full)
        assert level.Laplacian.shape == (n_full, n_full)

    def test_build_level_nodes(self, chebyshev_basis):
        """Level has correct Chebyshev-Lobatto nodes."""
        n = 8
        level = build_spectral_level(
            n=n,
            level_idx=0,
            basis_x=chebyshev_basis,
            basis_y=chebyshev_basis,
            Lx=1.0,
            Ly=1.0,
        )

        # Nodes should be on [0, 1]
        assert level.x_nodes[0] == pytest.approx(0.0)
        assert level.x_nodes[-1] == pytest.approx(1.0)
        assert level.y_nodes[0] == pytest.approx(0.0)
        assert level.y_nodes[-1] == pytest.approx(1.0)

        # Correct number of nodes
        assert len(level.x_nodes) == n + 1
        assert len(level.y_nodes) == n + 1

    def test_build_level_interpolation_matrices(self, chebyshev_basis):
        """Interpolation matrices have correct shapes."""
        n = 8
        level = build_spectral_level(
            n=n,
            level_idx=0,
            basis_x=chebyshev_basis,
            basis_y=chebyshev_basis,
        )

        # Interp maps from inner (n-1) to full (n+1) grid
        assert level.Interp_x.shape == (n + 1, n - 1)
        assert level.Interp_y.shape == (n + 1, n - 1)


class TestBuildHierarchy:
    """Tests for build_hierarchy function."""

    @pytest.mark.parametrize("n_fine,n_levels", [
        (8, 2),
        (16, 3),
        (32, 4),
    ])
    def test_hierarchy_size(self, chebyshev_basis, n_fine, n_levels):
        """Hierarchy has correct number of levels."""
        levels = build_hierarchy(
            n_fine=n_fine,
            n_levels=n_levels,
            basis_x=chebyshev_basis,
            basis_y=chebyshev_basis,
        )

        assert len(levels) == n_levels

    def test_hierarchy_coarsening(self, chebyshev_basis):
        """Levels coarsen by factor of 2."""
        levels = build_hierarchy(
            n_fine=16,
            n_levels=3,
            basis_x=chebyshev_basis,
            basis_y=chebyshev_basis,
        )

        # levels[0] is coarsest, levels[-1] is finest
        assert levels[0].n == 4   # coarsest
        assert levels[1].n == 8   # middle
        assert levels[2].n == 16  # finest

    def test_hierarchy_level_indices(self, chebyshev_basis):
        """Level indices are correctly assigned."""
        levels = build_hierarchy(
            n_fine=16,
            n_levels=3,
            basis_x=chebyshev_basis,
            basis_y=chebyshev_basis,
        )

        for i, level in enumerate(levels):
            assert level.level_idx == i

    def test_hierarchy_nested_grids(self, chebyshev_basis):
        """Coarse grid nodes are subset of fine grid nodes."""
        levels = build_hierarchy(
            n_fine=16,
            n_levels=3,
            basis_x=chebyshev_basis,
            basis_y=chebyshev_basis,
        )

        # Check that coarse nodes are contained in fine nodes
        for i in range(len(levels) - 1):
            coarse = levels[i]
            fine = levels[i + 1]

            # Every other fine node should match coarse nodes
            for xc in coarse.x_nodes:
                # Find closest fine node
                idx = np.argmin(np.abs(fine.x_nodes - xc))
                assert fine.x_nodes[idx] == pytest.approx(xc, abs=1e-12)


class TestSolutionTransfer:
    """Tests for solution transfer between levels."""

    def test_prolongate_polynomial(self, chebyshev_basis, transfer_ops):
        """Prolongation preserves polynomial solutions for velocity."""
        levels = build_hierarchy(
            n_fine=16,
            n_levels=2,
            basis_x=chebyshev_basis,
            basis_y=chebyshev_basis,
        )

        coarse, fine = levels[0], levels[1]

        # Set coarse solution to polynomial: u = x^2 + y^2
        X_c, Y_c = coarse.X, coarse.Y
        coarse.u[:] = (X_c**2 + Y_c**2).ravel()
        coarse.v[:] = (X_c * Y_c).ravel()

        # Prolongate to fine
        prolongate_solution(coarse, fine, transfer_ops)

        # Check fine velocity matches polynomial
        X_f, Y_f = fine.X, fine.Y
        u_exact = (X_f**2 + Y_f**2).ravel()
        v_exact = (X_f * Y_f).ravel()

        assert np.allclose(fine.u, u_exact, atol=1e-10)
        assert np.allclose(fine.v, v_exact, atol=1e-10)

    def test_restrict_polynomial(self, chebyshev_basis, transfer_ops):
        """Restriction preserves polynomial solutions for velocity."""
        levels = build_hierarchy(
            n_fine=16,
            n_levels=2,
            basis_x=chebyshev_basis,
            basis_y=chebyshev_basis,
        )

        coarse, fine = levels[0], levels[1]

        # Set fine solution to polynomial: u = x^2 + y^2
        X_f, Y_f = fine.X, fine.Y
        fine.u[:] = (X_f**2 + Y_f**2).ravel()
        fine.v[:] = (X_f * Y_f).ravel()

        # Restrict to coarse
        restrict_solution(fine, coarse, transfer_ops)

        # Check coarse velocity matches polynomial
        X_c, Y_c = coarse.X, coarse.Y
        u_exact = (X_c**2 + Y_c**2).ravel()
        v_exact = (X_c * Y_c).ravel()

        # FFT restriction may not be exact, use relaxed tolerance
        assert np.allclose(coarse.u, u_exact, atol=1e-8)
        assert np.allclose(coarse.v, v_exact, atol=1e-8)

    def test_roundtrip_preserves_coarse(self, chebyshev_basis, transfer_ops):
        """Prolongate then restrict preserves coarse grid solution."""
        levels = build_hierarchy(
            n_fine=16,
            n_levels=2,
            basis_x=chebyshev_basis,
            basis_y=chebyshev_basis,
        )

        coarse, fine = levels[0], levels[1]

        # Set coarse solution
        X_c, Y_c = coarse.X, coarse.Y
        u_orig = (X_c**2 + Y_c**2).ravel().copy()
        v_orig = (X_c * Y_c).ravel().copy()
        coarse.u[:] = u_orig
        coarse.v[:] = v_orig

        # Prolongate to fine
        prolongate_solution(coarse, fine, transfer_ops)

        # Restrict back to coarse
        restrict_solution(fine, coarse, transfer_ops)

        # Should recover original (approximately)
        assert np.allclose(coarse.u, u_orig, atol=1e-8)
        assert np.allclose(coarse.v, v_orig, atol=1e-8)


class TestDifferentiationMatrices:
    """Tests for differentiation matrices on levels."""

    def test_dx_polynomial(self, chebyshev_basis):
        """Dx correctly differentiates polynomial."""
        n = 8
        level = build_spectral_level(
            n=n,
            level_idx=0,
            basis_x=chebyshev_basis,
            basis_y=chebyshev_basis,
            Lx=1.0,
            Ly=1.0,
        )

        # f = x^2, df/dx = 2x
        X, Y = level.X, level.Y
        f = (X**2).ravel()
        df_dx_exact = (2 * X).ravel()

        df_dx = level.Dx @ f

        assert np.allclose(df_dx, df_dx_exact, atol=1e-10)

    def test_laplacian_polynomial(self, chebyshev_basis):
        """Laplacian correctly computes second derivatives."""
        n = 8
        level = build_spectral_level(
            n=n,
            level_idx=0,
            basis_x=chebyshev_basis,
            basis_y=chebyshev_basis,
            Lx=1.0,
            Ly=1.0,
        )

        # f = x^2 + y^2, ∇²f = 2 + 2 = 4
        X, Y = level.X, level.Y
        f = (X**2 + Y**2).ravel()
        lap_exact = np.full_like(f, 4.0)

        lap_f = level.Laplacian @ f

        assert np.allclose(lap_f, lap_exact, atol=1e-10)
