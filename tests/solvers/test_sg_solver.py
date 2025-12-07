"""Tests for Single Grid (SG) spectral solver.

Unit tests for solver initialization, stepping, and convergence.
Ghia validation tests are in test_solver_validation.py.
"""

import numpy as np
import pytest

from solvers.spectral.sg import SGSolver


class TestSGSolverInitialization:
    """Tests for solver initialization."""

    def test_solver_creates(self, small_grid_params):
        """Solver initializes without error."""
        solver = SGSolver(**small_grid_params)
        assert solver is not None

    def test_grid_shapes(self, small_grid_params):
        """Grids have correct shapes."""
        solver = SGSolver(**small_grid_params)
        N = small_grid_params["nx"]

        assert solver.shape_full == (N + 1, N + 1)
        assert solver.shape_inner == (N - 1, N - 1)

    def test_interpolation_matrices_exist(self, small_grid_params):
        """Interpolation matrices are created."""
        solver = SGSolver(**small_grid_params)

        assert hasattr(solver, "Interp_x")
        assert hasattr(solver, "Interp_y")

        N = small_grid_params["nx"]
        # Interp maps from inner (N-1) to full (N+1) grid
        assert solver.Interp_x.shape == (N + 1, N - 1)
        assert solver.Interp_y.shape == (N + 1, N - 1)


class TestSGSolverStep:
    """Tests for single solver step."""

    def test_step_runs(self, small_grid_params):
        """Single step executes without error."""
        solver = SGSolver(**small_grid_params)
        u, v, p = solver.step()

        assert u is not None
        assert v is not None
        assert p is not None

    def test_boundary_conditions_after_step(self, small_grid_params):
        """Boundary conditions are enforced after step."""
        solver = SGSolver(**small_grid_params)
        solver.step()

        u_2d = solver.arrays.u.reshape(solver.shape_full)
        v_2d = solver.arrays.v.reshape(solver.shape_full)

        # Wall BCs: u=v=0 on left, right, bottom walls
        assert np.allclose(u_2d[0, :], 0, atol=1e-14)  # left wall
        assert np.allclose(u_2d[-1, :], 0, atol=1e-14)  # right wall
        assert np.allclose(u_2d[:, 0], 0, atol=1e-14)  # bottom wall

        assert np.allclose(v_2d[0, :], 0, atol=1e-14)  # left wall
        assert np.allclose(v_2d[-1, :], 0, atol=1e-14)  # right wall
        assert np.allclose(v_2d[:, 0], 0, atol=1e-14)  # bottom wall
        assert np.allclose(v_2d[:, -1], 0, atol=1e-14)  # lid (v=0)


class TestSGSolverConvergence:
    """Tests for solver convergence."""

    def test_converges_on_small_grid(self, small_grid_params):
        """Solver converges on small grid."""
        # Use lower tolerance for faster test
        params = small_grid_params.copy()
        params["tolerance"] = 1e-3  # Relaxed tolerance for 8x8 grid
        params["max_iterations"] = 20000

        solver = SGSolver(**params)
        solver.solve()

        assert solver.metrics.converged

    def test_residual_decreases(self, small_grid_params):
        """Residual decreases monotonically (mostly)."""
        params = small_grid_params.copy()
        params["max_iterations"] = 500

        solver = SGSolver(**params)
        solver.solve()

        residuals = solver.time_series.rel_iter_residual

        # Check that residual generally decreases (allow some fluctuation)
        initial_residual = residuals[10]  # Skip first few iterations
        final_residual = residuals[-1]
        assert final_residual < initial_residual
