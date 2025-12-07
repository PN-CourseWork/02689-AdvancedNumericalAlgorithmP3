"""Tests for Single Grid (SG) spectral solver."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from scipy.interpolate import RectBivariateSpline

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


class TestGhiaValidation:
    """Validation tests against Ghia et al. (1982) benchmark data."""

    @pytest.fixture
    def ghia_data(self):
        """Load Ghia benchmark data."""
        data_dir = Path(__file__).parent.parent / "data" / "validation" / "ghia"
        ghia_u = pd.read_csv(data_dir / "ghia_Re100_u_centerline.csv")
        ghia_v = pd.read_csv(data_dir / "ghia_Re100_v_centerline.csv")
        return ghia_u, ghia_v

    @pytest.mark.slow
    def test_ghia_u_velocity(self, medium_grid_params, ghia_data):
        """U-velocity matches Ghia at vertical centerline."""
        ghia_u, _ = ghia_data

        solver = SGSolver(**medium_grid_params)
        solver.solve()

        assert solver.metrics.converged, "Solver did not converge"

        # Get solution
        u_2d = solver.arrays.u.reshape(solver.shape_full)
        nodes_x = solver.x_full[:, 0]
        nodes_y = solver.y_full[0, :]

        # Interpolate to exact x=0.5 centerline at Ghia y-points
        u_interp = RectBivariateSpline(nodes_x, nodes_y, u_2d)
        y_ghia = ghia_u["y"].values
        u_at_x05 = u_interp(0.5, y_ghia, grid=False)

        # Compare with Ghia
        u_ghia = ghia_u["u"].values
        max_error = np.max(np.abs(u_at_x05 - u_ghia))
        rms_error = np.sqrt(np.mean((u_at_x05 - u_ghia) ** 2))

        # Accept if max error < 1% and RMS error < 0.5%
        assert max_error < 0.01, f"U max error {max_error} exceeds 1%"
        assert rms_error < 0.005, f"U RMS error {rms_error} exceeds 0.5%"

    @pytest.mark.slow
    def test_ghia_v_velocity(self, medium_grid_params, ghia_data):
        """V-velocity matches Ghia at horizontal centerline."""
        _, ghia_v = ghia_data

        solver = SGSolver(**medium_grid_params)
        solver.solve()

        assert solver.metrics.converged, "Solver did not converge"

        # Get solution
        v_2d = solver.arrays.v.reshape(solver.shape_full)
        nodes_x = solver.x_full[:, 0]
        nodes_y = solver.y_full[0, :]

        # Interpolate to exact y=0.5 centerline at Ghia x-points
        v_interp = RectBivariateSpline(nodes_x, nodes_y, v_2d)
        x_ghia = ghia_v["x"].values
        v_at_y05 = v_interp(x_ghia, 0.5, grid=False)

        # Compare with Ghia
        v_ghia = ghia_v["v"].values
        max_error = np.max(np.abs(v_at_y05 - v_ghia))
        rms_error = np.sqrt(np.mean((v_at_y05 - v_ghia) ** 2))

        # Accept if max error < 5% and RMS error < 2%
        # (V-velocity is harder to match due to grid resolution)
        assert max_error < 0.05, f"V max error {max_error} exceeds 5%"
        assert rms_error < 0.02, f"V RMS error {rms_error} exceeds 2%"

    @pytest.mark.slow
    def test_ghia_extreme_values(self, medium_grid_params, ghia_data):
        """Extreme velocity values match Ghia within tolerance."""
        ghia_u, ghia_v = ghia_data

        solver = SGSolver(**medium_grid_params)
        solver.solve()

        assert solver.metrics.converged

        u_2d = solver.arrays.u.reshape(solver.shape_full)
        v_2d = solver.arrays.v.reshape(solver.shape_full)
        nodes_x = solver.x_full[:, 0]
        nodes_y = solver.y_full[0, :]

        u_interp = RectBivariateSpline(nodes_x, nodes_y, u_2d)
        v_interp = RectBivariateSpline(nodes_x, nodes_y, v_2d)

        # Get centerline values
        y_ghia = ghia_u["y"].values
        x_ghia = ghia_v["x"].values
        u_at_x05 = u_interp(0.5, y_ghia, grid=False)
        v_at_y05 = v_interp(x_ghia, 0.5, grid=False)

        # Check extreme values (Ghia Table I for Re=100)
        u_min_ghia = ghia_u["u"].min()  # ~ -0.2109
        v_min_ghia = ghia_v["v"].min()  # ~ -0.2453
        v_max_ghia = ghia_v["v"].max()  # ~ 0.1753

        u_min_ours = u_at_x05.min()
        v_min_ours = v_at_y05.min()
        v_max_ours = v_at_y05.max()

        # Allow 5% relative error on extreme values
        assert abs(u_min_ours - u_min_ghia) / abs(u_min_ghia) < 0.05
        assert abs(v_min_ours - v_min_ghia) / abs(v_min_ghia) < 0.05
        assert abs(v_max_ours - v_max_ghia) / abs(v_max_ghia) < 0.05
