"""Validation tests for spectral solvers (SG, FSG).

This module tests that:
1. All solvers converge for the lid-driven cavity problem
2. All solvers produce the same solution (within tolerance)
3. All solvers match Ghia et al. (1982) benchmark data
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from scipy.interpolate import RectBivariateSpline

from solvers.spectral.sg import SGSolver
from solvers.spectral.fsg import FSGSolver


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def validation_params():
    """Parameters for validation tests - medium grid that balances speed and accuracy."""
    return {
        "Re": 100,
        "nx": 15,
        "ny": 15,
        "tolerance": 1e-5,
        "max_iterations": 100000,
        "lid_velocity": 1.0,
        "Lx": 1.0,
        "Ly": 1.0,
        "CFL": 0.5,
        "beta_squared": 5.0,
        "basis_type": "chebyshev",
        "corner_treatment": "smoothing",
    }


@pytest.fixture
def multigrid_params(validation_params):
    """Additional parameters for multigrid solvers."""
    return {
        **validation_params,
        "n_levels": 3,
        "prolongation_method": "fft",
        "restriction_method": "fft",
    }


@pytest.fixture
def fsg_params(multigrid_params):
    """Parameters for FSG solver."""
    return {
        **multigrid_params,
        "coarse_tolerance_factor": 10.0,
    }


@pytest.fixture
def ghia_data():
    """Load Ghia benchmark data for Re=100."""
    data_dir = Path(__file__).parent.parent.parent / "data" / "validation" / "ghia"
    ghia_u = pd.read_csv(data_dir / "ghia_Re100_u_centerline.csv")
    ghia_v = pd.read_csv(data_dir / "ghia_Re100_v_centerline.csv")
    return ghia_u, ghia_v


# ============================================================================
# Helper functions
# ============================================================================

def extract_centerline_velocities(solver, ghia_u, ghia_v):
    """Extract u and v velocities at centerlines, interpolated to Ghia points."""
    u_2d = solver.arrays.u.reshape(solver.shape_full)
    v_2d = solver.arrays.v.reshape(solver.shape_full)
    nodes_x = solver.x_full[:, 0]
    nodes_y = solver.y_full[0, :]

    # Interpolate to Ghia measurement points
    u_interp = RectBivariateSpline(nodes_x, nodes_y, u_2d)
    v_interp = RectBivariateSpline(nodes_x, nodes_y, v_2d)

    y_ghia = ghia_u["y"].values
    x_ghia = ghia_v["x"].values

    u_at_x05 = u_interp(0.5, y_ghia, grid=False)
    v_at_y05 = v_interp(x_ghia, 0.5, grid=False)

    return u_at_x05, v_at_y05, y_ghia, x_ghia


def compute_ghia_errors(solver, ghia_u, ghia_v):
    """Compute errors against Ghia benchmark data."""
    u_at_x05, v_at_y05, _, _ = extract_centerline_velocities(solver, ghia_u, ghia_v)

    u_ghia = ghia_u["u"].values
    v_ghia = ghia_v["v"].values

    u_max_error = np.max(np.abs(u_at_x05 - u_ghia))
    u_rms_error = np.sqrt(np.mean((u_at_x05 - u_ghia) ** 2))

    v_max_error = np.max(np.abs(v_at_y05 - v_ghia))
    v_rms_error = np.sqrt(np.mean((v_at_y05 - v_ghia) ** 2))

    return {
        "u_max": u_max_error,
        "u_rms": u_rms_error,
        "v_max": v_max_error,
        "v_rms": v_rms_error,
    }


# ============================================================================
# Convergence Tests
# ============================================================================

class TestSolverConvergence:
    """Test that each solver converges."""

    @pytest.mark.slow
    def test_sg_converges(self, validation_params):
        """SG solver converges."""
        solver = SGSolver(**validation_params)
        solver.solve()
        assert solver.metrics.converged, "SG solver did not converge"

    @pytest.mark.slow
    def test_fsg_converges(self, fsg_params):
        """FSG solver converges."""
        solver = FSGSolver(**fsg_params)
        solver.solve()
        assert solver.metrics.converged, "FSG solver did not converge"


# ============================================================================
# Ghia Validation Tests
# ============================================================================

class TestGhiaValidation:
    """Validate all solvers against Ghia et al. (1982) benchmark data."""

    # Tolerance thresholds
    U_MAX_TOL = 0.01   # 1% max error for u-velocity
    U_RMS_TOL = 0.005  # 0.5% RMS error for u-velocity
    V_MAX_TOL = 0.05   # 5% max error for v-velocity
    V_RMS_TOL = 0.02   # 2% RMS error for v-velocity

    @pytest.mark.slow
    def test_sg_ghia_validation(self, validation_params, ghia_data):
        """SG solver matches Ghia benchmark."""
        ghia_u, ghia_v = ghia_data

        solver = SGSolver(**validation_params)
        solver.solve()
        assert solver.metrics.converged

        errors = compute_ghia_errors(solver, ghia_u, ghia_v)

        assert errors["u_max"] < self.U_MAX_TOL, f"SG u_max error {errors['u_max']:.4f} > {self.U_MAX_TOL}"
        assert errors["u_rms"] < self.U_RMS_TOL, f"SG u_rms error {errors['u_rms']:.4f} > {self.U_RMS_TOL}"
        assert errors["v_max"] < self.V_MAX_TOL, f"SG v_max error {errors['v_max']:.4f} > {self.V_MAX_TOL}"
        assert errors["v_rms"] < self.V_RMS_TOL, f"SG v_rms error {errors['v_rms']:.4f} > {self.V_RMS_TOL}"

    @pytest.mark.slow
    def test_fsg_ghia_validation(self, fsg_params, ghia_data):
        """FSG solver matches Ghia benchmark."""
        ghia_u, ghia_v = ghia_data

        solver = FSGSolver(**fsg_params)
        solver.solve()
        assert solver.metrics.converged

        errors = compute_ghia_errors(solver, ghia_u, ghia_v)

        assert errors["u_max"] < self.U_MAX_TOL, f"FSG u_max error {errors['u_max']:.4f} > {self.U_MAX_TOL}"
        assert errors["u_rms"] < self.U_RMS_TOL, f"FSG u_rms error {errors['u_rms']:.4f} > {self.U_RMS_TOL}"
        assert errors["v_max"] < self.V_MAX_TOL, f"FSG v_max error {errors['v_max']:.4f} > {self.V_MAX_TOL}"
        assert errors["v_rms"] < self.V_RMS_TOL, f"FSG v_rms error {errors['v_rms']:.4f} > {self.V_RMS_TOL}"


# ============================================================================
# Solution Consistency Tests
# ============================================================================

class TestSolutionConsistency:
    """Test that all solvers produce the same solution."""

    @pytest.mark.slow
    def test_sg_fsg_same_solution(self, validation_params, fsg_params, ghia_data):
        """SG and FSG converge to the same solution within tolerance."""
        ghia_u, ghia_v = ghia_data

        # Solve with SG and FSG
        sg_solver = SGSolver(**validation_params)
        sg_solver.solve()
        assert sg_solver.metrics.converged, "SG did not converge"

        fsg_solver = FSGSolver(**fsg_params)
        fsg_solver.solve()
        assert fsg_solver.metrics.converged, "FSG did not converge"

        # Extract centerline velocities
        sg_u, sg_v, _, _ = extract_centerline_velocities(sg_solver, ghia_u, ghia_v)
        fsg_u, fsg_v, _, _ = extract_centerline_velocities(fsg_solver, ghia_u, ghia_v)

        # Should produce same solution within 1% tolerance
        solution_tol = 0.01

        sg_fsg_u_diff = np.max(np.abs(sg_u - fsg_u))
        sg_fsg_v_diff = np.max(np.abs(sg_v - fsg_v))
        assert sg_fsg_u_diff < solution_tol, f"SG-FSG u difference {sg_fsg_u_diff:.6f} > {solution_tol}"
        assert sg_fsg_v_diff < solution_tol, f"SG-FSG v difference {sg_fsg_v_diff:.6f} > {solution_tol}"
