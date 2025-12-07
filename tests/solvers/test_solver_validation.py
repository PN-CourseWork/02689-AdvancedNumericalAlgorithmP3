"""Validation tests for all spectral solvers (SG, FSG, VMG).

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
from solvers.spectral.vmg import VMGSolver


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
def vmg_params(multigrid_params):
    """Parameters for VMG solver."""
    return {
        **multigrid_params,
        "pre_smoothing": [3, 2, 1],
        "post_smoothing": None,
        "correction_damping": 1.0,
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

    @pytest.mark.slow
    @pytest.mark.vmg
    def test_vmg_converges(self, vmg_params):
        """VMG solver converges."""
        solver = VMGSolver(**vmg_params)
        solver.solve()
        assert solver.metrics.converged, "VMG solver did not converge"


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

    @pytest.mark.slow
    @pytest.mark.vmg
    def test_vmg_ghia_validation(self, vmg_params, ghia_data):
        """VMG solver matches Ghia benchmark."""
        ghia_u, ghia_v = ghia_data

        solver = VMGSolver(**vmg_params)
        solver.solve()
        assert solver.metrics.converged

        errors = compute_ghia_errors(solver, ghia_u, ghia_v)

        assert errors["u_max"] < self.U_MAX_TOL, f"VMG u_max error {errors['u_max']:.4f} > {self.U_MAX_TOL}"
        assert errors["u_rms"] < self.U_RMS_TOL, f"VMG u_rms error {errors['u_rms']:.4f} > {self.U_RMS_TOL}"
        assert errors["v_max"] < self.V_MAX_TOL, f"VMG v_max error {errors['v_max']:.4f} > {self.V_MAX_TOL}"
        assert errors["v_rms"] < self.V_RMS_TOL, f"VMG v_rms error {errors['v_rms']:.4f} > {self.V_RMS_TOL}"


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

    @pytest.mark.slow
    @pytest.mark.vmg
    def test_all_solvers_same_solution(self, validation_params, fsg_params, vmg_params, ghia_data):
        """All solvers converge to the same solution within tolerance."""
        ghia_u, ghia_v = ghia_data

        # Solve with all three solvers
        sg_solver = SGSolver(**validation_params)
        sg_solver.solve()
        assert sg_solver.metrics.converged, "SG did not converge"

        fsg_solver = FSGSolver(**fsg_params)
        fsg_solver.solve()
        assert fsg_solver.metrics.converged, "FSG did not converge"

        vmg_solver = VMGSolver(**vmg_params)
        vmg_solver.solve()
        assert vmg_solver.metrics.converged, "VMG did not converge"

        # Extract centerline velocities for each solver
        sg_u, sg_v, _, _ = extract_centerline_velocities(sg_solver, ghia_u, ghia_v)
        fsg_u, fsg_v, _, _ = extract_centerline_velocities(fsg_solver, ghia_u, ghia_v)
        vmg_u, vmg_v, _, _ = extract_centerline_velocities(vmg_solver, ghia_u, ghia_v)

        # All solvers should produce same solution within 1% tolerance
        solution_tol = 0.01

        # Compare SG vs FSG
        sg_fsg_u_diff = np.max(np.abs(sg_u - fsg_u))
        sg_fsg_v_diff = np.max(np.abs(sg_v - fsg_v))
        assert sg_fsg_u_diff < solution_tol, f"SG-FSG u difference {sg_fsg_u_diff:.6f} > {solution_tol}"
        assert sg_fsg_v_diff < solution_tol, f"SG-FSG v difference {sg_fsg_v_diff:.6f} > {solution_tol}"

        # Compare SG vs VMG
        sg_vmg_u_diff = np.max(np.abs(sg_u - vmg_u))
        sg_vmg_v_diff = np.max(np.abs(sg_v - vmg_v))
        assert sg_vmg_u_diff < solution_tol, f"SG-VMG u difference {sg_vmg_u_diff:.6f} > {solution_tol}"
        assert sg_vmg_v_diff < solution_tol, f"SG-VMG v difference {sg_vmg_v_diff:.6f} > {solution_tol}"

        # Compare FSG vs VMG
        fsg_vmg_u_diff = np.max(np.abs(fsg_u - vmg_u))
        fsg_vmg_v_diff = np.max(np.abs(fsg_v - vmg_v))
        assert fsg_vmg_u_diff < solution_tol, f"FSG-VMG u difference {fsg_vmg_u_diff:.6f} > {solution_tol}"
        assert fsg_vmg_v_diff < solution_tol, f"FSG-VMG v difference {fsg_vmg_v_diff:.6f} > {solution_tol}"

    @pytest.mark.slow
    @pytest.mark.vmg
    def test_solvers_same_extreme_values(self, validation_params, fsg_params, vmg_params, ghia_data):
        """All solvers produce same extreme velocity values."""
        ghia_u, ghia_v = ghia_data

        # Solve with all three solvers
        solvers = {}

        sg_solver = SGSolver(**validation_params)
        sg_solver.solve()
        assert sg_solver.metrics.converged
        solvers["SG"] = sg_solver

        fsg_solver = FSGSolver(**fsg_params)
        fsg_solver.solve()
        assert fsg_solver.metrics.converged
        solvers["FSG"] = fsg_solver

        vmg_solver = VMGSolver(**vmg_params)
        vmg_solver.solve()
        assert vmg_solver.metrics.converged
        solvers["VMG"] = vmg_solver

        # Extract extreme values from each solver
        extremes = {}
        for name, solver in solvers.items():
            u_at_x05, v_at_y05, _, _ = extract_centerline_velocities(solver, ghia_u, ghia_v)
            extremes[name] = {
                "u_min": u_at_x05.min(),
                "v_min": v_at_y05.min(),
                "v_max": v_at_y05.max(),
            }

        # All solvers should have same extreme values within 1%
        rel_tol = 0.01

        for key in ["u_min", "v_min", "v_max"]:
            sg_val = extremes["SG"][key]
            fsg_val = extremes["FSG"][key]
            vmg_val = extremes["VMG"][key]

            # Check relative differences
            sg_fsg_diff = abs(sg_val - fsg_val) / abs(sg_val) if sg_val != 0 else abs(sg_val - fsg_val)
            sg_vmg_diff = abs(sg_val - vmg_val) / abs(sg_val) if sg_val != 0 else abs(sg_val - vmg_val)

            assert sg_fsg_diff < rel_tol, f"SG-FSG {key} relative diff {sg_fsg_diff:.4f} > {rel_tol}"
            assert sg_vmg_diff < rel_tol, f"SG-VMG {key} relative diff {sg_vmg_diff:.4f} > {rel_tol}"
