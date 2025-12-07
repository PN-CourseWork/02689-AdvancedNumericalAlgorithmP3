#!/usr/bin/env python
"""Test VMG with reduced CFL to fix instability.

The diagnostic showed CFL=2.5 causes oscillation on N=12 coarse grid.
Paper says for N < 12, use CFL=0.3-0.5.

Solution: Use level-specific CFL (larger on fine, smaller on coarse).
"""

import logging
import numpy as np
import sys

logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, "src")

from solvers.spectral.basis.spectral import ChebyshevLobattoBasis
from solvers.spectral.multigrid.fsg import (
    build_hierarchy,
    MultigridSmoother,
    restrict_solution,
    restrict_residual,
)
from solvers.spectral.operators.transfer_operators import create_transfer_operators
from solvers.spectral.operators.corner import create_corner_treatment


def compute_ns_residual_norm(smoother, level):
    """Compute the actual NS equation residual norm."""
    old_tau = (smoother.tau_u, smoother.tau_v, smoother.tau_p)
    smoother.clear_tau_correction()
    smoother._compute_residuals(level.u, level.v, level.p)
    res_norm = np.sqrt(np.sum(level.R_u**2) + np.sum(level.R_v**2) + np.sum(level.R_p**2))
    if old_tau[0] is not None:
        smoother.set_tau_correction(*old_tau)
    return res_norm


def test_coarse_grid_stability():
    """Test coarse grid stability with different CFL values."""
    print("=" * 70)
    print("Test: Coarse grid (N=12) stability with different CFL")
    print("=" * 70)

    basis = ChebyshevLobattoBasis(domain=(0.0, 1.0))
    corner = create_corner_treatment("smoothing")

    from solvers.spectral.multigrid.fsg import build_spectral_level

    level = build_spectral_level(12, 0, basis, basis)

    for cfl in [2.5, 1.5, 1.0, 0.8, 0.5]:
        params = dict(
            Re=100.0, beta_squared=5.0, lid_velocity=1.0, CFL=cfl,
            corner_treatment=corner, Lx=1.0, Ly=1.0
        )
        sm = MultigridSmoother(level=level, **params)
        sm.initialize_lid()

        # Track residual convergence
        res_history = []
        for i in range(100):
            sm._compute_residuals(level.u, level.v, level.p)
            R = np.sqrt(np.sum(level.R_u**2) + np.sum(level.R_v**2))
            res_history.append(R)
            sm.step()

        # Check if monotonically decreasing
        oscillating = False
        for i in range(1, len(res_history)):
            if res_history[i] > res_history[i-1] * 1.1:  # 10% increase = oscillation
                oscillating = True
                break

        status = "OSCILLATING" if oscillating else "STABLE"
        print(f"CFL={cfl}: Final R={res_history[-1]:.2e}, {status}")


def vmg_vcycle_with_level_cfl(
    levels, smoothers, transfer_ops,
    n_presmooth_fine, n_presmooth_coarse,
    damping, fine, coarse, fine_sm, coarse_sm
):
    """Perform one V-cycle with different CFL per level."""
    # Get NS residual before
    R_before = compute_ns_residual_norm(fine_sm, fine)

    # Pre-smooth fine
    for _ in range(n_presmooth_fine):
        fine_sm.step()

    # Save fine state
    u_fine_old = fine.u.copy()
    v_fine_old = fine.v.copy()
    p_fine_old = fine.p.copy()

    # Compute fine residual
    fine_sm._compute_residuals(fine.u, fine.v, fine.p)

    # Restrict solution
    restrict_solution(fine, coarse, transfer_ops)
    u_coarse_old = coarse.u.copy()
    v_coarse_old = coarse.v.copy()
    p_coarse_old = coarse.p.copy()

    # Restrict residual
    restrict_residual(fine, coarse, transfer_ops)
    I_R_u = coarse.R_u.copy()
    I_R_v = coarse.R_v.copy()
    I_R_p = coarse.R_p.copy()

    # Compute tau
    coarse_sm._compute_residuals(coarse.u, coarse.v, coarse.p)
    tau_u = I_R_u - coarse.R_u
    tau_v = I_R_v - coarse.R_v
    tau_p = I_R_p - coarse.R_p

    # Zero tau boundaries
    tau_u_2d = tau_u.reshape(coarse.shape_full)
    tau_v_2d = tau_v.reshape(coarse.shape_full)
    tau_u_2d[0, :] = tau_u_2d[-1, :] = tau_u_2d[:, 0] = tau_u_2d[:, -1] = 0
    tau_v_2d[0, :] = tau_v_2d[-1, :] = tau_v_2d[:, 0] = tau_v_2d[:, -1] = 0

    # Coarse solve with tau
    coarse_sm.set_tau_correction(tau_u, tau_v, tau_p)
    for _ in range(n_presmooth_coarse):
        coarse_sm.step()
    coarse_sm.clear_tau_correction()

    # Correction
    delta_u = coarse.u - u_coarse_old
    delta_v = coarse.v - v_coarse_old
    delta_p = coarse.p - p_coarse_old

    # Zero boundaries
    delta_u_2d = delta_u.reshape(coarse.shape_full).copy()
    delta_v_2d = delta_v.reshape(coarse.shape_full).copy()
    delta_u_2d[0, :] = delta_u_2d[-1, :] = delta_u_2d[:, 0] = delta_u_2d[:, -1] = 0
    delta_v_2d[0, :] = delta_v_2d[-1, :] = delta_v_2d[:, 0] = delta_v_2d[:, -1] = 0

    # Prolongate
    delta_u_fine_2d = transfer_ops.prolongation.prolongate_2d(delta_u_2d, fine.shape_full)
    delta_v_fine_2d = transfer_ops.prolongation.prolongate_2d(delta_v_2d, fine.shape_full)
    delta_p_fine_2d = transfer_ops.prolongation.prolongate_2d(
        delta_p.reshape(coarse.shape_inner), fine.shape_inner
    )

    # Zero boundaries on prolongated correction
    delta_u_fine_2d[0, :] = delta_u_fine_2d[-1, :] = delta_u_fine_2d[:, 0] = delta_u_fine_2d[:, -1] = 0
    delta_v_fine_2d[0, :] = delta_v_fine_2d[-1, :] = delta_v_fine_2d[:, 0] = delta_v_fine_2d[:, -1] = 0

    # Apply correction with damping
    fine.u[:] = u_fine_old + damping * delta_u_fine_2d.ravel()
    fine.v[:] = v_fine_old + damping * delta_v_fine_2d.ravel()
    fine.p[:] = p_fine_old + damping * delta_p_fine_2d.ravel()
    fine_sm._enforce_boundary_conditions(fine.u, fine.v)

    # Get NS residual after
    R_after = compute_ns_residual_norm(fine_sm, fine)

    return R_before, R_after


def test_vmg_with_level_cfl():
    """Test VMG with different CFL on fine vs coarse levels."""
    print("\n" + "=" * 70)
    print("Test: VMG with level-specific CFL")
    print("=" * 70)

    N_fine = 24
    n_levels = 2

    basis = ChebyshevLobattoBasis(domain=(0.0, 1.0))
    transfer_ops = create_transfer_operators("fft", "fft")
    corner = create_corner_treatment("smoothing")

    levels = build_hierarchy(N_fine, n_levels, basis, basis)
    coarse, fine = levels[0], levels[1]

    # Test different CFL combinations
    test_configs = [
        {"cfl_fine": 2.5, "cfl_coarse": 2.5, "damping": 1.0, "name": "Baseline (paper)"},
        {"cfl_fine": 2.5, "cfl_coarse": 1.0, "damping": 1.0, "name": "Reduced coarse CFL"},
        {"cfl_fine": 2.5, "cfl_coarse": 0.5, "damping": 1.0, "name": "Conservative coarse CFL"},
        {"cfl_fine": 2.5, "cfl_coarse": 0.5, "damping": 0.5, "name": "Conservative + damping"},
        {"cfl_fine": 1.0, "cfl_coarse": 0.5, "damping": 0.5, "name": "Both reduced"},
    ]

    for config in test_configs:
        print(f"\n--- {config['name']} ---")
        print(f"    CFL: fine={config['cfl_fine']}, coarse={config['cfl_coarse']}, damping={config['damping']}")

        # Reset levels
        fine.u[:] = 0
        fine.v[:] = 0
        fine.p[:] = 0
        coarse.u[:] = 0
        coarse.v[:] = 0
        coarse.p[:] = 0

        fine_sm = MultigridSmoother(
            level=fine, Re=100.0, beta_squared=5.0, lid_velocity=1.0,
            CFL=config["cfl_fine"], corner_treatment=corner, Lx=1.0, Ly=1.0
        )
        coarse_sm = MultigridSmoother(
            level=coarse, Re=100.0, beta_squared=5.0, lid_velocity=1.0,
            CFL=config["cfl_coarse"], corner_treatment=corner, Lx=1.0, Ly=1.0
        )

        fine_sm.initialize_lid()
        coarse_sm.initialize_lid()

        n_presmooth_fine = 1
        n_presmooth_coarse = 5  # More coarse smoothing since CFL is lower
        damping = config["damping"]

        converged = False
        oscillating = False
        prev_R = None

        for cycle in range(50):
            R_before, R_after = vmg_vcycle_with_level_cfl(
                levels, None, transfer_ops,
                n_presmooth_fine, n_presmooth_coarse,
                damping, fine, coarse, fine_sm, coarse_sm
            )

            ratio = R_after / R_before
            if cycle < 5 or cycle % 5 == 0:
                status = "✓" if ratio < 1.0 else "✗"
                print(f"    Cycle {cycle+1:2d}: R={R_after:.2e}, ratio={ratio:.3f} {status}")

            # Check convergence
            if R_after < 1e-3:
                print(f"    CONVERGED at cycle {cycle+1}!")
                converged = True
                break

            # Check oscillation (3 consecutive increases)
            if prev_R is not None and R_after > prev_R * 1.5:
                oscillating = True
                print(f"    Oscillating at cycle {cycle+1}")
                break

            prev_R = R_after

        if not converged and not oscillating:
            print(f"    Final R={R_after:.2e} after 50 cycles")


def test_vmg_paper_config_fixed():
    """Test VMG with paper config but fixed CFL for coarse level."""
    print("\n" + "=" * 70)
    print("Test: VMG with paper config (48->24->12) and fixed CFL")
    print("=" * 70)

    N_fine = 48
    n_levels = 3

    basis = ChebyshevLobattoBasis(domain=(0.0, 1.0))
    transfer_ops = create_transfer_operators("fft", "fft")
    corner = create_corner_treatment("smoothing")

    levels = build_hierarchy(N_fine, n_levels, basis, basis)

    print(f"Grid sizes: {[lvl.n for lvl in levels]}")

    # Level-specific CFL: larger on fine, smaller on coarse
    cfl_per_level = [0.5, 1.0, 2.0]  # coarsest to finest
    smoothing_per_level = [10, 5, 2]  # more smoothing on coarse

    smoothers = []
    for idx, level in enumerate(levels):
        sm = MultigridSmoother(
            level=level, Re=100.0, beta_squared=5.0, lid_velocity=1.0,
            CFL=cfl_per_level[idx], corner_treatment=corner, Lx=1.0, Ly=1.0
        )
        sm.initialize_lid()
        smoothers.append(sm)

    finest = levels[-1]
    finest_sm = smoothers[-1]

    damping = 0.5

    print(f"\nCFL per level: {cfl_per_level}")
    print(f"Smoothing per level: {smoothing_per_level}")
    print(f"Damping: {damping}")

    def vcycle_3level():
        """3-level V-cycle."""
        l2, l1, l0 = levels[2], levels[1], levels[0]
        sm2, sm1, sm0 = smoothers[2], smoothers[1], smoothers[0]

        # Pre-smooth finest
        for _ in range(smoothing_per_level[2]):
            sm2.step()

        # Compute residual
        sm2._compute_residuals(l2.u, l2.v, l2.p)
        u2_old, v2_old, p2_old = l2.u.copy(), l2.v.copy(), l2.p.copy()

        # Restrict to l1
        restrict_solution(l2, l1, transfer_ops)
        u1_old, v1_old, p1_old = l1.u.copy(), l1.v.copy(), l1.p.copy()
        restrict_residual(l2, l1, transfer_ops)
        I_R_u1, I_R_v1, I_R_p1 = l1.R_u.copy(), l1.R_v.copy(), l1.R_p.copy()

        # Tau for l1
        sm1._compute_residuals(l1.u, l1.v, l1.p)
        tau_u1 = I_R_u1 - l1.R_u
        tau_v1 = I_R_v1 - l1.R_v
        tau_p1 = I_R_p1 - l1.R_p
        tau_u1_2d = tau_u1.reshape(l1.shape_full)
        tau_v1_2d = tau_v1.reshape(l1.shape_full)
        tau_u1_2d[0, :] = tau_u1_2d[-1, :] = tau_u1_2d[:, 0] = tau_u1_2d[:, -1] = 0
        tau_v1_2d[0, :] = tau_v1_2d[-1, :] = tau_v1_2d[:, 0] = tau_v1_2d[:, -1] = 0

        # Pre-smooth l1 with tau
        sm1.set_tau_correction(tau_u1, tau_v1, tau_p1)
        for _ in range(smoothing_per_level[1]):
            sm1.step()

        # Compute residual on l1 (still with tau)
        sm1._compute_residuals(l1.u, l1.v, l1.p)

        # Restrict to l0 (coarsest)
        restrict_solution(l1, l0, transfer_ops)
        u0_old, v0_old, p0_old = l0.u.copy(), l0.v.copy(), l0.p.copy()
        restrict_residual(l1, l0, transfer_ops)
        I_R_u0, I_R_v0, I_R_p0 = l0.R_u.copy(), l0.R_v.copy(), l0.R_p.copy()

        # Tau for l0
        sm0._compute_residuals(l0.u, l0.v, l0.p)
        tau_u0 = I_R_u0 - l0.R_u
        tau_v0 = I_R_v0 - l0.R_v
        tau_p0 = I_R_p0 - l0.R_p
        tau_u0_2d = tau_u0.reshape(l0.shape_full)
        tau_v0_2d = tau_v0.reshape(l0.shape_full)
        tau_u0_2d[0, :] = tau_u0_2d[-1, :] = tau_u0_2d[:, 0] = tau_u0_2d[:, -1] = 0
        tau_v0_2d[0, :] = tau_v0_2d[-1, :] = tau_v0_2d[:, 0] = tau_v0_2d[:, -1] = 0

        # Solve coarsest with tau
        sm0.set_tau_correction(tau_u0, tau_v0, tau_p0)
        for _ in range(smoothing_per_level[0]):
            sm0.step()
        sm0.clear_tau_correction()

        # Correct l1 from l0
        delta_u0 = l0.u - u0_old
        delta_v0 = l0.v - v0_old
        delta_p0 = l0.p - p0_old
        delta_u0_2d = delta_u0.reshape(l0.shape_full).copy()
        delta_v0_2d = delta_v0.reshape(l0.shape_full).copy()
        delta_u0_2d[0, :] = delta_u0_2d[-1, :] = delta_u0_2d[:, 0] = delta_u0_2d[:, -1] = 0
        delta_v0_2d[0, :] = delta_v0_2d[-1, :] = delta_v0_2d[:, 0] = delta_v0_2d[:, -1] = 0

        delta_u1_2d = transfer_ops.prolongation.prolongate_2d(delta_u0_2d, l1.shape_full)
        delta_v1_2d = transfer_ops.prolongation.prolongate_2d(delta_v0_2d, l1.shape_full)
        delta_p1_2d = transfer_ops.prolongation.prolongate_2d(
            delta_p0.reshape(l0.shape_inner), l1.shape_inner
        )
        delta_u1_2d[0, :] = delta_u1_2d[-1, :] = delta_u1_2d[:, 0] = delta_u1_2d[:, -1] = 0
        delta_v1_2d[0, :] = delta_v1_2d[-1, :] = delta_v1_2d[:, 0] = delta_v1_2d[:, -1] = 0

        l1.u[:] = u1_old + damping * delta_u1_2d.ravel()
        l1.v[:] = v1_old + damping * delta_v1_2d.ravel()
        l1.p[:] = p1_old + damping * delta_p1_2d.ravel()
        sm1._enforce_boundary_conditions(l1.u, l1.v)
        sm1.clear_tau_correction()

        # Correct l2 from l1
        delta_u1 = l1.u - u1_old
        delta_v1 = l1.v - v1_old
        delta_p1 = l1.p - p1_old
        delta_u1_2d = delta_u1.reshape(l1.shape_full).copy()
        delta_v1_2d = delta_v1.reshape(l1.shape_full).copy()
        delta_u1_2d[0, :] = delta_u1_2d[-1, :] = delta_u1_2d[:, 0] = delta_u1_2d[:, -1] = 0
        delta_v1_2d[0, :] = delta_v1_2d[-1, :] = delta_v1_2d[:, 0] = delta_v1_2d[:, -1] = 0

        delta_u2_2d = transfer_ops.prolongation.prolongate_2d(delta_u1_2d, l2.shape_full)
        delta_v2_2d = transfer_ops.prolongation.prolongate_2d(delta_v1_2d, l2.shape_full)
        delta_p2_2d = transfer_ops.prolongation.prolongate_2d(
            delta_p1.reshape(l1.shape_inner), l2.shape_inner
        )
        delta_u2_2d[0, :] = delta_u2_2d[-1, :] = delta_u2_2d[:, 0] = delta_u2_2d[:, -1] = 0
        delta_v2_2d[0, :] = delta_v2_2d[-1, :] = delta_v2_2d[:, 0] = delta_v2_2d[:, -1] = 0

        l2.u[:] = u2_old + damping * delta_u2_2d.ravel()
        l2.v[:] = v2_old + damping * delta_v2_2d.ravel()
        l2.p[:] = p2_old + damping * delta_p2_2d.ravel()
        sm2._enforce_boundary_conditions(l2.u, l2.v)

    print("\n" + "-" * 50)

    for cycle in range(50):
        R_before = compute_ns_residual_norm(finest_sm, finest)
        vcycle_3level()
        R_after = compute_ns_residual_norm(finest_sm, finest)

        ratio = R_after / R_before
        if cycle < 10 or cycle % 5 == 0:
            status = "✓" if ratio < 1.0 else "✗"
            print(f"Cycle {cycle+1:2d}: R={R_after:.2e}, ratio={ratio:.3f} {status}")

        if R_after < 1e-3:
            print(f"CONVERGED at cycle {cycle+1}!")
            break

        if R_after > 1e10:
            print("DIVERGED!")
            break


if __name__ == "__main__":
    test_coarse_grid_stability()
    test_vmg_with_level_cfl()
    test_vmg_paper_config_fixed()
