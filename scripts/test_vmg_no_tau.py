#!/usr/bin/env python
"""Test V-cycle WITHOUT tau correction (simple coarse grid correction).

The FAS tau correction might be causing instability. Let's test without it
to see if the basic V-cycle structure works.
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
    prolongate_solution,
)
from solvers.spectral.operators.transfer_operators import create_transfer_operators
from solvers.spectral.operators.corner import create_corner_treatment


def vcycle_without_tau(fine, coarse, fine_sm, coarse_sm, transfer_ops,
                       n_presmooth_fine, n_presmooth_coarse, damping):
    """V-cycle WITHOUT FAS tau correction (pure CGC)."""

    # Pre-smooth fine
    for _ in range(n_presmooth_fine):
        fine_sm.step()

    # Save fine state
    u_fine_old = fine.u.copy()
    v_fine_old = fine.v.copy()
    p_fine_old = fine.p.copy()

    # Compute fine residual
    fine_sm._compute_residuals(fine.u, fine.v, fine.p)

    # Restrict solution (NOT residual) - coarse starts from restricted fine
    restrict_solution(fine, coarse, transfer_ops)
    u_coarse_old = coarse.u.copy()
    v_coarse_old = coarse.v.copy()
    p_coarse_old = coarse.p.copy()

    # NO TAU - just smooth the coarse grid problem (converge towards coarse solution)
    for _ in range(n_presmooth_coarse):
        coarse_sm.step()

    # Compute correction
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

    # Apply correction
    fine.u[:] = u_fine_old + damping * delta_u_fine_2d.ravel()
    fine.v[:] = v_fine_old + damping * delta_v_fine_2d.ravel()
    fine.p[:] = p_fine_old + damping * delta_p_fine_2d.ravel()
    fine_sm._enforce_boundary_conditions(fine.u, fine.v)


def test_vcycle_no_tau():
    """Test V-cycle without tau correction."""
    print("=" * 70)
    print("Test: V-cycle WITHOUT tau correction (pure CGC)")
    print("=" * 70)

    N_fine = 24
    n_levels = 2

    basis = ChebyshevLobattoBasis(domain=(0.0, 1.0))
    transfer_ops = create_transfer_operators("fft", "fft")
    corner = create_corner_treatment("smoothing")

    levels = build_hierarchy(N_fine, n_levels, basis, basis)
    coarse, fine = levels[0], levels[1]

    print(f"Grid: fine N={fine.n}, coarse N={coarse.n}")

    # Level-specific CFL
    fine_sm = MultigridSmoother(
        level=fine, Re=100.0, beta_squared=5.0, lid_velocity=1.0,
        CFL=2.0, corner_treatment=corner, Lx=1.0, Ly=1.0
    )
    coarse_sm = MultigridSmoother(
        level=coarse, Re=100.0, beta_squared=5.0, lid_velocity=1.0,
        CFL=1.0, corner_treatment=corner, Lx=1.0, Ly=1.0
    )

    fine_sm.initialize_lid()
    coarse_sm.initialize_lid()

    # Config
    n_presmooth_fine = 3
    n_presmooth_coarse = 10
    damping = 0.5

    print(f"\nConfig:")
    print(f"  Pre-smooth: fine={n_presmooth_fine}, coarse={n_presmooth_coarse}")
    print(f"  Damping: {damping}")
    print(f"  NO TAU CORRECTION")

    print("\n" + "-" * 50)

    u_prev = fine.u.copy()
    v_prev = fine.v.copy()

    for cycle in range(50):
        vcycle_without_tau(
            fine, coarse, fine_sm, coarse_sm, transfer_ops,
            n_presmooth_fine, n_presmooth_coarse, damping
        )

        # Convergence check
        u_change = np.linalg.norm(fine.u - u_prev) / (np.linalg.norm(u_prev) + 1e-12)
        v_change = np.linalg.norm(fine.v - v_prev) / (np.linalg.norm(v_prev) + 1e-12)
        u_prev[:] = fine.u
        v_prev[:] = fine.v

        max_res = max(u_change, v_change)
        if cycle < 10 or cycle % 5 == 0:
            status = "✓" if max_res < 1e-5 else ""
            print(f"Cycle {cycle+1:2d}: u_res={u_change:.2e}, v_res={v_change:.2e} {status}")

        if max_res < 1e-5:
            print(f"\nCONVERGED at cycle {cycle+1}!")
            return True

    print(f"\nDid not converge after 50 cycles")
    return False


def test_vcycle_with_tau():
    """Test V-cycle WITH tau correction for comparison."""
    print("\n" + "=" * 70)
    print("Test: V-cycle WITH FAS tau correction")
    print("=" * 70)

    N_fine = 24
    n_levels = 2

    basis = ChebyshevLobattoBasis(domain=(0.0, 1.0))
    transfer_ops = create_transfer_operators("fft", "fft")
    corner = create_corner_treatment("smoothing")

    levels = build_hierarchy(N_fine, n_levels, basis, basis)
    coarse, fine = levels[0], levels[1]

    print(f"Grid: fine N={fine.n}, coarse N={coarse.n}")

    fine_sm = MultigridSmoother(
        level=fine, Re=100.0, beta_squared=5.0, lid_velocity=1.0,
        CFL=2.0, corner_treatment=corner, Lx=1.0, Ly=1.0
    )
    coarse_sm = MultigridSmoother(
        level=coarse, Re=100.0, beta_squared=5.0, lid_velocity=1.0,
        CFL=1.0, corner_treatment=corner, Lx=1.0, Ly=1.0
    )

    fine_sm.initialize_lid()
    coarse_sm.initialize_lid()

    n_presmooth_fine = 3
    n_presmooth_coarse = 10
    damping = 0.5

    print(f"\nConfig:")
    print(f"  Pre-smooth: fine={n_presmooth_fine}, coarse={n_presmooth_coarse}")
    print(f"  Damping: {damping}")
    print(f"  WITH FAS TAU CORRECTION")

    print("\n" + "-" * 50)

    u_prev = fine.u.copy()
    v_prev = fine.v.copy()

    for cycle in range(50):
        # Pre-smooth fine
        for _ in range(n_presmooth_fine):
            fine_sm.step()

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

        # Coarse solve WITH tau
        coarse_sm.set_tau_correction(tau_u, tau_v, tau_p)
        for _ in range(n_presmooth_coarse):
            coarse_sm.step()
        coarse_sm.clear_tau_correction()

        # Correction
        delta_u = coarse.u - u_coarse_old
        delta_v = coarse.v - v_coarse_old
        delta_p = coarse.p - p_coarse_old

        delta_u_2d = delta_u.reshape(coarse.shape_full).copy()
        delta_v_2d = delta_v.reshape(coarse.shape_full).copy()
        delta_u_2d[0, :] = delta_u_2d[-1, :] = delta_u_2d[:, 0] = delta_u_2d[:, -1] = 0
        delta_v_2d[0, :] = delta_v_2d[-1, :] = delta_v_2d[:, 0] = delta_v_2d[:, -1] = 0

        delta_u_fine_2d = transfer_ops.prolongation.prolongate_2d(delta_u_2d, fine.shape_full)
        delta_v_fine_2d = transfer_ops.prolongation.prolongate_2d(delta_v_2d, fine.shape_full)
        delta_p_fine_2d = transfer_ops.prolongation.prolongate_2d(
            delta_p.reshape(coarse.shape_inner), fine.shape_inner
        )

        delta_u_fine_2d[0, :] = delta_u_fine_2d[-1, :] = delta_u_fine_2d[:, 0] = delta_u_fine_2d[:, -1] = 0
        delta_v_fine_2d[0, :] = delta_v_fine_2d[-1, :] = delta_v_fine_2d[:, 0] = delta_v_fine_2d[:, -1] = 0

        fine.u[:] = u_fine_old + damping * delta_u_fine_2d.ravel()
        fine.v[:] = v_fine_old + damping * delta_v_fine_2d.ravel()
        fine.p[:] = p_fine_old + damping * delta_p_fine_2d.ravel()
        fine_sm._enforce_boundary_conditions(fine.u, fine.v)

        # Convergence check
        u_change = np.linalg.norm(fine.u - u_prev) / (np.linalg.norm(u_prev) + 1e-12)
        v_change = np.linalg.norm(fine.v - v_prev) / (np.linalg.norm(v_prev) + 1e-12)
        u_prev[:] = fine.u
        v_prev[:] = fine.v

        max_res = max(u_change, v_change)
        if cycle < 10 or cycle % 5 == 0:
            print(f"Cycle {cycle+1:2d}: u_res={u_change:.2e}, v_res={v_change:.2e}")

        if max_res < 1e-5:
            print(f"\nCONVERGED at cycle {cycle+1}!")
            return True

    print(f"\nDid not converge after 50 cycles")
    return False


if __name__ == "__main__":
    success_no_tau = test_vcycle_no_tau()
    success_with_tau = test_vcycle_with_tau()

    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  V-cycle WITHOUT tau: {'PASS' if success_no_tau else 'FAIL'}")
    print(f"  V-cycle WITH tau:    {'PASS' if success_with_tau else 'FAIL'}")
    print("=" * 70)
