#!/usr/bin/env python
"""Diagnose why VMG V-cycle diverges after initial improvement.

Key questions to answer:
1. Does the coarse grid solve actually reduce the coarse residual (with tau)?
2. Is the correction magnitude appropriate?
3. Is the residual reduction from coarse correction in the right direction?
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
    smoother._compute_residuals(level.u, level.v, level.p)
    # Clear tau to get pure NS residual
    old_tau_u = smoother.tau_u
    old_tau_v = smoother.tau_v
    old_tau_p = smoother.tau_p
    smoother.clear_tau_correction()
    smoother._compute_residuals(level.u, level.v, level.p)
    res_norm = np.sqrt(np.sum(level.R_u**2) + np.sum(level.R_v**2) + np.sum(level.R_p**2))
    # Restore tau
    if old_tau_u is not None:
        smoother.set_tau_correction(old_tau_u, old_tau_v, old_tau_p)
    return res_norm


def diagnose_single_vcycle():
    """Diagnose one V-cycle in detail."""
    print("=" * 70)
    print("VMG DIAGNOSTIC: Single V-cycle analysis")
    print("=" * 70)

    # Paper-appropriate grid sizes
    N_fine = 24
    n_levels = 2

    basis = ChebyshevLobattoBasis(domain=(0.0, 1.0))
    transfer_ops = create_transfer_operators("fft", "fft")
    corner = create_corner_treatment("smoothing")

    levels = build_hierarchy(N_fine, n_levels, basis, basis)
    coarse, fine = levels[0], levels[1]

    print(f"\nGrid: fine N={fine.n}, coarse N={coarse.n}")

    params = dict(
        Re=100.0, beta_squared=5.0, lid_velocity=1.0, CFL=2.5,
        corner_treatment=corner, Lx=1.0, Ly=1.0
    )

    fine_sm = MultigridSmoother(level=fine, **params)
    coarse_sm = MultigridSmoother(level=coarse, **params)

    fine_sm.initialize_lid()
    coarse_sm.initialize_lid()

    # Pre-smooth fine to get a non-trivial state
    print("\n1. Pre-smoothing fine grid (50 iterations)...")
    for _ in range(50):
        fine_sm.step()

    # Compute fine residual before V-cycle
    R_fine_before = compute_ns_residual_norm(fine_sm, fine)
    print(f"   Fine NS residual: {R_fine_before:.4e}")

    # === V-CYCLE STEP BY STEP ===

    # Step 1: Compute fine residual
    print("\n2. Computing fine grid residual...")
    fine_sm._compute_residuals(fine.u, fine.v, fine.p)

    # Save fine state
    u_fine_old = fine.u.copy()
    v_fine_old = fine.v.copy()
    p_fine_old = fine.p.copy()

    # Step 2: Restrict solution (injection)
    print("\n3. Restricting solution (injection)...")
    restrict_solution(fine, coarse, transfer_ops)
    u_coarse_old = coarse.u.copy()
    v_coarse_old = coarse.v.copy()
    p_coarse_old = coarse.p.copy()

    # Step 3: Restrict residual (FFT)
    print("\n4. Restricting residual (FFT)...")
    restrict_residual(fine, coarse, transfer_ops)
    I_R_u = coarse.R_u.copy()
    I_R_v = coarse.R_v.copy()
    I_R_p = coarse.R_p.copy()

    I_R_norm = np.sqrt(np.sum(I_R_u**2) + np.sum(I_R_v**2) + np.sum(I_R_p**2))
    print(f"   |I(R_h)| = {I_R_norm:.4e}")

    # Step 4: Compute tau correction
    print("\n5. Computing tau correction...")
    coarse_sm._compute_residuals(coarse.u, coarse.v, coarse.p)
    R_H_norm = np.sqrt(np.sum(coarse.R_u**2) + np.sum(coarse.R_v**2) + np.sum(coarse.R_p**2))
    print(f"   |R_H(I(u_h))| = {R_H_norm:.4e}")

    tau_u = I_R_u - coarse.R_u
    tau_v = I_R_v - coarse.R_v
    tau_p = I_R_p - coarse.R_p

    # Zero boundaries on tau
    tau_u_2d = tau_u.reshape(coarse.shape_full)
    tau_v_2d = tau_v.reshape(coarse.shape_full)
    tau_u_2d[0, :] = tau_u_2d[-1, :] = tau_u_2d[:, 0] = tau_u_2d[:, -1] = 0
    tau_v_2d[0, :] = tau_v_2d[-1, :] = tau_v_2d[:, 0] = tau_v_2d[:, -1] = 0

    tau_norm = np.sqrt(np.sum(tau_u**2) + np.sum(tau_v**2) + np.sum(tau_p**2))
    print(f"   |tau| = {tau_norm:.4e}")

    # Step 5: Coarse solve with tau (TEST DIFFERENT ITERATIONS)
    print("\n6. Coarse grid solve with tau...")
    coarse_sm.set_tau_correction(tau_u, tau_v, tau_p)

    # Track coarse residual during solve
    for n_coarse_iters in [1, 3, 10, 30]:
        # Reset coarse to injected state
        coarse.u[:] = u_coarse_old.copy()
        coarse.v[:] = v_coarse_old.copy()
        coarse.p[:] = p_coarse_old.copy()
        coarse_sm.set_tau_correction(tau_u, tau_v, tau_p)

        # Get initial coarse residual (with tau)
        coarse_sm._compute_residuals(coarse.u, coarse.v, coarse.p)
        R_coarse_before = np.sqrt(np.sum(coarse.R_u**2) + np.sum(coarse.R_v**2))

        # Smooth
        for _ in range(n_coarse_iters):
            coarse_sm.step()

        # Get final coarse residual (with tau)
        coarse_sm._compute_residuals(coarse.u, coarse.v, coarse.p)
        R_coarse_after = np.sqrt(np.sum(coarse.R_u**2) + np.sum(coarse.R_v**2))

        # Compute correction
        delta_u = coarse.u - u_coarse_old
        delta_v = coarse.v - v_coarse_old
        delta_p = coarse.p - p_coarse_old

        delta_norm = np.sqrt(np.sum(delta_u**2) + np.sum(delta_v**2) + np.sum(delta_p**2))

        print(f"   Coarse iters={n_coarse_iters:2d}: R_before={R_coarse_before:.2e}, "
              f"R_after={R_coarse_after:.2e}, |delta|={delta_norm:.2e}")

        if n_coarse_iters == 10:
            # Use this for the rest of the diagnostic
            delta_u_save = delta_u.copy()
            delta_v_save = delta_v.copy()
            delta_p_save = delta_p.copy()

    coarse_sm.clear_tau_correction()

    # Step 6: Prolongate correction and test different damping
    print("\n7. Testing correction with different damping values...")

    # Zero boundaries on correction
    delta_u_2d = delta_u_save.reshape(coarse.shape_full).copy()
    delta_v_2d = delta_v_save.reshape(coarse.shape_full).copy()
    delta_p_2d = delta_p_save.reshape(coarse.shape_inner)

    delta_u_2d[0, :] = delta_u_2d[-1, :] = delta_u_2d[:, 0] = delta_u_2d[:, -1] = 0
    delta_v_2d[0, :] = delta_v_2d[-1, :] = delta_v_2d[:, 0] = delta_v_2d[:, -1] = 0

    # Prolongate
    delta_u_fine_2d = transfer_ops.prolongation.prolongate_2d(delta_u_2d, fine.shape_full)
    delta_v_fine_2d = transfer_ops.prolongation.prolongate_2d(delta_v_2d, fine.shape_full)
    delta_p_fine_2d = transfer_ops.prolongation.prolongate_2d(delta_p_2d, fine.shape_inner)

    # Zero boundaries on prolongated correction
    delta_u_fine_2d[0, :] = delta_u_fine_2d[-1, :] = delta_u_fine_2d[:, 0] = delta_u_fine_2d[:, -1] = 0
    delta_v_fine_2d[0, :] = delta_v_fine_2d[-1, :] = delta_v_fine_2d[:, 0] = delta_v_fine_2d[:, -1] = 0

    delta_u_fine = delta_u_fine_2d.ravel()
    delta_v_fine = delta_v_fine_2d.ravel()
    delta_p_fine = delta_p_fine_2d.ravel()

    for damping in [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]:
        # Apply correction
        fine.u[:] = u_fine_old + damping * delta_u_fine
        fine.v[:] = v_fine_old + damping * delta_v_fine
        fine.p[:] = p_fine_old + damping * delta_p_fine
        fine_sm._enforce_boundary_conditions(fine.u, fine.v)

        # Compute fine residual after correction
        R_fine_after = compute_ns_residual_norm(fine_sm, fine)
        ratio = R_fine_after / R_fine_before
        status = "✓" if ratio < 1.0 else "✗"
        print(f"   damping={damping:.1f}: R_after={R_fine_after:.4e}, ratio={ratio:.4f} {status}")


def test_multiple_vcycles_with_ns_residual():
    """Track actual NS residual over multiple V-cycles."""
    print("\n" + "=" * 70)
    print("VMG DIAGNOSTIC: Multiple V-cycles with NS residual tracking")
    print("=" * 70)

    N_fine = 24
    n_levels = 2

    basis = ChebyshevLobattoBasis(domain=(0.0, 1.0))
    transfer_ops = create_transfer_operators("fft", "fft")
    corner = create_corner_treatment("smoothing")

    levels = build_hierarchy(N_fine, n_levels, basis, basis)
    coarse, fine = levels[0], levels[1]

    params = dict(
        Re=100.0, beta_squared=5.0, lid_velocity=1.0, CFL=2.5,
        corner_treatment=corner, Lx=1.0, Ly=1.0
    )

    fine_sm = MultigridSmoother(level=fine, **params)
    coarse_sm = MultigridSmoother(level=coarse, **params)

    fine_sm.initialize_lid()
    coarse_sm.initialize_lid()

    # Pre-smoothing config per paper
    n_presmooth_fine = 1
    n_presmooth_coarse = 3
    damping = 1.0  # Test with full correction first

    print(f"\nConfig: N_fine={N_fine}, N_coarse={coarse.n}")
    print(f"Pre-smoothing: fine={n_presmooth_fine}, coarse={n_presmooth_coarse}")
    print(f"Damping: {damping}")
    print(f"CFL: 2.5")

    print("\n" + "-" * 70)
    print(f"{'Cycle':<6} {'R_before':>12} {'R_after':>12} {'Ratio':>10} {'Status':<8}")
    print("-" * 70)

    for cycle in range(30):
        # Get NS residual before V-cycle
        R_before = compute_ns_residual_norm(fine_sm, fine)

        # === V-CYCLE ===
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

        # Get NS residual after V-cycle
        R_after = compute_ns_residual_norm(fine_sm, fine)
        ratio = R_after / R_before
        status = "✓" if ratio < 1.0 else "✗"

        print(f"{cycle+1:<6d} {R_before:>12.4e} {R_after:>12.4e} {ratio:>10.4f} {status:<8}")

        if R_after > 1e10:
            print("DIVERGED!")
            break


def compare_coarse_solves():
    """Compare coarse grid behavior with and without tau."""
    print("\n" + "=" * 70)
    print("VMG DIAGNOSTIC: Coarse grid behavior with/without tau")
    print("=" * 70)

    N_fine = 24
    n_levels = 2

    basis = ChebyshevLobattoBasis(domain=(0.0, 1.0))
    transfer_ops = create_transfer_operators("fft", "fft")
    corner = create_corner_treatment("smoothing")

    levels = build_hierarchy(N_fine, n_levels, basis, basis)
    coarse, fine = levels[0], levels[1]

    params = dict(
        Re=100.0, beta_squared=5.0, lid_velocity=1.0, CFL=2.5,
        corner_treatment=corner, Lx=1.0, Ly=1.0
    )

    fine_sm = MultigridSmoother(level=fine, **params)
    coarse_sm = MultigridSmoother(level=coarse, **params)

    fine_sm.initialize_lid()
    coarse_sm.initialize_lid()

    # Pre-smooth fine
    for _ in range(50):
        fine_sm.step()

    # Get restriction and tau
    fine_sm._compute_residuals(fine.u, fine.v, fine.p)
    restrict_solution(fine, coarse, transfer_ops)
    u_coarse_init = coarse.u.copy()
    v_coarse_init = coarse.v.copy()
    p_coarse_init = coarse.p.copy()

    restrict_residual(fine, coarse, transfer_ops)
    I_R_u = coarse.R_u.copy()
    I_R_v = coarse.R_v.copy()
    I_R_p = coarse.R_p.copy()

    coarse_sm._compute_residuals(coarse.u, coarse.v, coarse.p)
    tau_u = I_R_u - coarse.R_u
    tau_v = I_R_v - coarse.R_v
    tau_p = I_R_p - coarse.R_p

    # Zero tau boundaries
    tau_u_2d = tau_u.reshape(coarse.shape_full)
    tau_v_2d = tau_v.reshape(coarse.shape_full)
    tau_u_2d[0, :] = tau_u_2d[-1, :] = tau_u_2d[:, 0] = tau_u_2d[:, -1] = 0
    tau_v_2d[0, :] = tau_v_2d[-1, :] = tau_v_2d[:, 0] = tau_v_2d[:, -1] = 0

    print("\nTest A: Coarse grid smoothing WITHOUT tau (should converge to coarse solution)")
    coarse.u[:] = u_coarse_init.copy()
    coarse.v[:] = v_coarse_init.copy()
    coarse.p[:] = p_coarse_init.copy()
    coarse_sm.clear_tau_correction()

    for i in range(10):
        coarse_sm._compute_residuals(coarse.u, coarse.v, coarse.p)
        R = np.sqrt(np.sum(coarse.R_u**2) + np.sum(coarse.R_v**2))
        if i % 2 == 0:
            print(f"  Iter {i}: R = {R:.4e}")
        for _ in range(10):
            coarse_sm.step()

    print("\nTest B: Coarse grid smoothing WITH tau (should stay near injected state)")
    coarse.u[:] = u_coarse_init.copy()
    coarse.v[:] = v_coarse_init.copy()
    coarse.p[:] = p_coarse_init.copy()
    coarse_sm.set_tau_correction(tau_u, tau_v, tau_p)

    for i in range(10):
        coarse_sm._compute_residuals(coarse.u, coarse.v, coarse.p)
        R = np.sqrt(np.sum(coarse.R_u**2) + np.sum(coarse.R_v**2))
        delta = np.linalg.norm(coarse.u - u_coarse_init)
        if i % 2 == 0:
            print(f"  Iter {i}: R_tau = {R:.4e}, |u - u_init| = {delta:.4e}")
        for _ in range(10):
            coarse_sm.step()

    coarse_sm.clear_tau_correction()


if __name__ == "__main__":
    diagnose_single_vcycle()
    test_multiple_vcycles_with_ns_residual()
    compare_coarse_solves()
