#!/usr/bin/env python
"""Test VMG with EXACT paper settings to diagnose issues.

Paper settings (Zhang & Xi 2010):
- VMG-111: 1 pre-smoothing step at each level
- No damping mentioned (use damping=1.0)
- Convergence criterion: ERMS < 1e-4
- CFL = 2.5
- beta^2 = 5.0
"""

import logging
import numpy as np
import sys
import time

logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, "src")

from solvers.spectral.basis.spectral import ChebyshevLobattoBasis
from solvers.spectral.multigrid.fsg import (
    build_hierarchy, build_spectral_level, MultigridSmoother,
    restrict_solution, restrict_residual, prolongate_solution
)
from solvers.spectral.operators.transfer_operators import create_transfer_operators
from solvers.spectral.operators.corner import create_corner_treatment


def get_erms(smoother, level):
    """Get ERMS = sqrt(sum(divergence^2) / N_inner) - paper's criterion."""
    smoother._compute_residuals(level.u, level.v, level.p)
    # R_p = -beta^2 * div(u), so div = -R_p / beta^2
    divergence = -level.R_p / smoother.beta_squared
    n_inner = level.shape_inner[0] * level.shape_inner[1]
    erms = np.sqrt(np.sum(divergence**2) / n_inner)
    return erms


def single_vcycle_paper(fine, coarse, fine_sm, coarse_sm, transfer_ops,
                        n_pre_fine, n_pre_coarse, damping):
    """Single V-cycle exactly as described in the paper."""

    # Pre-smooth fine level
    for _ in range(n_pre_fine):
        fine_sm.step()

    # Save fine solution before restriction
    u_fine_old = fine.u.copy()
    v_fine_old = fine.v.copy()
    p_fine_old = fine.p.copy()

    # Compute fine residual
    fine_sm._compute_residuals(fine.u, fine.v, fine.p)

    # Restrict solution (direct injection)
    restrict_solution(fine, coarse, transfer_ops)
    u_coarse_old = coarse.u.copy()
    v_coarse_old = coarse.v.copy()
    p_coarse_old = coarse.p.copy()

    # Restrict residual (FFT)
    restrict_residual(fine, coarse, transfer_ops)
    I_R_u = coarse.R_u.copy()
    I_R_v = coarse.R_v.copy()
    I_R_p = coarse.R_p.copy()

    # Compute coarse residual at restricted solution
    coarse_sm._compute_residuals(coarse.u, coarse.v, coarse.p)

    # FAS tau correction: tau = I(r_fine) - r_coarse(I(u_fine))
    tau_u = I_R_u - coarse.R_u
    tau_v = I_R_v - coarse.R_v
    tau_p = I_R_p - coarse.R_p

    # Zero tau boundaries (paper requirement)
    tau_u_2d = tau_u.reshape(coarse.shape_full)
    tau_v_2d = tau_v.reshape(coarse.shape_full)
    tau_u_2d[0, :] = tau_u_2d[-1, :] = tau_u_2d[:, 0] = tau_u_2d[:, -1] = 0
    tau_v_2d[0, :] = tau_v_2d[-1, :] = tau_v_2d[:, 0] = tau_v_2d[:, -1] = 0

    # Coarse solve with tau
    coarse_sm.set_tau_correction(tau_u, tau_v, tau_p)
    for _ in range(n_pre_coarse):
        coarse_sm.step()
    coarse_sm.clear_tau_correction()

    # Compute correction
    delta_u = coarse.u - u_coarse_old
    delta_v = coarse.v - v_coarse_old
    delta_p = coarse.p - p_coarse_old

    # Zero boundary corrections
    delta_u_2d = delta_u.reshape(coarse.shape_full).copy()
    delta_v_2d = delta_v.reshape(coarse.shape_full).copy()
    delta_u_2d[0, :] = delta_u_2d[-1, :] = delta_u_2d[:, 0] = delta_u_2d[:, -1] = 0
    delta_v_2d[0, :] = delta_v_2d[-1, :] = delta_v_2d[:, 0] = delta_v_2d[:, -1] = 0

    # Prolongate correction
    delta_u_fine_2d = transfer_ops.prolongation.prolongate_2d(delta_u_2d, fine.shape_full)
    delta_v_fine_2d = transfer_ops.prolongation.prolongate_2d(delta_v_2d, fine.shape_full)
    delta_p_fine_2d = transfer_ops.prolongation.prolongate_2d(
        delta_p.reshape(coarse.shape_inner), fine.shape_inner
    )

    # Zero prolongated boundary corrections
    delta_u_fine_2d[0, :] = delta_u_fine_2d[-1, :] = delta_u_fine_2d[:, 0] = delta_u_fine_2d[:, -1] = 0
    delta_v_fine_2d[0, :] = delta_v_fine_2d[-1, :] = delta_v_fine_2d[:, 0] = delta_v_fine_2d[:, -1] = 0

    # Apply correction with damping
    fine.u[:] = u_fine_old + damping * delta_u_fine_2d.ravel()
    fine.v[:] = v_fine_old + damping * delta_v_fine_2d.ravel()
    fine.p[:] = p_fine_old + damping * delta_p_fine_2d.ravel()
    fine_sm._enforce_boundary_conditions(fine.u, fine.v)


def test_paper_settings():
    """Test VMG with exact paper settings."""
    print("=" * 70)
    print("Testing VMG with EXACT paper settings")
    print("=" * 70)

    N_fine = 48
    n_levels = 3  # 48 -> 24 -> 12

    basis = ChebyshevLobattoBasis(domain=(0.0, 1.0))
    transfer_ops = create_transfer_operators("fft", "fft")
    corner = create_corner_treatment("smoothing")  # Paper uses subtraction!

    levels = build_hierarchy(N_fine, n_levels, basis, basis)

    print(f"Grid hierarchy: {[lvl.n for lvl in levels]}")
    print(f"Paper settings: CFL=2.5, beta^2=5.0, Re=100")
    print(f"Smoothing: VMG-111 (1 step per level)")
    print(f"Corner treatment: smoothing (paper uses subtraction)")

    # Paper: CFL = 2.5
    smoothers = []
    for lvl in levels:
        sm = MultigridSmoother(
            level=lvl, Re=100.0, beta_squared=5.0, lid_velocity=1.0,
            CFL=2.5, corner_treatment=corner, Lx=1.0, Ly=1.0
        )
        smoothers.append(sm)

    # Initialize finest level
    smoothers[-1].initialize_lid()

    # Paper VMG-111: 1 step per level (index 0=coarsest, 1=middle, 2=finest)
    pre_smoothing = [1, 1, 1]  # coarsest to finest

    print("\n" + "=" * 70)
    print("Test 1: VMG-111 with damping=1.0 (paper default)")
    print("=" * 70)

    # Reset solution
    for lvl in levels:
        lvl.u[:] = 0
        lvl.v[:] = 0
        lvl.p[:] = 0
    smoothers[-1].initialize_lid()

    tolerance = 1e-4
    finest = levels[-1]
    finest_sm = smoothers[-1]

    # For 2-level test (simplest case)
    fine = levels[2]  # N=48
    coarse = levels[1]  # N=24
    fine_sm = smoothers[2]
    coarse_sm = smoothers[1]

    t0 = time.time()
    for cycle in range(100):
        erms_before = get_erms(fine_sm, fine)

        single_vcycle_paper(fine, coarse, fine_sm, coarse_sm, transfer_ops,
                           n_pre_fine=1, n_pre_coarse=1, damping=1.0)

        erms_after = get_erms(fine_sm, fine)
        ratio = erms_after / erms_before if erms_before > 0 else 1.0

        if cycle < 10 or cycle % 10 == 0:
            status = "OK" if ratio < 1.0 else "DIVERGING!"
            print(f"Cycle {cycle+1:3d}: ERMS={erms_after:.2e}, ratio={ratio:.3f} {status}")

        if erms_after < tolerance:
            print(f"\nConverged at cycle {cycle+1}!")
            break

        if erms_after > 1e6 or np.isnan(erms_after):
            print(f"\nDiverged at cycle {cycle+1}!")
            break

    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s")

    print("\n" + "=" * 70)
    print("Test 2: VMG-111 with damping=0.5 (our fix)")
    print("=" * 70)

    # Reset solution
    for lvl in levels:
        lvl.u[:] = 0
        lvl.v[:] = 0
        lvl.p[:] = 0
    smoothers[-1].initialize_lid()

    t0 = time.time()
    for cycle in range(100):
        erms_before = get_erms(fine_sm, fine)

        single_vcycle_paper(fine, coarse, fine_sm, coarse_sm, transfer_ops,
                           n_pre_fine=1, n_pre_coarse=1, damping=0.5)

        erms_after = get_erms(fine_sm, fine)
        ratio = erms_after / erms_before if erms_before > 0 else 1.0

        if cycle < 10 or cycle % 10 == 0:
            status = "OK" if ratio < 1.0 else "DIVERGING!"
            print(f"Cycle {cycle+1:3d}: ERMS={erms_after:.2e}, ratio={ratio:.3f} {status}")

        if erms_after < tolerance:
            print(f"\nConverged at cycle {cycle+1}!")
            break

        if erms_after > 1e6 or np.isnan(erms_after):
            print(f"\nDiverged at cycle {cycle+1}!")
            break

    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s")

    print("\n" + "=" * 70)
    print("Test 3: Verify tau is computed correctly")
    print("=" * 70)

    # Reset solution with some initial smoothing
    for lvl in levels:
        lvl.u[:] = 0
        lvl.v[:] = 0
        lvl.p[:] = 0
    smoothers[-1].initialize_lid()

    # Do some smoothing to get non-zero solution
    for _ in range(10):
        fine_sm.step()

    # Compute fine residual
    fine_sm._compute_residuals(fine.u, fine.v, fine.p)
    r_fine_u = fine.R_u.copy()

    # Restrict solution
    restrict_solution(fine, coarse, transfer_ops)

    # Restrict residual
    restrict_residual(fine, coarse, transfer_ops)
    I_R_u = coarse.R_u.copy()

    # Compute coarse residual
    coarse_sm._compute_residuals(coarse.u, coarse.v, coarse.p)
    r_coarse_u = coarse.R_u.copy()

    # tau = I(r_fine) - r_coarse
    tau_u = I_R_u - r_coarse_u

    print(f"Fine residual norm:      |r_fine|   = {np.linalg.norm(r_fine_u):.4e}")
    print(f"Restricted residual:     |I(r_fine)| = {np.linalg.norm(I_R_u):.4e}")
    print(f"Coarse residual:         |r_coarse|  = {np.linalg.norm(r_coarse_u):.4e}")
    print(f"Tau correction:          |tau|       = {np.linalg.norm(tau_u):.4e}")

    # At the restricted solution, the modified residual should equal I(r_fine)
    # R_modified = r_coarse + tau = r_coarse + I(r_fine) - r_coarse = I(r_fine)
    R_modified = r_coarse_u + tau_u
    print(f"Modified residual:       |R_mod|     = {np.linalg.norm(R_modified):.4e}")
    print(f"Difference from I(r):    |R_mod - I(r)| = {np.linalg.norm(R_modified - I_R_u):.4e}")

    # Now check: does smoothing with tau reduce the modified residual?
    print("\nSmoothing with tau (5 steps):")
    coarse_sm.set_tau_correction(tau_u, tau_u*0, tau_u[:coarse.shape_inner[0]*coarse.shape_inner[1]]*0)  # Only add tau_u

    for i in range(5):
        coarse_sm._compute_residuals(coarse.u, coarse.v, coarse.p)
        # The residual computation ADDS tau, so coarse.R_u already has tau included
        R_with_tau = coarse.R_u.copy()  # This is r_coarse + tau
        print(f"  Step {i+1}: |R + tau| = {np.linalg.norm(R_with_tau):.4e}")
        coarse_sm.step()

    coarse_sm.clear_tau_correction()


if __name__ == "__main__":
    test_paper_settings()
