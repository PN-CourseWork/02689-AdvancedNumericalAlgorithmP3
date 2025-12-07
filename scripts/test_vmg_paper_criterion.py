#!/usr/bin/env python
"""Test VMG with paper's convergence criterion (RMS of divergence)."""

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


def get_divergence_rms(smoother, level):
    """Get RMS of divergence (paper's criterion)."""
    smoother._compute_residuals(level.u, level.v, level.p)
    # R_p = -beta^2 * div(u), so div = -R_p / beta^2
    # But R_p is on inner grid, we just use its RMS
    n_inner = level.shape_inner[0] * level.shape_inner[1]
    div_rms = np.sqrt(np.sum(level.R_p**2 / smoother.beta_squared**2) / n_inner)
    return div_rms


def test_fmg_like_approach():
    """Test FMG approach: solve coarse fully, then use V-cycles."""
    print("=" * 70)
    print("FMG-like approach: Solve coarse to convergence, then V-cycle on fine")
    print("=" * 70)

    N_fine = 24
    n_levels = 2

    basis = ChebyshevLobattoBasis(domain=(0.0, 1.0))
    transfer_ops = create_transfer_operators("fft", "fft")
    corner = create_corner_treatment("smoothing")

    levels = build_hierarchy(N_fine, n_levels, basis, basis)
    coarse, fine = levels[0], levels[1]

    print(f"Grid: fine N={fine.n}, coarse N={coarse.n}")

    # Paper's criterion: ERMS < 1e-4
    tolerance = 1e-4

    # Create smoothers
    fine_sm = MultigridSmoother(
        level=fine, Re=100.0, beta_squared=5.0, lid_velocity=1.0,
        CFL=2.5, corner_treatment=corner, Lx=1.0, Ly=1.0
    )
    coarse_sm = MultigridSmoother(
        level=coarse, Re=100.0, beta_squared=5.0, lid_velocity=1.0,
        CFL=2.5, corner_treatment=corner, Lx=1.0, Ly=1.0
    )

    t0 = time.time()

    # Step 1: Solve coarse grid to convergence (like FMG)
    print("\n1. Solving coarse grid (N=12) to convergence...")
    coarse_sm.initialize_lid()
    coarse_iters = 0
    for i in range(10000):
        coarse_sm.step()
        coarse_iters += 1
        div_rms = get_divergence_rms(coarse_sm, coarse)
        if div_rms < tolerance:
            print(f"   Coarse converged at iter {coarse_iters}: ERMS={div_rms:.2e}")
            break
        if i % 500 == 0 and i > 0:
            print(f"   Iter {i}: ERMS={div_rms:.2e}")

    # Step 2: Prolongate to fine grid
    print("\n2. Prolongating coarse solution to fine grid...")
    prolongate_solution(coarse, fine, transfer_ops)
    fine_sm._enforce_boundary_conditions(fine.u, fine.v)

    div_rms = get_divergence_rms(fine_sm, fine)
    print(f"   Fine ERMS after prolongation: {div_rms:.2e}")

    # Step 3: V-cycle iterations on fine grid
    print("\n3. V-cycle iterations on fine grid...")
    n_presmooth_fine = 1
    n_presmooth_coarse = 3
    damping = 1.0

    def vcycle():
        """One V-cycle with FAS."""
        # Pre-smooth fine
        for _ in range(n_presmooth_fine):
            fine_sm.step()

        u_fine_old = fine.u.copy()
        v_fine_old = fine.v.copy()
        p_fine_old = fine.p.copy()

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

    vcycle_count = 0
    fine_iters = 0
    for cycle in range(100):
        div_before = get_divergence_rms(fine_sm, fine)
        vcycle()
        vcycle_count += 1
        fine_iters += n_presmooth_fine + n_presmooth_coarse
        div_after = get_divergence_rms(fine_sm, fine)

        if cycle < 10 or cycle % 10 == 0:
            ratio = div_after / div_before if div_before > 0 else 1.0
            status = "✓" if ratio < 1.0 else "✗"
            print(f"   Cycle {cycle+1:2d}: ERMS={div_after:.2e}, ratio={ratio:.3f} {status}")

        if div_after < tolerance:
            print(f"\n   CONVERGED at cycle {vcycle_count}!")
            break

    t_total = time.time() - t0
    total_iters = coarse_iters + fine_iters

    print(f"\n4. Summary:")
    print(f"   Coarse iters: {coarse_iters}")
    print(f"   V-cycles: {vcycle_count}")
    print(f"   Total iters: {total_iters}")
    print(f"   Time: {t_total:.2f}s")

    return total_iters


def test_sg_with_paper_criterion():
    """Single grid with paper's criterion."""
    print("\n" + "=" * 70)
    print("Single Grid baseline (paper's criterion)")
    print("=" * 70)

    basis = ChebyshevLobattoBasis(domain=(0.0, 1.0))
    corner = create_corner_treatment("smoothing")

    level = build_spectral_level(24, 0, basis, basis)
    sm = MultigridSmoother(
        level=level, Re=100.0, beta_squared=5.0, lid_velocity=1.0,
        CFL=2.5, corner_treatment=corner, Lx=1.0, Ly=1.0
    )
    sm.initialize_lid()

    tolerance = 1e-4
    t0 = time.time()
    for i in range(20000):
        sm.step()
        div_rms = get_divergence_rms(sm, level)
        if div_rms < tolerance:
            print(f"   Converged at iter {i+1}: ERMS={div_rms:.2e}")
            t_sg = time.time() - t0
            print(f"   Time: {t_sg:.2f}s")
            return i + 1
        if i % 1000 == 0 and i > 0:
            print(f"   Iter {i}: ERMS={div_rms:.2e}")

    return 20000


if __name__ == "__main__":
    fmg_iters = test_fmg_like_approach()
    sg_iters = test_sg_with_paper_criterion()

    print("\n" + "=" * 70)
    print("Summary (using paper's ERMS < 1e-4 criterion):")
    print(f"  Single Grid: {sg_iters} iters")
    print(f"  FMG-like:    {fmg_iters} iters")
    if fmg_iters > 0 and sg_iters > 0:
        speedup = sg_iters / fmg_iters
        print(f"  Speedup:     {speedup:.1f}x")
    print("=" * 70)
