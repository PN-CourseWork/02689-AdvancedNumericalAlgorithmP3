#!/usr/bin/env python
"""Focused debug: Test if coarse correction improves fine grid residual."""

import logging
import numpy as np
import sys

logging.basicConfig(level=logging.WARNING, format='%(message)s')
log = logging.getLogger(__name__)

sys.path.insert(0, "src")

from solvers.spectral.basis.spectral import ChebyshevLobattoBasis
from solvers.spectral.multigrid.fsg import (
    build_spectral_level,
    build_hierarchy,
    MultigridSmoother,
    restrict_solution,
    restrict_residual,
)
from solvers.spectral.operators.transfer_operators import create_transfer_operators
from solvers.spectral.operators.corner import create_corner_treatment


def test_coarse_correction_quality():
    """Test if solving on coarse improves fine grid residual."""
    print("="*70)
    print("Testing: Does coarse grid correction improve fine grid residual?")
    print("="*70)

    basis = ChebyshevLobattoBasis(domain=(0.0, 1.0))
    transfer_ops = create_transfer_operators("fft", "fft")
    corner = create_corner_treatment("smoothing")

    # Build 2-level hierarchy
    levels = build_hierarchy(15, 2, basis, basis)
    coarse, fine = levels[0], levels[1]

    # Create smoothers
    params = dict(Re=100.0, beta_squared=5.0, lid_velocity=1.0, CFL=0.5,
                  corner_treatment=corner, Lx=1.0, Ly=1.0)
    coarse_sm = MultigridSmoother(level=coarse, **params)
    fine_sm = MultigridSmoother(level=fine, **params)

    # Initialize fine grid
    fine_sm.initialize_lid()

    print("\nPhase 1: Pre-smooth fine grid")
    for i in range(50):
        u_res, v_res = fine_sm.step()
        if i % 10 == 0:
            print(f"  Fine iter {i}: max_res = {max(u_res, v_res):.2e}")

    # Compute fine grid residual
    fine_sm._compute_residuals(fine.u, fine.v, fine.p)
    res_fine_before = np.sqrt(np.sum(fine.R_u**2) + np.sum(fine.R_v**2))
    print(f"\n  Fine residual norm after pre-smoothing: {res_fine_before:.4e}")

    # Save fine grid state
    u_fine_pre = fine.u.copy()
    v_fine_pre = fine.v.copy()
    p_fine_pre = fine.p.copy()
    R_u_fine = fine.R_u.copy()
    R_v_fine = fine.R_v.copy()
    R_p_fine = fine.R_p.copy()

    print("\nPhase 2: Restrict to coarse and compute tau")

    # Restrict solution
    restrict_solution(fine, coarse, transfer_ops)
    u_coarse_init = coarse.u.copy()
    v_coarse_init = coarse.v.copy()
    p_coarse_init = coarse.p.copy()

    # Restrict residual
    restrict_residual(fine, coarse, transfer_ops)
    I_R_u = coarse.R_u.copy()
    I_R_v = coarse.R_v.copy()
    I_R_p = coarse.R_p.copy()

    # Compute coarse residual from restricted solution
    coarse_sm.initialize_lid()
    coarse_sm._compute_residuals(coarse.u, coarse.v, coarse.p)

    # Tau correction
    tau_u = I_R_u - coarse.R_u
    tau_v = I_R_v - coarse.R_v
    tau_p = I_R_p - coarse.R_p

    print(f"  |I(R_u)|  = {np.linalg.norm(I_R_u):.4e}")
    print(f"  |L_H(u)|  = {np.linalg.norm(coarse.R_u):.4e}")
    print(f"  |tau_u|   = {np.linalg.norm(tau_u):.4e}")

    print("\nPhase 3: Solve coarse grid with tau correction")

    # Apply tau and solve coarse
    coarse_sm.set_tau_correction(tau_u, tau_v, tau_p)

    # Try different numbers of coarse iterations
    for n_coarse in [10, 50, 200, 500]:
        # Reset coarse solution
        coarse.u[:] = u_coarse_init
        coarse.v[:] = v_coarse_init
        coarse.p[:] = p_coarse_init
        coarse_sm.set_tau_correction(tau_u, tau_v, tau_p)

        for _ in range(n_coarse):
            coarse_sm.step()

        coarse_sm.clear_tau_correction()

        # Compute coarse correction
        delta_u = coarse.u - u_coarse_init
        delta_v = coarse.v - v_coarse_init
        delta_p = coarse.p - p_coarse_init

        # Zero boundary corrections
        delta_u_2d = delta_u.reshape(coarse.shape_full).copy()
        delta_v_2d = delta_v.reshape(coarse.shape_full).copy()
        delta_u_2d[0, :] = delta_u_2d[-1, :] = delta_u_2d[:, 0] = delta_u_2d[:, -1] = 0
        delta_v_2d[0, :] = delta_v_2d[-1, :] = delta_v_2d[:, 0] = delta_v_2d[:, -1] = 0

        # Prolongate
        delta_u_fine = transfer_ops.prolongation.prolongate_2d(delta_u_2d, fine.shape_full)
        delta_v_fine = transfer_ops.prolongation.prolongate_2d(delta_v_2d, fine.shape_full)
        delta_p_fine = transfer_ops.prolongation.prolongate_2d(
            delta_p.reshape(coarse.shape_inner), fine.shape_inner
        )

        # Zero boundary on fine
        delta_u_fine[0, :] = delta_u_fine[-1, :] = delta_u_fine[:, 0] = delta_u_fine[:, -1] = 0
        delta_v_fine[0, :] = delta_v_fine[-1, :] = delta_v_fine[:, 0] = delta_v_fine[:, -1] = 0

        print(f"\n  Coarse iters = {n_coarse}:")
        print(f"    |delta_u_fine| = {np.linalg.norm(delta_u_fine):.4e}")
        print(f"    |delta_v_fine| = {np.linalg.norm(delta_v_fine):.4e}")

        # Test different damping values
        for damping in [0.2, 0.5, 1.0]:
            # Apply correction
            fine.u[:] = u_fine_pre + damping * delta_u_fine.ravel()
            fine.v[:] = v_fine_pre + damping * delta_v_fine.ravel()
            fine.p[:] = p_fine_pre + damping * delta_p_fine.ravel()
            fine_sm._enforce_boundary_conditions(fine.u, fine.v)

            # Compute new residual
            fine_sm._compute_residuals(fine.u, fine.v, fine.p)
            res_fine_after = np.sqrt(np.sum(fine.R_u**2) + np.sum(fine.R_v**2))

            ratio = res_fine_after / res_fine_before
            status = "✓ BETTER" if ratio < 1.0 else "✗ WORSE"
            print(f"    damping={damping}: residual ratio = {ratio:.4f} {status}")


def test_smoother_convergence_comparison():
    """Compare pure smoothing vs V-cycle assisted smoothing."""
    print("\n" + "="*70)
    print("Comparing: Pure fine grid smoothing vs V-cycle assisted")
    print("="*70)

    basis = ChebyshevLobattoBasis(domain=(0.0, 1.0))
    transfer_ops = create_transfer_operators("fft", "fft")
    corner = create_corner_treatment("smoothing")

    params = dict(Re=100.0, beta_squared=5.0, lid_velocity=1.0, CFL=0.5,
                  corner_treatment=corner, Lx=1.0, Ly=1.0)

    # Test 1: Pure fine grid (N=15)
    print("\nTest A: Pure fine grid (N=15), 1000 iterations")
    fine = build_spectral_level(15, 0, basis, basis)
    fine_sm = MultigridSmoother(level=fine, **params)
    fine_sm.initialize_lid()

    res_history_pure = []
    for i in range(1000):
        u_res, v_res = fine_sm.step()
        if i % 200 == 0:
            res_history_pure.append(max(u_res, v_res))
            print(f"  Iter {i}: res = {res_history_pure[-1]:.4e}")

    # Test 2: V-cycle assisted (doing full V-cycle properly)
    print("\nTest B: With V-cycle (50 coarse iters + damping=0.5), same work")
    levels = build_hierarchy(15, 2, basis, basis)
    coarse, fine = levels[0], levels[1]
    coarse_sm = MultigridSmoother(level=coarse, **params)
    fine_sm = MultigridSmoother(level=fine, **params)
    fine_sm.initialize_lid()

    def one_vcycle():
        """One V-cycle: presmooth, restrict, coarse solve, prolongate."""
        # Pre-smooth (10 iters)
        for _ in range(10):
            fine_sm.step()

        # Compute and restrict
        fine_sm._compute_residuals(fine.u, fine.v, fine.p)
        u_old = fine.u.copy()
        v_old = fine.v.copy()
        p_old = fine.p.copy()

        restrict_solution(fine, coarse, transfer_ops)
        u_c_old = coarse.u.copy()
        v_c_old = coarse.v.copy()
        p_c_old = coarse.p.copy()

        restrict_residual(fine, coarse, transfer_ops)
        I_R_u = coarse.R_u.copy()
        I_R_v = coarse.R_v.copy()
        I_R_p = coarse.R_p.copy()

        coarse_sm._compute_residuals(coarse.u, coarse.v, coarse.p)
        tau_u = I_R_u - coarse.R_u
        tau_v = I_R_v - coarse.R_v
        tau_p = I_R_p - coarse.R_p

        coarse_sm.set_tau_correction(tau_u, tau_v, tau_p)

        # Coarse solve (50 iters = ~10 fine iters equivalent work)
        for _ in range(50):
            coarse_sm.step()
        coarse_sm.clear_tau_correction()

        # Compute correction
        delta_u = coarse.u - u_c_old
        delta_v = coarse.v - v_c_old
        delta_p = coarse.p - p_c_old

        # Zero boundaries
        delta_u_2d = delta_u.reshape(coarse.shape_full).copy()
        delta_v_2d = delta_v.reshape(coarse.shape_full).copy()
        delta_u_2d[0, :] = delta_u_2d[-1, :] = delta_u_2d[:, 0] = delta_u_2d[:, -1] = 0
        delta_v_2d[0, :] = delta_v_2d[-1, :] = delta_v_2d[:, 0] = delta_v_2d[:, -1] = 0

        # Prolongate
        delta_u_f = transfer_ops.prolongation.prolongate_2d(delta_u_2d, fine.shape_full)
        delta_v_f = transfer_ops.prolongation.prolongate_2d(delta_v_2d, fine.shape_full)
        delta_p_f = transfer_ops.prolongation.prolongate_2d(
            delta_p.reshape(coarse.shape_inner), fine.shape_inner
        )
        delta_u_f[0, :] = delta_u_f[-1, :] = delta_u_f[:, 0] = delta_u_f[:, -1] = 0
        delta_v_f[0, :] = delta_v_f[-1, :] = delta_v_f[:, 0] = delta_v_f[:, -1] = 0

        # Apply with damping
        fine.u[:] = u_old + 0.5 * delta_u_f.ravel()
        fine.v[:] = v_old + 0.5 * delta_v_f.ravel()
        fine.p[:] = p_old + 0.5 * delta_p_f.ravel()
        fine_sm._enforce_boundary_conditions(fine.u, fine.v)

        # Post-smooth (10 iters)
        for _ in range(10):
            u_res, v_res = fine_sm.step()

        return max(u_res, v_res)

    # Each V-cycle ~ 20 fine iters + 50 coarse iters (~20 + 50/4 = ~33 fine equiv)
    # So 30 V-cycles ~ 1000 fine iters
    res_history_vcycle = []
    for i in range(30):
        res = one_vcycle()
        res_history_vcycle.append(res)
        print(f"  V-cycle {i}: res = {res:.4e}")

    print("\n" + "="*70)
    print("Summary:")
    print(f"  Pure smoothing: started at {res_history_pure[0]:.4e}, ended at {res_history_pure[-1]:.4e}")
    print(f"  V-cycle:        started at {res_history_vcycle[0]:.4e}, ended at {res_history_vcycle[-1]:.4e}")
    print("="*70)


def test_coarse_grid_convergence():
    """Test if coarse grid can converge on its own."""
    print("\n" + "="*70)
    print("Testing: Can coarse grid (N=7) converge on its own?")
    print("="*70)

    basis = ChebyshevLobattoBasis(domain=(0.0, 1.0))
    corner = create_corner_treatment("smoothing")

    coarse = build_spectral_level(7, 0, basis, basis)
    params = dict(Re=100.0, beta_squared=5.0, lid_velocity=1.0, CFL=0.5,
                  corner_treatment=corner, Lx=1.0, Ly=1.0)
    smoother = MultigridSmoother(level=coarse, **params)
    smoother.initialize_lid()

    print("\nRunning 2000 iterations on N=7 grid:")
    for i in range(2000):
        u_res, v_res = smoother.step()
        if i % 400 == 0:
            print(f"  Iter {i}: u_res = {u_res:.4e}, v_res = {v_res:.4e}")

    print(f"\n  Final: u_res = {u_res:.4e}, v_res = {v_res:.4e}")
    if max(u_res, v_res) < 1e-5:
        print("  ✓ Coarse grid converges")
    else:
        print("  ⚠ Coarse grid hasn't converged yet")


if __name__ == "__main__":
    test_coarse_correction_quality()
    test_coarse_grid_convergence()
    test_smoother_convergence_comparison()
