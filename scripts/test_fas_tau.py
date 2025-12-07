#!/usr/bin/env python
"""Test FAS tau correction mechanism in isolation."""

import logging
import numpy as np
import sys

logging.basicConfig(level=logging.WARNING)

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


def test_fas_tau_at_convergence():
    """Verify: at converged solution, tau correction should be approximately zero.

    FAS property: when fine solution is converged (r_h ≈ 0), the tau correction
    should also be approximately zero, so the coarse problem has no correction.
    """
    print("="*60)
    print("Test: FAS tau at converged solution")
    print("="*60)

    basis = ChebyshevLobattoBasis(domain=(0.0, 1.0))
    transfer_ops = create_transfer_operators("fft", "fft")
    corner = create_corner_treatment("smoothing")

    # Get a converged fine grid solution using many iterations
    levels = build_hierarchy(15, 2, basis, basis)
    coarse, fine = levels[0], levels[1]

    params = dict(Re=100.0, beta_squared=5.0, lid_velocity=1.0, CFL=0.5,
                  corner_treatment=corner, Lx=1.0, Ly=1.0)

    fine_sm = MultigridSmoother(level=fine, **params)
    fine_sm.initialize_lid()

    # Iterate until converged
    print("\nConverging fine grid...")
    for i in range(20000):
        u_res, v_res = fine_sm.step()
        if max(u_res, v_res) < 1e-5:
            print(f"  Fine converged at iter {i}: res={max(u_res, v_res):.2e}")
            break
        if i % 2000 == 0:
            print(f"  Iter {i}: res={max(u_res, v_res):.2e}")

    # Compute fine residual
    fine_sm._compute_residuals(fine.u, fine.v, fine.p)
    fine_res = np.sqrt(np.sum(fine.R_u**2) + np.sum(fine.R_v**2))
    print(f"\n  Fine residual norm: {fine_res:.4e}")

    # Restrict solution
    restrict_solution(fine, coarse, transfer_ops)

    # Restrict residual (should be near zero)
    restrict_residual(fine, coarse, transfer_ops)
    I_R_u = coarse.R_u.copy()
    I_R_v = coarse.R_v.copy()
    I_R_p = coarse.R_p.copy()
    print(f"  |I(R_u)| = {np.linalg.norm(I_R_u):.4e} (should be ~0)")

    # Compute coarse residual from restricted solution
    coarse_sm = MultigridSmoother(level=coarse, **params)
    coarse_sm.initialize_lid()
    coarse_sm._compute_residuals(coarse.u, coarse.v, coarse.p)
    print(f"  |R_H(I(u))| = {np.linalg.norm(coarse.R_u):.4e}")

    # Tau = I(R_h) - R_H(I(u_h))
    tau_u = I_R_u - coarse.R_u
    tau_v = I_R_v - coarse.R_v
    tau_p = I_R_p - coarse.R_p
    print(f"  |tau_u| = {np.linalg.norm(tau_u):.4e}")

    # At convergence, I(R_h) ≈ 0, so tau ≈ -R_H(I(u_h))
    # This means the coarse problem RHS = R_H + tau ≈ R_H - R_H(I(u_h))
    # At the start when u_H = I(u_h), this gives RHS ≈ 0, meaning no change
    print("\n  Expected behavior: with tau set, coarse should stay at I(u_h)")


def test_single_vcycle_effect():
    """Test: does a single V-cycle reduce fine grid residual?"""
    print("\n" + "="*60)
    print("Test: Single V-cycle effect on fine grid residual")
    print("="*60)

    basis = ChebyshevLobattoBasis(domain=(0.0, 1.0))
    transfer_ops = create_transfer_operators("fft", "fft")
    corner = create_corner_treatment("smoothing")

    levels = build_hierarchy(15, 2, basis, basis)
    coarse, fine = levels[0], levels[1]

    params = dict(Re=100.0, beta_squared=5.0, lid_velocity=1.0, CFL=0.5,
                  corner_treatment=corner, Lx=1.0, Ly=1.0)

    coarse_sm = MultigridSmoother(level=coarse, **params)
    fine_sm = MultigridSmoother(level=fine, **params)

    # Initialize and pre-smooth fine
    fine_sm.initialize_lid()
    for _ in range(100):
        fine_sm.step()

    # Get fine residual before V-cycle
    fine_sm._compute_residuals(fine.u, fine.v, fine.p)
    res_before = np.sqrt(np.sum(fine.R_u**2) + np.sum(fine.R_v**2))
    print(f"\n  Fine residual before V-cycle: {res_before:.4e}")

    # Save fine state
    u_fine_old = fine.u.copy()
    v_fine_old = fine.v.copy()
    p_fine_old = fine.p.copy()

    # Restrict solution
    restrict_solution(fine, coarse, transfer_ops)
    u_coarse_old = coarse.u.copy()
    v_coarse_old = coarse.v.copy()
    p_coarse_old = coarse.p.copy()

    # Restrict residual and compute tau
    restrict_residual(fine, coarse, transfer_ops)
    I_R_u = coarse.R_u.copy()
    I_R_v = coarse.R_v.copy()
    I_R_p = coarse.R_p.copy()

    coarse_sm.initialize_lid()
    coarse_sm._compute_residuals(coarse.u, coarse.v, coarse.p)

    tau_u = I_R_u - coarse.R_u
    tau_v = I_R_v - coarse.R_v
    tau_p = I_R_p - coarse.R_p

    # Zero boundaries on tau (per paper)
    tau_u_2d = tau_u.reshape(coarse.shape_full)
    tau_v_2d = tau_v.reshape(coarse.shape_full)
    tau_u_2d[0, :] = tau_u_2d[-1, :] = tau_u_2d[:, 0] = tau_u_2d[:, -1] = 0
    tau_v_2d[0, :] = tau_v_2d[-1, :] = tau_v_2d[:, 0] = tau_v_2d[:, -1] = 0

    print(f"  |tau_u| = {np.linalg.norm(tau_u):.4e}")
    print(f"  |tau_v| = {np.linalg.norm(tau_v):.4e}")

    # Coarse smooth with tau
    coarse_sm.set_tau_correction(tau_u, tau_v, tau_p)
    for i in [10, 50, 100]:
        coarse.u[:] = u_coarse_old.copy()
        coarse.v[:] = v_coarse_old.copy()
        coarse.p[:] = p_coarse_old.copy()
        coarse_sm.set_tau_correction(tau_u, tau_v, tau_p)

        for _ in range(i):
            coarse_sm.step()
        coarse_sm.clear_tau_correction()

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
        delta_u_fine = transfer_ops.prolongation.prolongate_2d(delta_u_2d, fine.shape_full)
        delta_v_fine = transfer_ops.prolongation.prolongate_2d(delta_v_2d, fine.shape_full)
        delta_p_fine = transfer_ops.prolongation.prolongate_2d(
            delta_p.reshape(coarse.shape_inner), fine.shape_inner
        )

        delta_u_fine[0, :] = delta_u_fine[-1, :] = delta_u_fine[:, 0] = delta_u_fine[:, -1] = 0
        delta_v_fine[0, :] = delta_v_fine[-1, :] = delta_v_fine[:, 0] = delta_v_fine[:, -1] = 0

        # Test different damping
        print(f"\n  Coarse iters = {i}:")
        for damping in [0.1, 0.5, 1.0, 2.0]:
            fine.u[:] = u_fine_old + damping * delta_u_fine.ravel()
            fine.v[:] = v_fine_old + damping * delta_v_fine.ravel()
            fine.p[:] = p_fine_old + damping * delta_p_fine.ravel()
            fine_sm._enforce_boundary_conditions(fine.u, fine.v)

            fine_sm._compute_residuals(fine.u, fine.v, fine.p)
            res_after = np.sqrt(np.sum(fine.R_u**2) + np.sum(fine.R_v**2))
            ratio = res_after / res_before
            status = "✓" if ratio < 1.0 else "✗"
            print(f"    damping={damping}: res_ratio={ratio:.4f} {status}")


if __name__ == "__main__":
    test_fas_tau_at_convergence()
    test_single_vcycle_effect()
