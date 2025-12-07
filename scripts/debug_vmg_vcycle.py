#!/usr/bin/env python
"""Debug a single V-cycle step-by-step per Zhang & Xi (2010) paper.

Paper FAS algorithm (Section 3.1):
1. Fine grid: A_k * phi_old_k = b_k + r_k  (r_k is residual)
2. Restrict to coarse: A_{k-1} * phi_{k-1} = b_{k-1} + R(r_k)
3. Update fine: phi_new_k = phi_old_k + P(phi_new_{k-1} - phi_old_{k-1})

Key paper notes:
- Variables restriction: direct injection (coarse GLL are subset of fine)
- Residual restriction: FFT-based (truncate high frequencies)
- Correction prolongation: FFT-based
- "residuals and corrections on the boundary points are all set to zero"
"""

import numpy as np
import sys
sys.path.insert(0, "src")

from solvers.spectral.basis.spectral import ChebyshevLobattoBasis
from solvers.spectral.multigrid.fsg import (
    build_hierarchy, MultigridSmoother, restrict_solution, restrict_residual
)
from solvers.spectral.operators.transfer_operators import create_transfer_operators
from solvers.spectral.operators.corner import create_corner_treatment


def debug_single_vcycle():
    """Trace one V-cycle step by step."""
    # Small problem for fast debugging
    N_fine = 15
    n_levels = 2

    basis = ChebyshevLobattoBasis(domain=(0.0, 1.0))
    transfer_ops = create_transfer_operators('fft', 'fft')
    corner = create_corner_treatment('smoothing')

    levels = build_hierarchy(N_fine, n_levels, basis, basis)
    coarse, fine = levels[0], levels[1]

    print(f"Grid sizes: coarse N={coarse.n}, fine N={fine.n}")
    print("=" * 60)

    params = dict(Re=100.0, beta_squared=5.0, lid_velocity=1.0, CFL=2.5,
                  corner_treatment=corner, Lx=1.0, Ly=1.0)

    fine_sm = MultigridSmoother(level=fine, **params)
    coarse_sm = MultigridSmoother(level=coarse, **params)

    # Initialize
    fine_sm.initialize_lid()
    coarse_sm.initialize_lid()

    # Pre-smooth on fine (paper uses 1-3 steps)
    n_presmooth = 3
    print(f"\n1. Pre-smooth fine grid ({n_presmooth} steps)")
    for i in range(n_presmooth):
        u_res, v_res = fine_sm.step()
    print(f"   After presmooth: u_res={u_res:.2e}, v_res={v_res:.2e}")

    # Compute fine residual
    fine_sm._compute_residuals(fine.u, fine.v, fine.p)
    R_h_norm = np.sqrt(np.sum(fine.R_u**2) + np.sum(fine.R_v**2))
    print(f"\n2. Fine residual: |R_h| = {R_h_norm:.4e}")

    # Save fine state
    u_fine_old = fine.u.copy()
    v_fine_old = fine.v.copy()
    p_fine_old = fine.p.copy()

    # RESTRICT VARIABLES (injection per paper)
    print("\n3. Restrict variables (injection)")
    restrict_solution(fine, coarse, transfer_ops)
    u_coarse_old = coarse.u.copy()
    v_coarse_old = coarse.v.copy()
    p_coarse_old = coarse.p.copy()
    print(f"   |u_H| = {np.linalg.norm(u_coarse_old):.4e}")

    # RESTRICT RESIDUALS (FFT per paper)
    print("\n4. Restrict residuals (FFT)")
    restrict_residual(fine, coarse, transfer_ops)
    I_R_u = coarse.R_u.copy()
    I_R_v = coarse.R_v.copy()
    I_R_p = coarse.R_p.copy()
    print(f"   |I(R_u)| = {np.linalg.norm(I_R_u):.4e}")

    # Compute tau = I(r_h) - R_H(I(u_h))
    print("\n5. Compute tau correction")
    coarse_sm._compute_residuals(coarse.u, coarse.v, coarse.p)
    R_H_norm = np.linalg.norm(coarse.R_u)
    print(f"   |R_H(I(u_h))| = {R_H_norm:.4e}")

    tau_u = I_R_u - coarse.R_u
    tau_v = I_R_v - coarse.R_v
    tau_p = I_R_p - coarse.R_p

    # Zero tau boundaries (per paper: "residuals... on boundary points are all set to zero")
    tau_u_2d = tau_u.reshape(coarse.shape_full)
    tau_v_2d = tau_v.reshape(coarse.shape_full)
    tau_u_2d[0, :] = tau_u_2d[-1, :] = tau_u_2d[:, 0] = tau_u_2d[:, -1] = 0
    tau_v_2d[0, :] = tau_v_2d[-1, :] = tau_v_2d[:, 0] = tau_v_2d[:, -1] = 0

    print(f"   |tau_u| = {np.linalg.norm(tau_u):.4e}")
    print(f"   |tau_v| = {np.linalg.norm(tau_v):.4e}")

    # Coarse smooth WITH tau
    n_coarse_smooth = 3
    print(f"\n6. Coarse smooth with tau ({n_coarse_smooth} steps)")
    coarse_sm.set_tau_correction(tau_u, tau_v, tau_p)
    for i in range(n_coarse_smooth):
        u_res, v_res = coarse_sm.step()
    coarse_sm.clear_tau_correction()
    print(f"   After coarse smooth: u_res={u_res:.2e}, v_res={v_res:.2e}")

    # Compute CORRECTION (phi_new - phi_old on coarse)
    print("\n7. Compute coarse correction")
    delta_u = coarse.u - u_coarse_old
    delta_v = coarse.v - v_coarse_old
    delta_p = coarse.p - p_coarse_old
    print(f"   |delta_u| = {np.linalg.norm(delta_u):.4e}")
    print(f"   |delta_v| = {np.linalg.norm(delta_v):.4e}")

    # Zero correction boundaries (per paper)
    delta_u_2d = delta_u.reshape(coarse.shape_full).copy()
    delta_v_2d = delta_v.reshape(coarse.shape_full).copy()
    delta_u_2d[0, :] = delta_u_2d[-1, :] = delta_u_2d[:, 0] = delta_u_2d[:, -1] = 0
    delta_v_2d[0, :] = delta_v_2d[-1, :] = delta_v_2d[:, 0] = delta_v_2d[:, -1] = 0

    # PROLONGATE correction (FFT per paper)
    print("\n8. Prolongate correction (FFT)")
    delta_u_fine = transfer_ops.prolongation.prolongate_2d(delta_u_2d, fine.shape_full)
    delta_v_fine = transfer_ops.prolongation.prolongate_2d(delta_v_2d, fine.shape_full)
    delta_p_fine = transfer_ops.prolongation.prolongate_2d(
        delta_p.reshape(coarse.shape_inner), fine.shape_inner
    )

    # Zero prolongated correction boundaries
    delta_u_fine[0, :] = delta_u_fine[-1, :] = delta_u_fine[:, 0] = delta_u_fine[:, -1] = 0
    delta_v_fine[0, :] = delta_v_fine[-1, :] = delta_v_fine[:, 0] = delta_v_fine[:, -1] = 0

    print(f"   |P(delta_u)| = {np.linalg.norm(delta_u_fine):.4e}")
    print(f"   |P(delta_v)| = {np.linalg.norm(delta_v_fine):.4e}")

    # UPDATE fine grid
    print("\n9. Update fine grid: phi_new = phi_old + P(correction)")
    fine.u[:] = u_fine_old + delta_u_fine.ravel()
    fine.v[:] = v_fine_old + delta_v_fine.ravel()
    fine.p[:] = p_fine_old + delta_p_fine.ravel()
    fine_sm._enforce_boundary_conditions(fine.u, fine.v)

    # Check result
    fine_sm._compute_residuals(fine.u, fine.v, fine.p)
    R_h_new_norm = np.sqrt(np.sum(fine.R_u**2) + np.sum(fine.R_v**2))

    print(f"\n10. Result:")
    print(f"    Fine residual BEFORE V-cycle: {R_h_norm:.4e}")
    print(f"    Fine residual AFTER V-cycle:  {R_h_new_norm:.4e}")
    ratio = R_h_new_norm / R_h_norm
    status = "IMPROVED" if ratio < 1.0 else "WORSE"
    print(f"    Ratio: {ratio:.4f} [{status}]")

    return ratio < 1.0


def test_multiple_vcycles():
    """Run multiple V-cycles and track convergence."""
    N_fine = 15
    n_levels = 2

    basis = ChebyshevLobattoBasis(domain=(0.0, 1.0))
    transfer_ops = create_transfer_operators('fft', 'fft')
    corner = create_corner_treatment('smoothing')

    levels = build_hierarchy(N_fine, n_levels, basis, basis)
    coarse, fine = levels[0], levels[1]

    params = dict(Re=100.0, beta_squared=5.0, lid_velocity=1.0, CFL=2.5,
                  corner_treatment=corner, Lx=1.0, Ly=1.0)

    fine_sm = MultigridSmoother(level=fine, **params)
    coarse_sm = MultigridSmoother(level=coarse, **params)

    fine_sm.initialize_lid()
    coarse_sm.initialize_lid()

    print("\n" + "=" * 60)
    print("Multiple V-cycles test (N=15, 2 levels)")
    print("=" * 60)

    n_presmooth_fine = 1
    n_presmooth_coarse = 3

    for cycle in range(20):
        # Get initial residual
        fine_sm._compute_residuals(fine.u, fine.v, fine.p)
        R_before = np.sqrt(np.sum(fine.R_u**2) + np.sum(fine.R_v**2))

        # Pre-smooth fine
        for _ in range(n_presmooth_fine):
            fine_sm.step()

        # Compute residual after presmooth
        fine_sm._compute_residuals(fine.u, fine.v, fine.p)
        u_fine_old = fine.u.copy()
        v_fine_old = fine.v.copy()
        p_fine_old = fine.p.copy()

        # Restrict
        restrict_solution(fine, coarse, transfer_ops)
        u_coarse_old = coarse.u.copy()
        v_coarse_old = coarse.v.copy()
        p_coarse_old = coarse.p.copy()

        restrict_residual(fine, coarse, transfer_ops)
        I_R_u = coarse.R_u.copy()
        I_R_v = coarse.R_v.copy()
        I_R_p = coarse.R_p.copy()

        # Tau
        coarse_sm._compute_residuals(coarse.u, coarse.v, coarse.p)
        tau_u = I_R_u - coarse.R_u
        tau_v = I_R_v - coarse.R_v
        tau_p = I_R_p - coarse.R_p

        tau_u_2d = tau_u.reshape(coarse.shape_full)
        tau_v_2d = tau_v.reshape(coarse.shape_full)
        tau_u_2d[0, :] = tau_u_2d[-1, :] = tau_u_2d[:, 0] = tau_u_2d[:, -1] = 0
        tau_v_2d[0, :] = tau_v_2d[-1, :] = tau_v_2d[:, 0] = tau_v_2d[:, -1] = 0

        # Coarse smooth
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

        # Prolongate
        delta_u_fine = transfer_ops.prolongation.prolongate_2d(delta_u_2d, fine.shape_full)
        delta_v_fine = transfer_ops.prolongation.prolongate_2d(delta_v_2d, fine.shape_full)
        delta_p_fine = transfer_ops.prolongation.prolongate_2d(
            delta_p.reshape(coarse.shape_inner), fine.shape_inner
        )

        delta_u_fine[0, :] = delta_u_fine[-1, :] = delta_u_fine[:, 0] = delta_u_fine[:, -1] = 0
        delta_v_fine[0, :] = delta_v_fine[-1, :] = delta_v_fine[:, 0] = delta_v_fine[:, -1] = 0

        # Update
        fine.u[:] = u_fine_old + delta_u_fine.ravel()
        fine.v[:] = v_fine_old + delta_v_fine.ravel()
        fine.p[:] = p_fine_old + delta_p_fine.ravel()
        fine_sm._enforce_boundary_conditions(fine.u, fine.v)

        # Final residual
        fine_sm._compute_residuals(fine.u, fine.v, fine.p)
        R_after = np.sqrt(np.sum(fine.R_u**2) + np.sum(fine.R_v**2))

        ratio = R_after / R_before if R_before > 0 else 1.0
        status = "✓" if ratio < 1.0 else "✗"
        print(f"Cycle {cycle+1:2d}: R_before={R_before:.2e}, R_after={R_after:.2e}, ratio={ratio:.3f} {status}")

        if R_after > 1e10:
            print("DIVERGED!")
            break


if __name__ == "__main__":
    print("=" * 60)
    print("DEBUG: Single V-cycle step-by-step")
    print("=" * 60)
    debug_single_vcycle()

    test_multiple_vcycles()
