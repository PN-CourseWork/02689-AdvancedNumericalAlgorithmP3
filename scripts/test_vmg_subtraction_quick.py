#!/usr/bin/env python
"""Quick test: Does subtraction method fix VMG coarse grid smoothing?

The key test: with subtraction method, does smoothing with tau DECREASE
the modified residual? (Previously it INCREASED.)
"""

import logging
import numpy as np
import sys
import time

logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, "src")

from solvers.spectral.basis.spectral import ChebyshevLobattoBasis
from solvers.spectral.multigrid.fsg import (
    build_hierarchy, MultigridSmoother,
    restrict_solution, restrict_residual,
)
from solvers.spectral.operators.transfer_operators import create_transfer_operators
from solvers.spectral.operators.corner import create_corner_treatment


def test_coarse_smoothing_with_tau(corner_method):
    """Test if coarse grid smoothing with tau reduces modified residual."""
    print(f"\n{'='*60}")
    print(f"Testing coarse smoothing with tau ({corner_method} corner)")
    print('='*60)

    N_fine = 24
    n_levels = 2

    basis = ChebyshevLobattoBasis(domain=(0.0, 1.0))
    transfer_ops = create_transfer_operators("fft", "fft")
    corner = create_corner_treatment(corner_method)

    levels = build_hierarchy(N_fine, n_levels, basis, basis)
    fine, coarse = levels[1], levels[0]

    fine_sm = MultigridSmoother(
        level=fine, Re=100.0, beta_squared=5.0, lid_velocity=1.0,
        CFL=2.5, corner_treatment=corner, Lx=1.0, Ly=1.0
    )
    coarse_sm = MultigridSmoother(
        level=coarse, Re=100.0, beta_squared=5.0, lid_velocity=1.0,
        CFL=2.5, corner_treatment=corner, Lx=1.0, Ly=1.0
    )

    fine_sm.initialize_lid()

    # Do some fine smoothing to get non-trivial solution
    print("Initial fine smoothing (10 steps)...")
    for _ in range(10):
        fine_sm.step()

    # Compute fine residual
    fine_sm._compute_residuals(fine.u, fine.v, fine.p)
    print(f"Fine residual norm: |R_fine| = {np.linalg.norm(fine.R_u):.4e}")

    # Restrict solution
    restrict_solution(fine, coarse, transfer_ops)

    # Restrict residual
    restrict_residual(fine, coarse, transfer_ops)
    I_R_u = coarse.R_u.copy()
    print(f"Restricted fine residual: |I(R_fine)| = {np.linalg.norm(I_R_u):.4e}")

    # Compute coarse residual at restricted solution
    coarse_sm._compute_residuals(coarse.u, coarse.v, coarse.p)
    r_coarse = coarse.R_u.copy()
    print(f"Coarse residual: |R_coarse| = {np.linalg.norm(r_coarse):.4e}")

    # Compute tau
    tau_u = I_R_u - r_coarse
    tau_v = coarse.R_v.copy() * 0  # Simplified: only test u
    tau_p = coarse.R_p.copy() * 0
    print(f"Tau norm: |tau| = {np.linalg.norm(tau_u):.4e}")

    # Zero tau boundaries
    tau_u_2d = tau_u.reshape(coarse.shape_full)
    tau_u_2d[0, :] = tau_u_2d[-1, :] = tau_u_2d[:, 0] = tau_u_2d[:, -1] = 0

    # Verify: modified residual at restricted solution should equal I(R_fine)
    R_mod = r_coarse + tau_u
    print(f"Modified residual at start: |R + tau| = {np.linalg.norm(R_mod):.4e}")
    print(f"Should equal I(R_fine): diff = {np.linalg.norm(R_mod - I_R_u):.4e}")

    # KEY TEST: Does smoothing with tau REDUCE |R + tau|?
    print(f"\nSmoothing coarse grid with tau (10 steps):")
    coarse_sm.set_tau_correction(tau_u, tau_v, tau_p)

    residuals = []
    for i in range(10):
        coarse_sm._compute_residuals(coarse.u, coarse.v, coarse.p)
        R_with_tau = coarse.R_u.copy()  # Already includes tau
        res_norm = np.linalg.norm(R_with_tau)
        residuals.append(res_norm)

        if i == 0:
            initial_res = res_norm

        ratio = res_norm / residuals[i-1] if i > 0 else 1.0
        status = "DECREASING" if ratio < 1.0 else "INCREASING!"
        print(f"  Step {i+1}: |R + tau| = {res_norm:.4e}, ratio = {ratio:.3f} {status}")

        coarse_sm.step()

    coarse_sm.clear_tau_correction()

    final_res = residuals[-1]
    overall_ratio = final_res / initial_res
    print(f"\nOverall: {initial_res:.4e} -> {final_res:.4e} ({overall_ratio:.3f}x)")

    return overall_ratio < 0.5  # Success if significantly reduced


def main():
    print("="*60)
    print("KEY TEST: Does subtraction method fix coarse grid smoothing?")
    print("="*60)

    t0 = time.time()
    success_smoothing = test_coarse_smoothing_with_tau("smoothing")
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s")

    t0 = time.time()
    success_subtraction = test_coarse_smoothing_with_tau("subtraction")
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s")

    print("\n" + "="*60)
    print("RESULTS:")
    print("="*60)
    print(f"Smoothing method:    {'PASS - reduced' if success_smoothing else 'FAIL - increased'}")
    print(f"Subtraction method:  {'PASS - reduced' if success_subtraction else 'FAIL - increased'}")

    if success_subtraction and not success_smoothing:
        print("\nCONCLUSION: Subtraction method FIXES coarse grid smoothing!")
        print("VMG should now work properly.")
    elif success_subtraction and success_smoothing:
        print("\nBoth methods work - subtraction may still be better for VMG")
    elif not success_subtraction:
        print("\nSubtraction method also fails - issue is elsewhere")


if __name__ == "__main__":
    main()
