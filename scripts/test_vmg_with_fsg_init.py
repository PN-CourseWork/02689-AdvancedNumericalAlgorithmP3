#!/usr/bin/env python
"""Test VMG with FSG initialization (like FMG approach).

The paper uses FMG which starts from coarse solution. Let's use FSG
to get a good initial guess, then VMG to refine it faster.
"""

import logging
import numpy as np
import sys
import time

logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
sys.path.insert(0, "src")

from solvers.spectral.basis.spectral import ChebyshevLobattoBasis
from solvers.spectral.multigrid.fsg import (
    build_hierarchy, solve_vmg, solve_fsg, MultigridSmoother
)
from solvers.spectral.operators.transfer_operators import create_transfer_operators
from solvers.spectral.operators.corner import create_corner_treatment


def test_vmg_with_fsg_init():
    """Use FSG to get initial guess, then VMG to refine."""
    print("=" * 70)
    print("Test: VMG with FSG initialization (like FMG)")
    print("=" * 70)

    N_fine = 24
    n_levels = 2

    basis = ChebyshevLobattoBasis(domain=(0.0, 1.0))
    transfer_ops = create_transfer_operators("fft", "fft")
    corner = create_corner_treatment("smoothing")

    levels = build_hierarchy(N_fine, n_levels, basis, basis)
    print(f"Grid sizes: {[lvl.n for lvl in levels]}")

    # Step 1: Use FSG to get rough initial solution (tol 1e-3)
    print("\n1. FSG to get initial guess (tol=1e-3)...")
    t0 = time.time()
    _, fsg_iters, _ = solve_fsg(
        levels=levels,
        Re=100.0,
        beta_squared=5.0,
        lid_velocity=1.0,
        CFL=2.0,
        tolerance=1e-3,
        max_iterations=5000,
        transfer_ops=transfer_ops,
        corner_treatment=corner,
    )
    t_fsg = time.time() - t0
    print(f"   FSG: {fsg_iters} iters, {t_fsg:.2f}s")

    # Rebuild hierarchy to reset for VMG (keep finest solution)
    finest_u = levels[-1].u.copy()
    finest_v = levels[-1].v.copy()
    finest_p = levels[-1].p.copy()

    levels2 = build_hierarchy(N_fine, n_levels, basis, basis)
    levels2[-1].u[:] = finest_u
    levels2[-1].v[:] = finest_v
    levels2[-1].p[:] = finest_p

    # Step 2: Use VMG to refine (tol 1e-5)
    print("\n2. VMG to refine (tol=1e-5)...")
    t0 = time.time()
    _, vmg_iters, converged = solve_vmg(
        levels=levels2,
        Re=100.0,
        beta_squared=5.0,
        lid_velocity=1.0,
        CFL=2.0,
        tolerance=1e-5,
        max_iterations=100,
        transfer_ops=transfer_ops,
        corner_treatment=corner,
        pre_smoothing=[5, 2],
        correction_damping=0.7,  # Higher damping for faster convergence
    )
    t_vmg = time.time() - t0
    print(f"   VMG: {vmg_iters} iters, {t_vmg:.2f}s, converged={converged}")

    total_iters = fsg_iters + vmg_iters
    total_time = t_fsg + t_vmg
    print(f"\n   Total: {total_iters} iters, {total_time:.2f}s")

    return converged, total_iters


def test_sg_baseline():
    """Single grid baseline for comparison."""
    print("\n" + "=" * 70)
    print("Baseline: Single Grid (N=24)")
    print("=" * 70)

    from solvers.spectral.multigrid.fsg import build_spectral_level

    basis = ChebyshevLobattoBasis(domain=(0.0, 1.0))
    corner = create_corner_treatment("smoothing")

    level = build_spectral_level(24, 0, basis, basis)
    sm = MultigridSmoother(
        level=level, Re=100.0, beta_squared=5.0, lid_velocity=1.0,
        CFL=2.0, corner_treatment=corner, Lx=1.0, Ly=1.0
    )
    sm.initialize_lid()

    t0 = time.time()
    sg_iters = 0
    for i in range(20000):
        u_res, v_res = sm.step()
        sg_iters += 1
        if max(u_res, v_res) < 1e-5:
            print(f"  Converged at iter {sg_iters}: res={max(u_res, v_res):.2e}")
            break
        if i % 2000 == 0 and i > 0:
            print(f"  Iter {i}: res={max(u_res, v_res):.2e}")
    t_sg = time.time() - t0

    print(f"\n  SG: {sg_iters} iters, {t_sg:.2f}s")
    return sg_iters, t_sg


def test_vmg_pure_more_smoothing():
    """Test VMG with more smoothing to see if it helps."""
    print("\n" + "=" * 70)
    print("Test: VMG with more smoothing (pure VMG, no FSG init)")
    print("=" * 70)

    N_fine = 24
    n_levels = 2

    basis = ChebyshevLobattoBasis(domain=(0.0, 1.0))
    transfer_ops = create_transfer_operators("fft", "fft")
    corner = create_corner_treatment("smoothing")

    levels = build_hierarchy(N_fine, n_levels, basis, basis)

    # More aggressive: more smoothing, higher damping
    t0 = time.time()
    _, vmg_iters, converged = solve_vmg(
        levels=levels,
        Re=100.0,
        beta_squared=5.0,
        lid_velocity=1.0,
        CFL=2.0,
        tolerance=1e-5,
        max_iterations=200,
        transfer_ops=transfer_ops,
        corner_treatment=corner,
        pre_smoothing=[10, 5],  # More smoothing
        correction_damping=0.8,  # Higher damping
    )
    t_vmg = time.time() - t0

    print(f"\n  VMG: {vmg_iters} iters, {t_vmg:.2f}s, converged={converged}")
    return converged, vmg_iters


if __name__ == "__main__":
    # Baseline
    sg_iters, t_sg = test_sg_baseline()

    # VMG with more smoothing
    vmg_converged, vmg_iters = test_vmg_pure_more_smoothing()

    # VMG with FSG init
    fmg_converged, fmg_iters = test_vmg_with_fsg_init()

    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  Single Grid:    {sg_iters:5d} iters")
    print(f"  Pure VMG:       {vmg_iters:5d} iters (converged={vmg_converged})")
    print(f"  FSG+VMG (FMG):  {fmg_iters:5d} iters (converged={fmg_converged})")
    print("=" * 70)
