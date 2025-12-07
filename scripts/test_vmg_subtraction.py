#!/usr/bin/env python
"""Test VMG with SUBTRACTION corner treatment (like the paper).

The paper uses subtraction method to analytically remove corner singularities.
This should dramatically improve VMG convergence.
"""

import logging
import numpy as np
import sys
import time

logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, "src")

from solvers.spectral.basis.spectral import ChebyshevLobattoBasis
from solvers.spectral.multigrid.fsg import (
    build_hierarchy, build_spectral_level, MultigridSmoother, solve_vmg, solve_fsg
)
from solvers.spectral.operators.transfer_operators import create_transfer_operators
from solvers.spectral.operators.corner import create_corner_treatment


def get_erms(smoother, level):
    """Get ERMS = sqrt(sum(divergence^2) / N_inner) - paper's criterion."""
    smoother._compute_residuals(level.u, level.v, level.p)
    divergence = -level.R_p / smoother.beta_squared
    n_inner = level.shape_inner[0] * level.shape_inner[1]
    erms = np.sqrt(np.sum(divergence**2) / n_inner)
    return erms


def test_sg_subtraction(N, tolerance):
    """Single grid with subtraction method."""
    print(f"\n[SG N={N}] with subtraction corner treatment")

    basis = ChebyshevLobattoBasis(domain=(0.0, 1.0))
    corner = create_corner_treatment("subtraction")  # KEY: subtraction method!

    level = build_spectral_level(N, 0, basis, basis)
    sm = MultigridSmoother(
        level=level, Re=100.0, beta_squared=5.0, lid_velocity=1.0,
        CFL=2.5, corner_treatment=corner, Lx=1.0, Ly=1.0
    )
    sm.initialize_lid()

    t0 = time.time()
    for i in range(50000):
        sm.step()
        erms = get_erms(sm, level)
        if erms < tolerance:
            t_elapsed = time.time() - t0
            print(f"  Converged at iter {i+1}: ERMS={erms:.2e}, time={t_elapsed:.2f}s")
            return i + 1, t_elapsed
        if i % 1000 == 0 and i > 0:
            print(f"  Iter {i}: ERMS={erms:.2e}")

    t_elapsed = time.time() - t0
    print(f"  Did not converge after 50000 iters, ERMS={erms:.2e}")
    return 50000, t_elapsed


def test_sg_smoothing(N, tolerance):
    """Single grid with smoothing method (baseline)."""
    print(f"\n[SG N={N}] with smoothing corner treatment (baseline)")

    basis = ChebyshevLobattoBasis(domain=(0.0, 1.0))
    corner = create_corner_treatment("smoothing")

    level = build_spectral_level(N, 0, basis, basis)
    sm = MultigridSmoother(
        level=level, Re=100.0, beta_squared=5.0, lid_velocity=1.0,
        CFL=2.5, corner_treatment=corner, Lx=1.0, Ly=1.0
    )
    sm.initialize_lid()

    t0 = time.time()
    for i in range(50000):
        sm.step()
        erms = get_erms(sm, level)
        if erms < tolerance:
            t_elapsed = time.time() - t0
            print(f"  Converged at iter {i+1}: ERMS={erms:.2e}, time={t_elapsed:.2f}s")
            return i + 1, t_elapsed
        if i % 1000 == 0 and i > 0:
            print(f"  Iter {i}: ERMS={erms:.2e}")

    t_elapsed = time.time() - t0
    print(f"  Did not converge after 50000 iters, ERMS={erms:.2e}")
    return 50000, t_elapsed


def test_vmg_subtraction(N, n_levels, tolerance, max_cycles=500):
    """VMG with subtraction method."""
    print(f"\n[VMG N={N}, {n_levels} levels] with subtraction corner treatment")

    basis = ChebyshevLobattoBasis(domain=(0.0, 1.0))
    transfer_ops = create_transfer_operators("fft", "fft")
    corner = create_corner_treatment("subtraction")  # KEY: subtraction method!

    levels = build_hierarchy(N, n_levels, basis, basis)

    # Paper uses VMG-111 (1 step per level)
    pre_smoothing = [1] * n_levels

    t0 = time.time()
    _, iters, converged = solve_vmg(
        levels=levels,
        Re=100.0,
        beta_squared=5.0,
        lid_velocity=1.0,
        CFL=2.5,
        tolerance=tolerance,
        max_iterations=max_cycles,
        transfer_ops=transfer_ops,
        corner_treatment=corner,
        pre_smoothing=pre_smoothing,
        correction_damping=1.0,  # Paper uses full correction (no damping)
    )
    t_elapsed = time.time() - t0

    status = "converged" if converged else "NOT converged"
    print(f"  {status} at {iters} cycles, time={t_elapsed:.2f}s")
    return iters, t_elapsed, converged


def test_vmg_smoothing(N, n_levels, tolerance, max_cycles=500):
    """VMG with smoothing method (baseline)."""
    print(f"\n[VMG N={N}, {n_levels} levels] with smoothing corner treatment (baseline)")

    basis = ChebyshevLobattoBasis(domain=(0.0, 1.0))
    transfer_ops = create_transfer_operators("fft", "fft")
    corner = create_corner_treatment("smoothing")

    levels = build_hierarchy(N, n_levels, basis, basis)

    # Use more aggressive settings since smoothing doesn't work well with VMG
    pre_smoothing = [5, 2] if n_levels == 2 else [5, 3, 1]

    t0 = time.time()
    _, iters, converged = solve_vmg(
        levels=levels,
        Re=100.0,
        beta_squared=5.0,
        lid_velocity=1.0,
        CFL=2.5,
        tolerance=tolerance,
        max_iterations=max_cycles,
        transfer_ops=transfer_ops,
        corner_treatment=corner,
        pre_smoothing=pre_smoothing,
        correction_damping=0.5,  # Need damping for smoothing method
    )
    t_elapsed = time.time() - t0

    status = "converged" if converged else "NOT converged"
    print(f"  {status} at {iters} cycles, time={t_elapsed:.2f}s")
    return iters, t_elapsed, converged


def main():
    print("=" * 80)
    print("Comparing VMG with SUBTRACTION vs SMOOTHING corner treatment")
    print("Paper (Zhang & Xi 2010) uses subtraction method")
    print("=" * 80)

    tolerance = 1e-4  # Paper's criterion
    N = 48
    n_levels = 3  # 48 -> 24 -> 12

    print(f"\nSettings: N={N}, {n_levels} levels, tolerance={tolerance}")
    print(f"Paper settings: CFL=2.5, beta^2=5.0, Re=100")

    # Test single grid
    print("\n" + "=" * 80)
    print("SINGLE GRID COMPARISON")
    print("=" * 80)

    sg_sub_iters, sg_sub_time = test_sg_subtraction(N, tolerance)
    sg_smooth_iters, sg_smooth_time = test_sg_smoothing(N, tolerance)

    # Test VMG
    print("\n" + "=" * 80)
    print("VMG COMPARISON")
    print("=" * 80)

    vmg_sub_iters, vmg_sub_time, vmg_sub_conv = test_vmg_subtraction(N, n_levels, tolerance)
    vmg_smooth_iters, vmg_smooth_time, vmg_smooth_conv = test_vmg_smoothing(N, n_levels, tolerance)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY (tolerance = 1e-4)")
    print("=" * 80)
    print(f"\nSingle Grid:")
    print(f"  Subtraction: {sg_sub_iters:6d} iters, {sg_sub_time:.2f}s")
    print(f"  Smoothing:   {sg_smooth_iters:6d} iters, {sg_smooth_time:.2f}s")

    print(f"\nVMG ({n_levels} levels):")
    print(f"  Subtraction: {vmg_sub_iters:6d} cycles, {vmg_sub_time:.2f}s {'[PASS]' if vmg_sub_conv else '[FAIL]'}")
    print(f"  Smoothing:   {vmg_smooth_iters:6d} cycles, {vmg_smooth_time:.2f}s {'[PASS]' if vmg_smooth_conv else '[FAIL]'}")

    print(f"\nSpeedup (SG_subtraction / VMG_subtraction):")
    if vmg_sub_conv and sg_sub_iters > 0:
        speedup = sg_sub_iters / vmg_sub_iters
        time_speedup = sg_sub_time / vmg_sub_time if vmg_sub_time > 0 else float('inf')
        print(f"  Iteration speedup: {speedup:.1f}x")
        print(f"  Time speedup:      {time_speedup:.1f}x")
    else:
        print("  VMG with subtraction did not converge!")

    print(f"\nPaper reports ~26x speedup for 48-VMG-111 vs 48-SG")
    print("=" * 80)


if __name__ == "__main__":
    main()
