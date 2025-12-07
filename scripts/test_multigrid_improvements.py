#!/usr/bin/env python
"""Test different multigrid improvement strategies."""

import logging
import numpy as np
import sys
import time

logging.basicConfig(level=logging.INFO)
sys.path.insert(0, "src")

from solvers.spectral.basis.spectral import ChebyshevLobattoBasis
from solvers.spectral.multigrid.fsg import (
    build_hierarchy, build_spectral_level, MultigridSmoother,
    solve_fsg, prolongate_solution, restrict_solution
)
from solvers.spectral.operators.transfer_operators import create_transfer_operators
from solvers.spectral.operators.corner import create_corner_treatment


def test_single_grid(N, tolerance=1e-4, max_iters=50000):
    """Baseline single grid solver."""
    basis = ChebyshevLobattoBasis(domain=(0.0, 1.0))
    corner = create_corner_treatment("smoothing")

    level = build_spectral_level(N, 0, basis, basis)
    sm = MultigridSmoother(
        level=level, Re=100.0, beta_squared=5.0, lid_velocity=1.0,
        CFL=2.5, corner_treatment=corner, Lx=1.0, Ly=1.0
    )
    sm.initialize_lid()

    t0 = time.time()
    for i in range(max_iters):
        u_res, v_res = sm.step()
        if max(u_res, v_res) < tolerance:
            return i + 1, time.time() - t0, True
    return max_iters, time.time() - t0, False


def test_fsg(N, n_levels, tolerance=1e-4, max_iters=50000):
    """Standard FSG solver."""
    basis = ChebyshevLobattoBasis(domain=(0.0, 1.0))
    transfer_ops = create_transfer_operators("fft", "fft")
    corner = create_corner_treatment("smoothing")

    levels = build_hierarchy(N, n_levels, basis, basis)

    t0 = time.time()
    _, iters, converged = solve_fsg(
        levels=levels, Re=100.0, beta_squared=5.0, lid_velocity=1.0,
        CFL=2.5, tolerance=tolerance, max_iterations=max_iters,
        transfer_ops=transfer_ops, corner_treatment=corner
    )
    t_elapsed = time.time() - t0
    return iters, t_elapsed, converged


def test_wfsg(N, n_levels, tolerance=1e-4, max_iters=50000, cycles=3):
    """W-cycle FSG: Multiple coarse-to-fine passes with partial convergence."""
    basis = ChebyshevLobattoBasis(domain=(0.0, 1.0))
    transfer_ops = create_transfer_operators("fft", "fft")
    corner = create_corner_treatment("smoothing")

    levels = build_hierarchy(N, n_levels, basis, basis)

    # Create smoothers for all levels
    smoothers = []
    for level in levels:
        sm = MultigridSmoother(
            level=level, Re=100.0, beta_squared=5.0, lid_velocity=1.0,
            CFL=2.5, corner_treatment=corner, Lx=1.0, Ly=1.0
        )
        sm.initialize_lid()
        smoothers.append(sm)

    total_iters = 0
    t0 = time.time()

    # Initialize coarsest level
    levels[0].u[:] = 0.0
    levels[0].v[:] = 0.0
    levels[0].p[:] = 0.0

    # W-cycle: multiple coarse-to-fine sweeps
    for cycle in range(cycles):
        cycle_tolerance = tolerance * (10 ** (cycles - cycle - 1))  # Tighter each cycle
        cycle_tolerance = max(cycle_tolerance, tolerance)

        for level_idx in range(n_levels):
            level = levels[level_idx]
            smoother = smoothers[level_idx]

            # Prolongate from previous level (except first)
            if level_idx > 0:
                prolongate_solution(levels[level_idx - 1], level, transfer_ops)

            # Solve on this level to cycle tolerance
            iters_per_level = max_iters // (n_levels * cycles)
            for i in range(iters_per_level):
                u_res, v_res = smoother.step()
                total_iters += 1

                if max(u_res, v_res) < cycle_tolerance:
                    break

    # Final fine-grid solve to full tolerance
    finest_smoother = smoothers[-1]
    for i in range(max_iters):
        u_res, v_res = finest_smoother.step()
        total_iters += 1
        if max(u_res, v_res) < tolerance:
            return total_iters, time.time() - t0, True

    return total_iters, time.time() - t0, False


def test_cascadic(N, n_levels, tolerance=1e-4, max_iters=50000):
    """Cascadic MG: More aggressive coarse solving, loose fine tolerance initially."""
    basis = ChebyshevLobattoBasis(domain=(0.0, 1.0))
    transfer_ops = create_transfer_operators("fft", "fft")
    corner = create_corner_treatment("smoothing")

    levels = build_hierarchy(N, n_levels, basis, basis)

    # Create smoothers for all levels
    smoothers = []
    for level in levels:
        sm = MultigridSmoother(
            level=level, Re=100.0, beta_squared=5.0, lid_velocity=1.0,
            CFL=2.5, corner_treatment=corner, Lx=1.0, Ly=1.0
        )
        sm.initialize_lid()
        smoothers.append(sm)

    total_iters = 0
    t0 = time.time()

    # Initialize coarsest level
    levels[0].u[:] = 0.0
    levels[0].v[:] = 0.0
    levels[0].p[:] = 0.0

    # Cascadic: solve coarse very accurately, then progressively less on finer levels
    for level_idx in range(n_levels):
        level = levels[level_idx]
        smoother = smoothers[level_idx]
        is_finest = level_idx == n_levels - 1

        # Prolongate from previous level (except first)
        if level_idx > 0:
            prolongate_solution(levels[level_idx - 1], level, transfer_ops)

        # Coarse levels: very tight tolerance (they're cheap)
        # Fine levels: use standard tolerance
        if is_finest:
            level_tol = tolerance
        else:
            # Coarse levels: solve more accurately
            level_tol = tolerance / 10

        # Solve on this level
        for i in range(max_iters):
            u_res, v_res = smoother.step()
            total_iters += 1

            if max(u_res, v_res) < level_tol:
                break

    t_elapsed = time.time() - t0
    converged = max(u_res, v_res) < tolerance
    return total_iters, t_elapsed, converged


def test_recycling_fsg(N, n_levels, tolerance=1e-4, max_iters=50000, recycles=2):
    """FSG with solution recycling: periodically re-coarsen and re-solve."""
    basis = ChebyshevLobattoBasis(domain=(0.0, 1.0))
    transfer_ops = create_transfer_operators("fft", "fft")
    corner = create_corner_treatment("smoothing")

    levels = build_hierarchy(N, n_levels, basis, basis)

    # Create smoothers for all levels
    smoothers = []
    for level in levels:
        sm = MultigridSmoother(
            level=level, Re=100.0, beta_squared=5.0, lid_velocity=1.0,
            CFL=2.5, corner_treatment=corner, Lx=1.0, Ly=1.0
        )
        sm.initialize_lid()
        smoothers.append(sm)

    total_iters = 0
    t0 = time.time()

    # Initialize coarsest level
    levels[0].u[:] = 0.0
    levels[0].v[:] = 0.0
    levels[0].p[:] = 0.0

    for recycle in range(recycles + 1):
        # Standard coarse-to-fine sweep
        for level_idx in range(n_levels):
            level = levels[level_idx]
            smoother = smoothers[level_idx]
            is_finest = level_idx == n_levels - 1

            # Prolongate from previous level (except on first sweep for coarsest)
            if level_idx > 0 or recycle > 0:
                if level_idx > 0:
                    prolongate_solution(levels[level_idx - 1], level, transfer_ops)
                elif recycle > 0:
                    # For coarsest on recycle: restrict from finest
                    restrict_solution(levels[-1], levels[0], transfer_ops)

            # Partial convergence on non-finest levels
            if not is_finest:
                level_tol = tolerance * 10  # Looser tolerance for intermediate
            else:
                level_tol = tolerance

            # Solve on this level
            for i in range(max_iters // ((recycles + 1) * n_levels)):
                u_res, v_res = smoother.step()
                total_iters += 1

                if max(u_res, v_res) < level_tol:
                    break

        # Check if finest converged
        if max(u_res, v_res) < tolerance:
            return total_iters, time.time() - t0, True

    # Final push on finest level
    finest_smoother = smoothers[-1]
    for i in range(max_iters):
        u_res, v_res = finest_smoother.step()
        total_iters += 1
        if max(u_res, v_res) < tolerance:
            return total_iters, time.time() - t0, True

    return total_iters, time.time() - t0, False


def test_optimized_recycling(N, n_levels, tolerance=1e-4, max_iters=50000):
    """Optimized recycling FSG with tuned parameters."""
    basis = ChebyshevLobattoBasis(domain=(0.0, 1.0))
    transfer_ops = create_transfer_operators("fft", "fft")
    corner = create_corner_treatment("smoothing")

    levels = build_hierarchy(N, n_levels, basis, basis)

    # Create smoothers for all levels
    smoothers = []
    for level in levels:
        sm = MultigridSmoother(
            level=level, Re=100.0, beta_squared=5.0, lid_velocity=1.0,
            CFL=2.5, corner_treatment=corner, Lx=1.0, Ly=1.0
        )
        sm.initialize_lid()
        smoothers.append(sm)

    total_iters = 0
    t0 = time.time()

    # Initialize coarsest level
    levels[0].u[:] = 0.0
    levels[0].v[:] = 0.0
    levels[0].p[:] = 0.0

    # Phase 1: Initial coarse-to-fine sweep with tight coarse tolerance
    for level_idx in range(n_levels - 1):  # Don't fully solve finest yet
        level = levels[level_idx]
        smoother = smoothers[level_idx]

        if level_idx > 0:
            prolongate_solution(levels[level_idx - 1], level, transfer_ops)

        # Tight tolerance on coarse levels (cheap iterations)
        level_tol = tolerance / 5

        for i in range(max_iters // 3):
            u_res, v_res = smoother.step()
            total_iters += 1
            if max(u_res, v_res) < level_tol:
                break

    # Prolongate to finest
    prolongate_solution(levels[-2], levels[-1], transfer_ops)

    # Phase 2: Iterative recycling - partial fine solve, recycle, repeat
    finest_smoother = smoothers[-1]
    coarse_smoother = smoothers[0]

    for recycle in range(4):
        # Partial fine grid solve
        target_iters = 500 if recycle < 3 else max_iters
        for i in range(target_iters):
            u_res, v_res = finest_smoother.step()
            total_iters += 1
            if max(u_res, v_res) < tolerance:
                return total_iters, time.time() - t0, True

        # Restrict to coarsest and re-solve
        if recycle < 3:
            restrict_solution(levels[-1], levels[0], transfer_ops)
            for i in range(200):  # Quick coarse refresh
                coarse_smoother.step()
                total_iters += 1

            # Re-propagate up
            for level_idx in range(1, n_levels):
                prolongate_solution(levels[level_idx - 1], levels[level_idx], transfer_ops)
                sm = smoothers[level_idx]
                for j in range(100):
                    sm.step()
                    total_iters += 1

    return total_iters, time.time() - t0, False


def main():
    print("=" * 70)
    print("Multigrid Improvement Strategies Comparison")
    print("=" * 70)

    N = 24
    n_levels = 3
    tolerance = 1e-4

    print(f"\nSettings: N={N}, {n_levels} levels, tolerance={tolerance}")

    # Test Single Grid baseline
    print("\n" + "-" * 70)
    print("SINGLE GRID (baseline)")
    print("-" * 70)
    sg_iters, sg_time, sg_conv = test_single_grid(N, tolerance)
    print(f"  {sg_iters} iterations, {sg_time:.2f}s, converged={sg_conv}")

    # Test standard FSG
    print("\n" + "-" * 70)
    print("STANDARD FSG")
    print("-" * 70)
    fsg_iters, fsg_time, fsg_conv = test_fsg(N, n_levels, tolerance)
    print(f"  {fsg_iters} iterations, {fsg_time:.2f}s, converged={fsg_conv}")
    if fsg_conv and sg_conv:
        print(f"  Speedup: {sg_iters / fsg_iters:.2f}x (iters), {sg_time / fsg_time:.2f}x (time)")

    # Test W-cycle FSG
    print("\n" + "-" * 70)
    print("W-CYCLE FSG (3 cycles)")
    print("-" * 70)
    wfsg_iters, wfsg_time, wfsg_conv = test_wfsg(N, n_levels, tolerance, cycles=3)
    print(f"  {wfsg_iters} iterations, {wfsg_time:.2f}s, converged={wfsg_conv}")
    if wfsg_conv and sg_conv:
        print(f"  Speedup vs SG: {sg_iters / wfsg_iters:.2f}x (iters)")
    if wfsg_conv and fsg_conv:
        print(f"  Speedup vs FSG: {fsg_iters / wfsg_iters:.2f}x (iters)")

    # Test Cascadic MG
    print("\n" + "-" * 70)
    print("CASCADIC MG (tight coarse tolerance)")
    print("-" * 70)
    cmg_iters, cmg_time, cmg_conv = test_cascadic(N, n_levels, tolerance)
    print(f"  {cmg_iters} iterations, {cmg_time:.2f}s, converged={cmg_conv}")
    if cmg_conv and sg_conv:
        print(f"  Speedup vs SG: {sg_iters / cmg_iters:.2f}x (iters)")
    if cmg_conv and fsg_conv:
        print(f"  Speedup vs FSG: {fsg_iters / cmg_iters:.2f}x (iters)")

    # Test Recycling FSG
    print("\n" + "-" * 70)
    print("RECYCLING FSG (2 recycles)")
    print("-" * 70)
    rfsg_iters, rfsg_time, rfsg_conv = test_recycling_fsg(N, n_levels, tolerance, recycles=2)
    print(f"  {rfsg_iters} iterations, {rfsg_time:.2f}s, converged={rfsg_conv}")
    if rfsg_conv and sg_conv:
        print(f"  Speedup vs SG: {sg_iters / rfsg_iters:.2f}x (iters)")
    if rfsg_conv and fsg_conv:
        print(f"  Speedup vs FSG: {fsg_iters / rfsg_iters:.2f}x (iters)")

    # Test Optimized Recycling FSG
    print("\n" + "-" * 70)
    print("OPTIMIZED RECYCLING FSG")
    print("-" * 70)
    orfsg_iters, orfsg_time, orfsg_conv = test_optimized_recycling(N, n_levels, tolerance)
    print(f"  {orfsg_iters} iterations, {orfsg_time:.2f}s, converged={orfsg_conv}")
    if orfsg_conv and sg_conv:
        print(f"  Speedup vs SG: {sg_iters / orfsg_iters:.2f}x (iters)")
    if orfsg_conv and fsg_conv:
        print(f"  Speedup vs FSG: {fsg_iters / orfsg_iters:.2f}x (iters)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Method':<30} {'Iters':<10} {'Time':<10} {'Speedup vs SG':<15}")
    print("-" * 70)
    print(f"{'Single Grid':<30} {sg_iters:<10} {sg_time:<10.2f} {'1.00x':<15}")
    if fsg_conv:
        print(f"{'Standard FSG':<30} {fsg_iters:<10} {fsg_time:<10.2f} {sg_iters/fsg_iters:<15.2f}x")
    if wfsg_conv:
        print(f"{'W-cycle FSG':<30} {wfsg_iters:<10} {wfsg_time:<10.2f} {sg_iters/wfsg_iters:<15.2f}x")
    if cmg_conv:
        print(f"{'Cascadic MG':<30} {cmg_iters:<10} {cmg_time:<10.2f} {sg_iters/cmg_iters:<15.2f}x")
    if rfsg_conv:
        print(f"{'Recycling FSG':<30} {rfsg_iters:<10} {rfsg_time:<10.2f} {sg_iters/rfsg_iters:<15.2f}x")
    if orfsg_conv:
        print(f"{'Optimized Recycling FSG':<30} {orfsg_iters:<10} {orfsg_time:<10.2f} {sg_iters/orfsg_iters:<15.2f}x")


if __name__ == "__main__":
    main()
