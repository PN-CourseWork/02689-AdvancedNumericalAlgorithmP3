#!/usr/bin/env python
"""Fast VMG comparison: smoothing vs subtraction corner treatment."""

import logging
import numpy as np
import sys
import time

logging.basicConfig(level=logging.INFO)
sys.path.insert(0, "src")

from solvers.spectral.basis.spectral import ChebyshevLobattoBasis
from solvers.spectral.multigrid.fsg import (
    build_hierarchy, build_spectral_level, MultigridSmoother, solve_vmg
)
from solvers.spectral.operators.transfer_operators import create_transfer_operators
from solvers.spectral.operators.corner import create_corner_treatment


def test_vmg(N, n_levels, corner_method, pre_smoothing, damping, tolerance=1e-4, max_cycles=200):
    """Run VMG and return iterations and time."""
    basis = ChebyshevLobattoBasis(domain=(0.0, 1.0))
    transfer_ops = create_transfer_operators("fft", "fft")
    corner = create_corner_treatment(corner_method)

    levels = build_hierarchy(N, n_levels, basis, basis)

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
        correction_damping=damping,
    )
    t_elapsed = time.time() - t0

    return iters, t_elapsed, converged


def test_sg(N, corner_method, tolerance=1e-4, max_iters=10000):
    """Run single grid and return iterations and time."""
    basis = ChebyshevLobattoBasis(domain=(0.0, 1.0))
    corner = create_corner_treatment(corner_method)

    level = build_spectral_level(N, 0, basis, basis)
    sm = MultigridSmoother(
        level=level, Re=100.0, beta_squared=5.0, lid_velocity=1.0,
        CFL=2.5, corner_treatment=corner, Lx=1.0, Ly=1.0
    )
    sm.initialize_lid()

    u_prev = level.u.copy()
    v_prev = level.v.copy()

    t0 = time.time()
    for i in range(max_iters):
        sm.step()

        u_change = np.linalg.norm(level.u - u_prev) / (np.linalg.norm(u_prev) + 1e-12)
        v_change = np.linalg.norm(level.v - v_prev) / (np.linalg.norm(v_prev) + 1e-12)
        u_prev[:] = level.u
        v_prev[:] = level.v

        if max(u_change, v_change) < tolerance:
            t_elapsed = time.time() - t0
            return i + 1, t_elapsed, True

    t_elapsed = time.time() - t0
    return max_iters, t_elapsed, False


def main():
    print("=" * 70)
    print("VMG Comparison: Smoothing vs Subtraction Corner Treatment")
    print("=" * 70)

    N = 24
    n_levels = 2
    tolerance = 1e-4

    print(f"\nSettings: N={N}, {n_levels} levels, tolerance={tolerance}")

    # Test Single Grid
    print("\n" + "-" * 70)
    print("SINGLE GRID (baseline)")
    print("-" * 70)

    print("\n[SG] Smoothing corner:")
    sg_smooth_iters, sg_smooth_time, sg_smooth_conv = test_sg(N, "smoothing", tolerance)
    print(f"  {sg_smooth_iters} iters, {sg_smooth_time:.2f}s, conv={sg_smooth_conv}")

    print("\n[SG] Subtraction corner:")
    sg_sub_iters, sg_sub_time, sg_sub_conv = test_sg(N, "subtraction", tolerance)
    print(f"  {sg_sub_iters} iters, {sg_sub_time:.2f}s, conv={sg_sub_conv}")

    # Test VMG with different settings
    print("\n" + "-" * 70)
    print("VMG - Paper settings (VMG-11, damping=1.0)")
    print("-" * 70)

    print("\n[VMG] Smoothing corner:")
    vmg_smooth_iters, vmg_smooth_time, vmg_smooth_conv = test_vmg(
        N, n_levels, "smoothing", [1, 1], 1.0, tolerance
    )
    print(f"  {vmg_smooth_iters} cycles, {vmg_smooth_time:.2f}s, conv={vmg_smooth_conv}")

    print("\n[VMG] Subtraction corner:")
    vmg_sub_iters, vmg_sub_time, vmg_sub_conv = test_vmg(
        N, n_levels, "subtraction", [1, 1], 1.0, tolerance
    )
    print(f"  {vmg_sub_iters} cycles, {vmg_sub_time:.2f}s, conv={vmg_sub_conv}")

    # Test VMG with our stabilized settings
    print("\n" + "-" * 70)
    print("VMG - Our settings ([5,2], damping=0.5)")
    print("-" * 70)

    print("\n[VMG] Smoothing corner:")
    vmg_smooth2_iters, vmg_smooth2_time, vmg_smooth2_conv = test_vmg(
        N, n_levels, "smoothing", [5, 2], 0.5, tolerance
    )
    print(f"  {vmg_smooth2_iters} cycles, {vmg_smooth2_time:.2f}s, conv={vmg_smooth2_conv}")

    print("\n[VMG] Subtraction corner:")
    vmg_sub2_iters, vmg_sub2_time, vmg_sub2_conv = test_vmg(
        N, n_levels, "subtraction", [5, 2], 0.5, tolerance
    )
    print(f"  {vmg_sub2_iters} cycles, {vmg_sub2_time:.2f}s, conv={vmg_sub2_conv}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Method':<35} {'Iters':<10} {'Time':<10} {'Conv':<8}")
    print("-" * 70)
    print(f"{'SG + smoothing':<35} {sg_smooth_iters:<10} {sg_smooth_time:<10.2f} {str(sg_smooth_conv):<8}")
    print(f"{'SG + subtraction':<35} {sg_sub_iters:<10} {sg_sub_time:<10.2f} {str(sg_sub_conv):<8}")
    print(f"{'VMG-11 + smoothing (d=1.0)':<35} {vmg_smooth_iters:<10} {vmg_smooth_time:<10.2f} {str(vmg_smooth_conv):<8}")
    print(f"{'VMG-11 + subtraction (d=1.0)':<35} {vmg_sub_iters:<10} {vmg_sub_time:<10.2f} {str(vmg_sub_conv):<8}")
    print(f"{'VMG-52 + smoothing (d=0.5)':<35} {vmg_smooth2_iters:<10} {vmg_smooth2_time:<10.2f} {str(vmg_smooth2_conv):<8}")
    print(f"{'VMG-52 + subtraction (d=0.5)':<35} {vmg_sub2_iters:<10} {vmg_sub2_time:<10.2f} {str(vmg_sub2_conv):<8}")

    # Speedup calculation
    print("\nSpeedups (SG iters / VMG cycles):")
    if vmg_smooth_conv and sg_smooth_conv:
        print(f"  VMG-11 + smoothing: {sg_smooth_iters / vmg_smooth_iters:.1f}x")
    if vmg_sub_conv and sg_sub_conv:
        print(f"  VMG-11 + subtraction: {sg_sub_iters / vmg_sub_iters:.1f}x")
    if vmg_smooth2_conv and sg_smooth_conv:
        print(f"  VMG-52 + smoothing: {sg_smooth_iters / vmg_smooth2_iters:.1f}x")
    if vmg_sub2_conv and sg_sub_conv:
        print(f"  VMG-52 + subtraction: {sg_sub_iters / vmg_sub2_iters:.1f}x")


if __name__ == "__main__":
    main()
