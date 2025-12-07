#!/usr/bin/env python
"""Final VMG test: compare iteration counts to single grid."""

import logging
import numpy as np
import sys
import time

logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, "src")

from solvers.spectral.basis.spectral import ChebyshevLobattoBasis
from solvers.spectral.multigrid.fsg import (
    build_hierarchy, solve_vmg, solve_fsg,
    build_spectral_level, MultigridSmoother
)
from solvers.spectral.operators.transfer_operators import create_transfer_operators
from solvers.spectral.operators.corner import create_corner_treatment


def test_single_grid(N, tolerance):
    """Run single grid solver."""
    basis = ChebyshevLobattoBasis(domain=(0.0, 1.0))
    corner = create_corner_treatment("smoothing")

    level = build_spectral_level(N, 0, basis, basis)
    sm = MultigridSmoother(
        level=level, Re=100.0, beta_squared=5.0, lid_velocity=1.0,
        CFL=2.0, corner_treatment=corner, Lx=1.0, Ly=1.0
    )
    sm.initialize_lid()

    t0 = time.time()
    for i in range(50000):
        u_res, v_res = sm.step()
        if max(u_res, v_res) < tolerance:
            return i + 1, time.time() - t0, True
    return 50000, time.time() - t0, False


def test_fsg(N, n_levels, tolerance):
    """Run FSG solver."""
    basis = ChebyshevLobattoBasis(domain=(0.0, 1.0))
    transfer_ops = create_transfer_operators("fft", "fft")
    corner = create_corner_treatment("smoothing")

    levels = build_hierarchy(N, n_levels, basis, basis)

    t0 = time.time()
    _, iters, converged = solve_fsg(
        levels=levels,
        Re=100.0,
        beta_squared=5.0,
        lid_velocity=1.0,
        CFL=2.0,
        tolerance=tolerance,
        max_iterations=50000,
        transfer_ops=transfer_ops,
        corner_treatment=corner,
    )
    return iters, time.time() - t0, converged


def test_vmg(N, n_levels, tolerance, max_cycles=500):
    """Run VMG solver."""
    basis = ChebyshevLobattoBasis(domain=(0.0, 1.0))
    transfer_ops = create_transfer_operators("fft", "fft")
    corner = create_corner_treatment("smoothing")

    levels = build_hierarchy(N, n_levels, basis, basis)

    # Use stable settings
    pre_smoothing = [5, 2] if n_levels == 2 else [5, 3, 1]

    t0 = time.time()
    _, iters, converged = solve_vmg(
        levels=levels,
        Re=100.0,
        beta_squared=5.0,
        lid_velocity=1.0,
        CFL=2.0,
        tolerance=tolerance,
        max_iterations=max_cycles,
        transfer_ops=transfer_ops,
        corner_treatment=corner,
        pre_smoothing=pre_smoothing,
        correction_damping=0.5,
    )
    return iters, time.time() - t0, converged


def main():
    print("=" * 80)
    print("VMG Final Comparison: SG vs FSG vs VMG")
    print("=" * 80)

    # Test with tolerance 1e-4 (VMG can achieve this)
    tolerances = [1e-3, 1e-4]

    for tol in tolerances:
        print(f"\n{'=' * 80}")
        print(f"Tolerance: {tol}")
        print("=" * 80)

        N = 24

        # Single Grid
        print(f"\n[N={N}] Single Grid...")
        sg_iters, sg_time, sg_conv = test_single_grid(N, tol)
        print(f"  SG:  iters={sg_iters:6d}, time={sg_time:.2f}s, converged={sg_conv}")

        # FSG (2 levels)
        print(f"\n[N={N}] FSG (24->12)...")
        fsg_iters, fsg_time, fsg_conv = test_fsg(N, 2, tol)
        print(f"  FSG: iters={fsg_iters:6d}, time={fsg_time:.2f}s, converged={fsg_conv}")

        # VMG (2 levels)
        print(f"\n[N={N}] VMG (24->12)...")
        vmg_iters, vmg_time, vmg_conv = test_vmg(N, 2, tol, max_cycles=500)
        print(f"  VMG: iters={vmg_iters:6d}, time={vmg_time:.2f}s, converged={vmg_conv}")

        print(f"\n  Summary for tol={tol}:")
        print(f"    SG:  {sg_iters:6d} iters {'[PASS]' if sg_conv else '[FAIL]'}")
        print(f"    FSG: {fsg_iters:6d} iters {'[PASS]' if fsg_conv else '[FAIL]'}")
        print(f"    VMG: {vmg_iters:6d} iters {'[PASS]' if vmg_conv else '[FAIL]'}")

        if sg_conv and fsg_conv:
            speedup = sg_iters / fsg_iters
            print(f"    FSG speedup vs SG: {speedup:.1f}x")
        if sg_conv and vmg_conv:
            speedup = sg_iters / vmg_iters
            print(f"    VMG speedup vs SG: {speedup:.1f}x")


if __name__ == "__main__":
    main()
