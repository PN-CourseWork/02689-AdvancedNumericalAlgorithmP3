#!/usr/bin/env python
"""Test VMG with fixed settings (level-specific CFL, damping)."""

import logging
import numpy as np
import sys

logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
sys.path.insert(0, "src")

from solvers.spectral.basis.spectral import ChebyshevLobattoBasis
from solvers.spectral.multigrid.fsg import build_hierarchy, solve_vmg
from solvers.spectral.operators.transfer_operators import create_transfer_operators
from solvers.spectral.operators.corner import create_corner_treatment


def test_vmg_2level():
    """Test 2-level VMG (24->12)."""
    print("=" * 70)
    print("Test: 2-level VMG (24->12) with level-specific CFL")
    print("=" * 70)

    basis = ChebyshevLobattoBasis(domain=(0.0, 1.0))
    transfer_ops = create_transfer_operators("fft", "fft")
    corner = create_corner_treatment("smoothing")

    levels = build_hierarchy(24, 2, basis, basis)
    print(f"Grid sizes: {[lvl.n for lvl in levels]}")

    # CFL will be automatically scaled per level (coarsest gets smallest)
    _, iters, converged = solve_vmg(
        levels=levels,
        Re=100.0,
        beta_squared=5.0,
        lid_velocity=1.0,
        CFL=2.0,  # Finest level CFL (will be halved for coarse)
        tolerance=1e-5,
        max_iterations=100,
        transfer_ops=transfer_ops,
        corner_treatment=corner,
        pre_smoothing=[5, 2],  # [coarsest, finest]
        post_smoothing=None,
        correction_damping=0.5,
    )

    print(f"\nResult: converged={converged}, iterations={iters}")
    return converged


def test_vmg_3level():
    """Test 3-level VMG (48->24->12)."""
    print("\n" + "=" * 70)
    print("Test: 3-level VMG (48->24->12) with level-specific CFL")
    print("=" * 70)

    basis = ChebyshevLobattoBasis(domain=(0.0, 1.0))
    transfer_ops = create_transfer_operators("fft", "fft")
    corner = create_corner_treatment("smoothing")

    levels = build_hierarchy(48, 3, basis, basis)
    print(f"Grid sizes: {[lvl.n for lvl in levels]}")

    # CFL will be automatically scaled per level
    _, iters, converged = solve_vmg(
        levels=levels,
        Re=100.0,
        beta_squared=5.0,
        lid_velocity=1.0,
        CFL=2.0,  # Finest level CFL
        tolerance=1e-5,
        max_iterations=100,
        transfer_ops=transfer_ops,
        corner_treatment=corner,
        pre_smoothing=[5, 3, 1],  # [coarsest, intermediate, finest]
        post_smoothing=None,
        correction_damping=0.5,
    )

    print(f"\nResult: converged={converged}, iterations={iters}")
    return converged


def compare_sg_vs_vmg():
    """Compare single grid vs VMG for N=24."""
    print("\n" + "=" * 70)
    print("Comparison: SG vs VMG for N=24 at Re=100")
    print("=" * 70)

    from solvers.spectral.multigrid.fsg import build_spectral_level, MultigridSmoother

    basis = ChebyshevLobattoBasis(domain=(0.0, 1.0))
    transfer_ops = create_transfer_operators("fft", "fft")
    corner = create_corner_treatment("smoothing")

    params = dict(Re=100.0, beta_squared=5.0, lid_velocity=1.0, CFL=2.0,
                  corner_treatment=corner, Lx=1.0, Ly=1.0)

    # Single grid
    print("\nTest A: Single Grid (N=24)")
    sg_level = build_spectral_level(24, 0, basis, basis)
    sg_sm = MultigridSmoother(level=sg_level, **params)
    sg_sm.initialize_lid()

    sg_iters = 0
    for i in range(20000):
        u_res, v_res = sg_sm.step()
        sg_iters += 1
        if max(u_res, v_res) < 1e-5:
            print(f"  SG converged at iter {sg_iters}: res={max(u_res, v_res):.2e}")
            break
        if i % 2000 == 0:
            print(f"  Iter {i}: res={max(u_res, v_res):.2e}")

    # VMG
    print("\nTest B: VMG (24->12, 2 levels)")
    levels = build_hierarchy(24, 2, basis, basis)
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
        pre_smoothing=[5, 2],
        post_smoothing=None,
        correction_damping=0.5,
    )
    print(f"  VMG converged={converged}, total iterations={vmg_iters}")

    print("\n" + "=" * 70)
    print(f"Summary: SG={sg_iters} iters, VMG={vmg_iters} iters")
    if vmg_iters > 0 and sg_iters > 0:
        speedup = sg_iters / vmg_iters
        print(f"Speedup factor: {speedup:.1f}x")
    print("=" * 70)


if __name__ == "__main__":
    success_2level = test_vmg_2level()
    success_3level = test_vmg_3level()

    if success_2level:
        compare_sg_vs_vmg()

    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  2-level VMG (24->12): {'PASS' if success_2level else 'FAIL'}")
    print(f"  3-level VMG (48->24->12): {'PASS' if success_3level else 'FAIL'}")
    print("=" * 70)
