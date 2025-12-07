#!/usr/bin/env python
"""Test VMG with paper-appropriate grid configurations.

From Zhang & Xi (2010), Section 4.3 Table 3:
- Minimum coarsest N = 12
- Example configs: 48-VMG-111, 64-VMG-111, 96-VMG-111
- Smoothing "111" means 1 step on each level (faster than 123!)
- CFL = 2.5 works for N >= 12, smaller CFL needed for N < 12
"""

import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')

sys.path.insert(0, "src")

from solvers.spectral.basis.spectral import ChebyshevLobattoBasis
from solvers.spectral.multigrid.fsg import build_hierarchy, solve_vmg
from solvers.spectral.operators.transfer_operators import create_transfer_operators
from solvers.spectral.operators.corner import create_corner_treatment


def test_2level_vmg():
    """Test 2-level VMG with coarsest N=12 (like 24-VMG-11)."""
    print("=" * 70)
    print("Test: 2-level VMG (24->12) with paper configuration")
    print("=" * 70)

    basis = ChebyshevLobattoBasis(domain=(0.0, 1.0))
    transfer_ops = create_transfer_operators("fft", "fft")
    corner = create_corner_treatment("smoothing")

    # 2 levels: N=24 -> N=12 (coarsest = 12, per paper minimum)
    levels = build_hierarchy(24, 2, basis, basis)
    print(f"\nGrid sizes: {[lvl.n for lvl in levels]}")

    # Paper uses smoothing "11" for 2 levels (1 step each)
    _, iters, converged = solve_vmg(
        levels=levels,
        Re=100.0,
        beta_squared=5.0,
        lid_velocity=1.0,
        CFL=0.5,
        tolerance=1e-5,
        max_iterations=500,
        transfer_ops=transfer_ops,
        corner_treatment=corner,
        pre_smoothing=[1, 1],  # Paper: 11 means 1 step on each level
        post_smoothing=None,
        correction_damping=1.0,
    )
    print(f"\nResult: converged={converged}, iterations={iters}")
    return converged


def test_3level_vmg():
    """Test 3-level VMG with coarsest N=12 (like 48-VMG-111)."""
    print("\n" + "=" * 70)
    print("Test: 3-level VMG (48->24->12) with paper configuration")
    print("=" * 70)

    basis = ChebyshevLobattoBasis(domain=(0.0, 1.0))
    transfer_ops = create_transfer_operators("fft", "fft")
    corner = create_corner_treatment("smoothing")

    # 3 levels: N=48 -> N=24 -> N=12 (per paper Table 3)
    levels = build_hierarchy(48, 3, basis, basis)
    print(f"\nGrid sizes: {[lvl.n for lvl in levels]}")

    # Paper uses smoothing "111" (1 step each level - faster than 123!)
    _, iters, converged = solve_vmg(
        levels=levels,
        Re=100.0,
        beta_squared=5.0,
        lid_velocity=1.0,
        CFL=0.5,
        tolerance=1e-5,
        max_iterations=500,
        transfer_ops=transfer_ops,
        corner_treatment=corner,
        pre_smoothing=[1, 1, 1],  # Paper: 111 (1 step each level)
        post_smoothing=None,
        correction_damping=1.0,
    )
    print(f"\nResult: converged={converged}, iterations={iters}")
    return converged


def test_vmg_with_reduced_cfl_coarse():
    """Test VMG with smaller grid but reduced CFL for coarse levels.

    Paper says: with small N (N=4-10), CFL=0.3-0.5 needed for convergence.
    Try using adaptive CFL on coarse levels.
    """
    print("\n" + "=" * 70)
    print("Test: 2-level VMG (15->7) with standard config")
    print("(Note: N=7 is below paper minimum of 12)")
    print("=" * 70)

    basis = ChebyshevLobattoBasis(domain=(0.0, 1.0))
    transfer_ops = create_transfer_operators("fft", "fft")
    corner = create_corner_treatment("smoothing")

    levels = build_hierarchy(15, 2, basis, basis)
    print(f"\nGrid sizes: {[lvl.n for lvl in levels]}")
    print("Warning: Coarsest N=7 is below paper's minimum of N=12!")

    _, iters, converged = solve_vmg(
        levels=levels,
        Re=100.0,
        beta_squared=5.0,
        lid_velocity=1.0,
        CFL=0.5,  # Might need to reduce for N=7
        tolerance=1e-5,
        max_iterations=500,
        transfer_ops=transfer_ops,
        corner_treatment=corner,
        pre_smoothing=[1, 1],
        post_smoothing=None,
        correction_damping=1.0,
    )
    print(f"\nResult: converged={converged}, iterations={iters}")

    if not converged:
        print("\nAs expected, N=7 coarsest doesn't work well!")
        print("Paper uses minimum coarsest N=12.")
    return converged


def test_comparison_sg_vs_vmg():
    """Compare Single Grid vs VMG convergence at Re=100."""
    print("\n" + "=" * 70)
    print("Comparison: SG vs VMG for N=24 at Re=100")
    print("=" * 70)

    from solvers.spectral.multigrid.fsg import build_spectral_level, MultigridSmoother

    basis = ChebyshevLobattoBasis(domain=(0.0, 1.0))
    transfer_ops = create_transfer_operators("fft", "fft")
    corner = create_corner_treatment("smoothing")

    params = dict(Re=100.0, beta_squared=5.0, lid_velocity=1.0, CFL=0.5,
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
        CFL=0.5,
        tolerance=1e-5,
        max_iterations=500,
        transfer_ops=transfer_ops,
        corner_treatment=corner,
        pre_smoothing=[1, 1],
        post_smoothing=None,
        correction_damping=1.0,
    )
    print(f"  VMG converged={converged}, total iterations={vmg_iters}")

    print("\n" + "=" * 70)
    print(f"Summary: SG={sg_iters} iters, VMG={vmg_iters} iters")
    if vmg_iters > 0 and sg_iters > 0:
        speedup = sg_iters / vmg_iters
        print(f"Speedup factor: {speedup:.1f}x")
    print("=" * 70)


if __name__ == "__main__":
    # Test with paper-appropriate configurations
    success_2level = test_2level_vmg()
    success_3level = test_3level_vmg()

    # Also test the "too small" configuration to demonstrate the issue
    test_vmg_with_reduced_cfl_coarse()

    # Compare SG vs VMG
    if success_2level:
        test_comparison_sg_vs_vmg()

    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  2-level VMG (24->12): {'PASS' if success_2level else 'FAIL'}")
    print(f"  3-level VMG (48->24->12): {'PASS' if success_3level else 'FAIL'}")
    print("=" * 70)
