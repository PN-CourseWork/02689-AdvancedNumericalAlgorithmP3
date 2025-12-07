#!/usr/bin/env python
"""Quick VMG test after boundary fix."""

import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')

sys.path.insert(0, "src")

from solvers.spectral.basis.spectral import ChebyshevLobattoBasis
from solvers.spectral.multigrid.fsg import build_hierarchy, solve_vmg
from solvers.spectral.operators.transfer_operators import create_transfer_operators
from solvers.spectral.operators.corner import create_corner_treatment


def test_vmg():
    """Test VMG solver with paper settings."""
    print("="*60)
    print("Testing VMG with tau boundary fix")
    print("="*60)

    basis = ChebyshevLobattoBasis(domain=(0.0, 1.0))
    transfer_ops = create_transfer_operators("fft", "fft")
    corner = create_corner_treatment("smoothing")

    # Test 1: 2 levels (N=7->15) - like paper 64-VMG-11
    print("\nTest 1: 2 levels (7->15), damping=1.0")
    levels = build_hierarchy(15, 2, basis, basis)
    _, iters, converged = solve_vmg(
        levels=levels,
        Re=100.0,
        beta_squared=5.0,
        lid_velocity=1.0,
        CFL=0.5,
        tolerance=1e-5,
        max_iterations=200,
        transfer_ops=transfer_ops,
        corner_treatment=corner,
        pre_smoothing=[2, 1],  # Paper: 11 for 2 levels
        post_smoothing=None,
        correction_damping=1.0,
    )
    print(f"  Result: converged={converged}, iterations={iters}")

    # Test 2: 2 levels with damping=0.5
    print("\nTest 2: 2 levels (7->15), damping=0.5")
    levels = build_hierarchy(15, 2, basis, basis)
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
        pre_smoothing=[2, 1],
        post_smoothing=None,
        correction_damping=0.5,
    )

    print(f"\nResult: converged={converged}, iterations={iters}")
    if converged:
        print("✓ VMG converged!")
    else:
        print("⚠ VMG did not converge in 500 V-cycles")


if __name__ == "__main__":
    test_vmg()
