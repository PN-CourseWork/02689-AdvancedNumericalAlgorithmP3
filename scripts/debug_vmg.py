#!/usr/bin/env python
"""Debug script for VMG solver.

Tests each component of the VMG solver independently to identify issues.

Components tested:
1. MultigridSmoother on a single level
2. FAS tau correction computation
3. V-cycle correction application
4. Convergence behavior comparison with SG/FSG
"""

import logging
import numpy as np
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
log = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, "src")

from solvers.spectral.basis.spectral import ChebyshevLobattoBasis
from solvers.spectral.multigrid.fsg import (
    SpectralLevel,
    build_spectral_level,
    build_hierarchy,
    MultigridSmoother,
    prolongate_solution,
    restrict_solution,
    restrict_residual,
    solve_fsg,
    solve_vmg,
)
from solvers.spectral.operators.transfer_operators import (
    create_transfer_operators,
    InjectionRestriction,
)
from solvers.spectral.operators.corner import create_corner_treatment


def create_test_setup(n_fine=15, n_levels=3):
    """Create standard test setup."""
    basis_x = ChebyshevLobattoBasis(domain=(0.0, 1.0))
    basis_y = ChebyshevLobattoBasis(domain=(0.0, 1.0))

    levels = build_hierarchy(
        n_fine=n_fine,
        n_levels=n_levels,
        basis_x=basis_x,
        basis_y=basis_y,
        Lx=1.0,
        Ly=1.0,
    )

    transfer_ops = create_transfer_operators(
        prolongation_method="fft",
        restriction_method="fft",
    )

    corner_treatment = create_corner_treatment(method="smoothing")

    return levels, transfer_ops, corner_treatment


def test_smoother_single_level():
    """Test 1: Verify MultigridSmoother converges on a single level."""
    print("\n" + "="*60)
    print("TEST 1: MultigridSmoother on single level")
    print("="*60)

    basis_x = ChebyshevLobattoBasis(domain=(0.0, 1.0))
    basis_y = ChebyshevLobattoBasis(domain=(0.0, 1.0))
    corner_treatment = create_corner_treatment(method="smoothing")

    for n in [7, 15]:
        level = build_spectral_level(
            n=n,
            level_idx=0,
            basis_x=basis_x,
            basis_y=basis_y,
            Lx=1.0,
            Ly=1.0,
        )

        smoother = MultigridSmoother(
            level=level,
            Re=100.0,
            beta_squared=5.0,
            lid_velocity=1.0,
            CFL=0.5,
            corner_treatment=corner_treatment,
            Lx=1.0,
            Ly=1.0,
        )
        smoother.initialize_lid()

        # Run 500 iterations and track residual
        residuals = []
        for i in range(500):
            u_res, v_res = smoother.step()
            if i % 100 == 0:
                residuals.append((i, max(u_res, v_res)))

        print(f"\n  N={n}: Residual evolution:")
        for i, res in residuals:
            print(f"    Iter {i:4d}: residual = {res:.2e}")

        final_res = max(u_res, v_res)
        print(f"    Final (iter 500): residual = {final_res:.2e}")

        # Check if residual is decreasing
        if residuals[-1][1] < residuals[0][1]:
            print(f"  ✓ Residual decreasing on N={n}")
        else:
            print(f"  ✗ Residual NOT decreasing on N={n} - PROBLEM!")


def test_tau_correction():
    """Test 2: Verify FAS tau correction computation."""
    print("\n" + "="*60)
    print("TEST 2: FAS Tau Correction Computation")
    print("="*60)

    levels, transfer_ops, corner_treatment = create_test_setup(n_fine=15, n_levels=2)

    coarse_level = levels[0]
    fine_level = levels[1]

    # Create smoothers
    coarse_smoother = MultigridSmoother(
        level=coarse_level,
        Re=100.0, beta_squared=5.0, lid_velocity=1.0, CFL=0.5,
        corner_treatment=corner_treatment, Lx=1.0, Ly=1.0,
    )
    fine_smoother = MultigridSmoother(
        level=fine_level,
        Re=100.0, beta_squared=5.0, lid_velocity=1.0, CFL=0.5,
        corner_treatment=corner_treatment, Lx=1.0, Ly=1.0,
    )

    # Initialize with non-zero solution on fine level
    fine_smoother.initialize_lid()
    for _ in range(100):  # Run some iterations to get non-trivial solution
        fine_smoother.step()

    print(f"\n  Fine level solution after 100 iters:")
    print(f"    |u| = {np.linalg.norm(fine_level.u):.4e}")
    print(f"    |v| = {np.linalg.norm(fine_level.v):.4e}")
    print(f"    |p| = {np.linalg.norm(fine_level.p):.4e}")

    # Step 1: Compute residuals on fine level
    fine_smoother._compute_residuals(fine_level.u, fine_level.v, fine_level.p)

    print(f"\n  Fine level residuals:")
    print(f"    |R_u| = {np.linalg.norm(fine_level.R_u):.4e}")
    print(f"    |R_v| = {np.linalg.norm(fine_level.R_v):.4e}")
    print(f"    |R_p| = {np.linalg.norm(fine_level.R_p):.4e}")

    # Step 2: Restrict solution
    restrict_solution(fine_level, coarse_level, transfer_ops)

    print(f"\n  Coarse level after restriction:")
    print(f"    |u| = {np.linalg.norm(coarse_level.u):.4e}")
    print(f"    |v| = {np.linalg.norm(coarse_level.v):.4e}")
    print(f"    |p| = {np.linalg.norm(coarse_level.p):.4e}")

    # Step 3: Restrict residuals
    restrict_residual(fine_level, coarse_level, transfer_ops)
    I_R_u = coarse_level.R_u.copy()
    I_R_v = coarse_level.R_v.copy()
    I_R_p = coarse_level.R_p.copy()

    print(f"\n  Restricted fine residuals I(r_fine):")
    print(f"    |I(R_u)| = {np.linalg.norm(I_R_u):.4e}")
    print(f"    |I(R_v)| = {np.linalg.norm(I_R_v):.4e}")
    print(f"    |I(R_p)| = {np.linalg.norm(I_R_p):.4e}")

    # Step 4: Compute coarse residual from restricted solution
    coarse_smoother.initialize_lid()
    coarse_smoother._compute_residuals(coarse_level.u, coarse_level.v, coarse_level.p)

    L_H_u = coarse_level.R_u.copy()
    L_H_v = coarse_level.R_v.copy()
    L_H_p = coarse_level.R_p.copy()

    print(f"\n  Coarse residual L_H(I_H(u_h)):")
    print(f"    |L_H(u)| = {np.linalg.norm(L_H_u):.4e}")
    print(f"    |L_H(v)| = {np.linalg.norm(L_H_v):.4e}")
    print(f"    |L_H(p)| = {np.linalg.norm(L_H_p):.4e}")

    # Step 5: Compute tau = I(r_fine) - L_H(I_H(u_h))
    tau_u = I_R_u - L_H_u
    tau_v = I_R_v - L_H_v
    tau_p = I_R_p - L_H_p

    print(f"\n  FAS tau correction τ = I(r) - L_H(I(u)):")
    print(f"    |tau_u| = {np.linalg.norm(tau_u):.4e}")
    print(f"    |tau_v| = {np.linalg.norm(tau_v):.4e}")
    print(f"    |tau_p| = {np.linalg.norm(tau_p):.4e}")

    # Sanity check: tau should not be zero or huge
    tau_norm = np.linalg.norm(tau_u) + np.linalg.norm(tau_v)
    if tau_norm < 1e-14:
        print("\n  ⚠ Warning: Tau is essentially zero - FAS may not be doing anything!")
    elif tau_norm > 1e10:
        print("\n  ✗ ERROR: Tau is huge - numerical instability!")
    else:
        print("\n  ✓ Tau has reasonable magnitude")


def test_coarse_correction():
    """Test 3: Verify coarse grid correction improves fine grid solution."""
    print("\n" + "="*60)
    print("TEST 3: Coarse Grid Correction")
    print("="*60)

    levels, transfer_ops, corner_treatment = create_test_setup(n_fine=15, n_levels=2)

    coarse_level = levels[0]
    fine_level = levels[1]

    # Create smoothers
    coarse_smoother = MultigridSmoother(
        level=coarse_level,
        Re=100.0, beta_squared=5.0, lid_velocity=1.0, CFL=0.5,
        corner_treatment=corner_treatment, Lx=1.0, Ly=1.0,
    )
    fine_smoother = MultigridSmoother(
        level=fine_level,
        Re=100.0, beta_squared=5.0, lid_velocity=1.0, CFL=0.5,
        corner_treatment=corner_treatment, Lx=1.0, Ly=1.0,
    )

    # Initialize and run some fine grid iterations
    fine_smoother.initialize_lid()
    for _ in range(50):
        fine_smoother.step()

    # Get fine grid residual before correction
    fine_smoother._compute_residuals(fine_level.u, fine_level.v, fine_level.p)
    res_before = np.linalg.norm(fine_level.R_u) + np.linalg.norm(fine_level.R_v)

    print(f"\n  Fine grid residual before coarse correction: {res_before:.4e}")

    # Save fine solution
    u_fine_old = fine_level.u.copy()
    v_fine_old = fine_level.v.copy()
    p_fine_old = fine_level.p.copy()

    # Restrict solution to coarse
    restrict_solution(fine_level, coarse_level, transfer_ops)
    u_coarse_old = coarse_level.u.copy()
    v_coarse_old = coarse_level.v.copy()
    p_coarse_old = coarse_level.p.copy()

    # Restrict residual and compute tau
    restrict_residual(fine_level, coarse_level, transfer_ops)
    I_R_u = coarse_level.R_u.copy()
    I_R_v = coarse_level.R_v.copy()
    I_R_p = coarse_level.R_p.copy()

    coarse_smoother.initialize_lid()
    coarse_smoother._compute_residuals(coarse_level.u, coarse_level.v, coarse_level.p)

    tau_u = I_R_u - coarse_level.R_u
    tau_v = I_R_v - coarse_level.R_v
    tau_p = I_R_p - coarse_level.R_p

    # Apply tau and run coarse smoothing
    coarse_smoother.set_tau_correction(tau_u, tau_v, tau_p)

    print(f"\n  Running 100 coarse grid iterations with tau correction...")
    for i in range(100):
        u_res, v_res = coarse_smoother.step()
        if i % 25 == 0:
            print(f"    Coarse iter {i}: u_res={u_res:.2e}, v_res={v_res:.2e}")

    coarse_smoother.clear_tau_correction()

    # Compute coarse correction
    delta_u = coarse_level.u - u_coarse_old
    delta_v = coarse_level.v - v_coarse_old
    delta_p = coarse_level.p - p_coarse_old

    print(f"\n  Coarse correction magnitude:")
    print(f"    |delta_u| = {np.linalg.norm(delta_u):.4e}")
    print(f"    |delta_v| = {np.linalg.norm(delta_v):.4e}")
    print(f"    |delta_p| = {np.linalg.norm(delta_p):.4e}")

    # Zero out boundary corrections
    delta_u_2d = delta_u.reshape(coarse_level.shape_full).copy()
    delta_v_2d = delta_v.reshape(coarse_level.shape_full).copy()
    delta_u_2d[0, :] = 0.0
    delta_u_2d[-1, :] = 0.0
    delta_u_2d[:, 0] = 0.0
    delta_u_2d[:, -1] = 0.0
    delta_v_2d[0, :] = 0.0
    delta_v_2d[-1, :] = 0.0
    delta_v_2d[:, 0] = 0.0
    delta_v_2d[:, -1] = 0.0

    # Prolongate correction to fine
    delta_u_fine = transfer_ops.prolongation.prolongate_2d(
        delta_u_2d, fine_level.shape_full
    )
    delta_v_fine = transfer_ops.prolongation.prolongate_2d(
        delta_v_2d, fine_level.shape_full
    )
    delta_p_fine = transfer_ops.prolongation.prolongate_2d(
        delta_p.reshape(coarse_level.shape_inner), fine_level.shape_inner
    )

    # Zero boundary on fine
    delta_u_fine[0, :] = 0.0
    delta_u_fine[-1, :] = 0.0
    delta_u_fine[:, 0] = 0.0
    delta_u_fine[:, -1] = 0.0
    delta_v_fine[0, :] = 0.0
    delta_v_fine[-1, :] = 0.0
    delta_v_fine[:, 0] = 0.0
    delta_v_fine[:, -1] = 0.0

    print(f"\n  Prolongated correction magnitude:")
    print(f"    |delta_u_fine| = {np.linalg.norm(delta_u_fine):.4e}")
    print(f"    |delta_v_fine| = {np.linalg.norm(delta_v_fine):.4e}")
    print(f"    |delta_p_fine| = {np.linalg.norm(delta_p_fine):.4e}")

    # Test different damping values
    for damping in [0.1, 0.2, 0.5, 1.0]:
        # Apply correction
        fine_level.u[:] = u_fine_old + damping * delta_u_fine.ravel()
        fine_level.v[:] = v_fine_old + damping * delta_v_fine.ravel()
        fine_level.p[:] = p_fine_old + damping * delta_p_fine.ravel()
        fine_smoother._enforce_boundary_conditions(fine_level.u, fine_level.v)

        # Compute residual after correction
        fine_smoother._compute_residuals(fine_level.u, fine_level.v, fine_level.p)
        res_after = np.linalg.norm(fine_level.R_u) + np.linalg.norm(fine_level.R_v)

        improvement = (res_before - res_after) / res_before * 100
        print(f"\n  Damping={damping}: residual after = {res_after:.4e} ({improvement:+.1f}%)")

        if res_after < res_before:
            print(f"    ✓ Correction improved solution")
        else:
            print(f"    ✗ Correction made solution WORSE")


def test_vcycle_convergence():
    """Test 4: Compare V-cycle convergence with SG and FSG."""
    print("\n" + "="*60)
    print("TEST 4: V-cycle Convergence Comparison")
    print("="*60)

    basis_x = ChebyshevLobattoBasis(domain=(0.0, 1.0))
    basis_y = ChebyshevLobattoBasis(domain=(0.0, 1.0))
    transfer_ops = create_transfer_operators("fft", "fft")
    corner_treatment = create_corner_treatment(method="smoothing")

    n_fine = 15
    n_levels = 3
    Re = 100.0
    tolerance = 1e-5
    max_iter = 5000  # Low for debugging

    # Build fresh hierarchies for each test
    def build_fresh_hierarchy():
        return build_hierarchy(
            n_fine=n_fine,
            n_levels=n_levels,
            basis_x=basis_x,
            basis_y=basis_y,
            Lx=1.0,
            Ly=1.0,
        )

    # Test FSG
    print("\n  Testing FSG solver...")
    levels_fsg = build_fresh_hierarchy()
    _, fsg_iters, fsg_converged = solve_fsg(
        levels=levels_fsg,
        Re=Re,
        beta_squared=5.0,
        lid_velocity=1.0,
        CFL=0.5,
        tolerance=tolerance,
        max_iterations=max_iter * 3,  # FSG needs more per-level
        transfer_ops=transfer_ops,
        corner_treatment=corner_treatment,
    )
    print(f"    FSG: converged={fsg_converged}, iterations={fsg_iters}")

    # Test VMG with different damping values
    for damping in [0.2, 0.5, 0.8, 1.0]:
        print(f"\n  Testing VMG with damping={damping}...")
        levels_vmg = build_fresh_hierarchy()

        # Initialize finest level same as FSG starts coarsest
        # (Actually VMG should work from scratch too)

        try:
            _, vmg_iters, vmg_converged = solve_vmg(
                levels=levels_vmg,
                Re=Re,
                beta_squared=5.0,
                lid_velocity=1.0,
                CFL=0.5,
                tolerance=tolerance,
                max_iterations=max_iter,
                transfer_ops=transfer_ops,
                corner_treatment=corner_treatment,
                pre_smoothing=[3, 2, 1],
                post_smoothing=[0, 0, 0],
                correction_damping=damping,
            )
            print(f"    VMG (damping={damping}): converged={vmg_converged}, iterations={vmg_iters}")
        except Exception as e:
            print(f"    VMG (damping={damping}): FAILED with {type(e).__name__}: {e}")


def test_fsg_initialized_vmg():
    """Test 5: Run VMG starting from FSG-converged solution."""
    print("\n" + "="*60)
    print("TEST 5: VMG Starting from FSG Solution")
    print("="*60)

    basis_x = ChebyshevLobattoBasis(domain=(0.0, 1.0))
    basis_y = ChebyshevLobattoBasis(domain=(0.0, 1.0))
    transfer_ops = create_transfer_operators("fft", "fft")
    corner_treatment = create_corner_treatment(method="smoothing")

    n_fine = 15
    n_levels = 3
    Re = 100.0
    tolerance = 1e-5

    # First run FSG to convergence
    print("\n  Running FSG to convergence first...")
    levels = build_hierarchy(
        n_fine=n_fine,
        n_levels=n_levels,
        basis_x=basis_x,
        basis_y=basis_y,
    )

    finest, fsg_iters, fsg_conv = solve_fsg(
        levels=levels,
        Re=Re,
        beta_squared=5.0,
        lid_velocity=1.0,
        CFL=0.5,
        tolerance=tolerance,
        max_iterations=100000,
        transfer_ops=transfer_ops,
        corner_treatment=corner_treatment,
    )

    print(f"    FSG converged in {fsg_iters} iterations")
    print(f"    |u| = {np.linalg.norm(finest.u):.4e}")
    print(f"    |v| = {np.linalg.norm(finest.v):.4e}")

    # Save converged solution
    u_converged = finest.u.copy()
    v_converged = finest.v.copy()
    p_converged = finest.p.copy()

    # Now run VMG V-cycles on converged solution - should stay converged
    print("\n  Running VMG V-cycles on converged solution...")

    # Rebuild levels but initialize finest with converged solution
    levels_vmg = build_hierarchy(
        n_fine=n_fine,
        n_levels=n_levels,
        basis_x=basis_x,
        basis_y=basis_y,
    )

    # Copy converged solution to finest level
    levels_vmg[-1].u[:] = u_converged
    levels_vmg[-1].v[:] = v_converged
    levels_vmg[-1].p[:] = p_converged

    # Run just a few V-cycles
    _, vmg_iters, vmg_conv = solve_vmg(
        levels=levels_vmg,
        Re=Re,
        beta_squared=5.0,
        lid_velocity=1.0,
        CFL=0.5,
        tolerance=tolerance,
        max_iterations=10,  # Just 10 V-cycles
        transfer_ops=transfer_ops,
        corner_treatment=corner_treatment,
        pre_smoothing=[3, 2, 1],
        correction_damping=1.0,
    )

    finest_vmg = levels_vmg[-1]

    # Check if solution changed much
    u_change = np.linalg.norm(finest_vmg.u - u_converged) / np.linalg.norm(u_converged)
    v_change = np.linalg.norm(finest_vmg.v - v_converged) / np.linalg.norm(v_converged)

    print(f"\n  After {vmg_iters} VMG iterations:")
    print(f"    Relative u change: {u_change:.4e}")
    print(f"    Relative v change: {v_change:.4e}")

    if u_change < 0.01 and v_change < 0.01:
        print("    ✓ VMG preserves converged solution well")
    elif u_change < 0.1 and v_change < 0.1:
        print("    ⚠ VMG modifies converged solution slightly")
    else:
        print("    ✗ VMG significantly disturbs converged solution - PROBLEM!")


def test_single_vcycle_detailed():
    """Test 6: Step through a single V-cycle with detailed output."""
    print("\n" + "="*60)
    print("TEST 6: Single V-cycle Detailed Trace")
    print("="*60)

    levels, transfer_ops, corner_treatment = create_test_setup(n_fine=15, n_levels=3)

    # Create smoothers for each level
    smoothers = []
    for level in levels:
        smoother = MultigridSmoother(
            level=level,
            Re=100.0, beta_squared=5.0, lid_velocity=1.0, CFL=0.5,
            corner_treatment=corner_treatment, Lx=1.0, Ly=1.0,
        )
        smoother.initialize_lid()
        smoothers.append(smoother)

    # Initialize finest level with some iterations
    print("\n  Initializing finest level with 50 iterations...")
    finest = levels[-1]
    finest_smoother = smoothers[-1]
    for _ in range(50):
        finest_smoother.step()

    print(f"    |u| = {np.linalg.norm(finest.u):.4e}")

    # Manual V-cycle trace
    print("\n  === DOWNWARD LEG ===")

    for level_idx in range(len(levels) - 1, 0, -1):
        fine = levels[level_idx]
        coarse = levels[level_idx - 1]
        fine_sm = smoothers[level_idx]
        coarse_sm = smoothers[level_idx - 1]

        print(f"\n  Level {level_idx} (N={fine.n}) -> Level {level_idx-1} (N={coarse.n})")

        # Pre-smooth
        pre_iters = 3 - level_idx + 1  # e.g., 1 on finest, 2, 3 on coarsest
        print(f"    Pre-smoothing {pre_iters} iterations...")
        for _ in range(pre_iters):
            fine_sm.step()

        # Compute residual
        fine_sm._compute_residuals(fine.u, fine.v, fine.p)
        print(f"    |R_u| = {np.linalg.norm(fine.R_u):.4e}")

        # Store state
        u_fine_old = fine.u.copy()

        # Restrict solution
        restrict_solution(fine, coarse, transfer_ops)
        u_coarse_old = coarse.u.copy()
        print(f"    After restriction: |u_coarse| = {np.linalg.norm(coarse.u):.4e}")

        # Restrict residual
        restrict_residual(fine, coarse, transfer_ops)
        I_R_u = coarse.R_u.copy()

        # Compute coarse residual
        coarse_sm._compute_residuals(coarse.u, coarse.v, coarse.p)

        # Compute tau
        tau_u = I_R_u - coarse.R_u
        tau_v = coarse.R_v.copy() * 0  # Simplified
        tau_p = coarse.R_p.copy() * 0  # Simplified

        print(f"    |tau_u| = {np.linalg.norm(tau_u):.4e}")

        # Set tau for next level
        coarse_sm.set_tau_correction(tau_u, tau_v, tau_p)

    # Coarsest level solve
    print("\n  Coarsest level (extensive smoothing)...")
    coarse = levels[0]
    coarse_sm = smoothers[0]
    for _ in range(50):
        coarse_sm.step()
    coarse_sm.clear_tau_correction()

    print(f"    After 50 iters: |u| = {np.linalg.norm(coarse.u):.4e}")

    print("\n  === UPWARD LEG ===")

    for level_idx in range(1, len(levels)):
        coarse = levels[level_idx - 1]
        fine = levels[level_idx]
        fine_sm = smoothers[level_idx]

        print(f"\n  Level {level_idx-1} (N={coarse.n}) -> Level {level_idx} (N={fine.n})")

        # In a real V-cycle, we'd have stored the old coarse solution
        # For now, just prolongate current coarse to fine
        print(f"    Before: |u_fine| = {np.linalg.norm(fine.u):.4e}")

        # Post-smooth
        print(f"    Post-smoothing 1 iteration...")
        fine_sm.step()

        print(f"    After: |u_fine| = {np.linalg.norm(fine.u):.4e}")


def main():
    """Run all VMG debug tests."""
    print("VMG Solver Debug Script")
    print("="*60)

    test_smoother_single_level()
    test_tau_correction()
    test_coarse_correction()
    test_vcycle_convergence()
    test_fsg_initialized_vmg()
    test_single_vcycle_detailed()

    print("\n" + "="*60)
    print("DEBUG COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
