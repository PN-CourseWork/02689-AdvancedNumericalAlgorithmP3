"""Diagnostic script to trace what happens during a FAS V-cycle."""

import numpy as np
import sys
sys.path.insert(0, 'src')

from solvers.spectral.fas import (
    build_fas_hierarchy,
    fas_rk4_step,
    compute_residuals,
    compute_continuity_rms,
    restrict_solution,
    restrict_residual,
    prolongate_correction,
    enforce_boundary_conditions,
)
from solvers.spectral.basis.spectral import ChebyshevLobattoBasis
from solvers.spectral.operators.corner import create_corner_treatment

# Parameters
N = 24
Re = 100.0
beta_squared = 5.0
lid_velocity = 1.0
CFL = 2.5
Lx = Ly = 1.0

# Build hierarchy
basis_x = ChebyshevLobattoBasis()
basis_y = ChebyshevLobattoBasis()
corner_treatment = create_corner_treatment("smoothing", Lx=Lx, Ly=Ly)

levels = build_fas_hierarchy(
    n_fine=N,
    n_levels=2,
    basis_x=basis_x,
    basis_y=basis_y,
    Lx=Lx,
    Ly=Ly,
    coarsest_n=12,
)

fine = levels[1]  # N=24
coarse = levels[0]  # N=12

print(f"Fine level: N={fine.n}, shape={fine.shape_full}")
print(f"Coarse level: N={coarse.n}, shape={coarse.shape_full}")

# Initialize fine level
fine.u[:] = 0.0
fine.v[:] = 0.0
fine.p[:] = 0.0
enforce_boundary_conditions(fine, fine.u, fine.v, lid_velocity, corner_treatment, Lx, Ly)

print("\n=== Initial state ===")
e_rms_0 = compute_continuity_rms(fine)
print(f"Fine E_RMS initial: {e_rms_0:.6e}")

# Do 100 SG iterations to get to a non-trivial state
print("\n=== Running 100 SG iterations to establish flow ===")
for i in range(100):
    fas_rk4_step(fine, Re, beta_squared, lid_velocity, CFL, corner_treatment, Lx, Ly)

e_rms_after_sg = compute_continuity_rms(fine)
print(f"Fine E_RMS after 100 SG: {e_rms_after_sg:.6e}")
print(f"Fine max |u|: {np.max(np.abs(fine.u)):.6f}")
print(f"Fine max |p|: {np.max(np.abs(fine.p)):.6f}")

# Now trace through ONE V-cycle step by step
print("\n" + "="*60)
print("=== TRACING ONE V-CYCLE ===")
print("="*60)

# Save fine solution before V-cycle
u_fine_before = fine.u.copy()
v_fine_before = fine.v.copy()
p_fine_before = fine.p.copy()

# Step 1: Pre-smoothing on fine (1 RK4 step)
print("\n--- Step 1: Pre-smoothing on fine ---")
fas_rk4_step(fine, Re, beta_squared, lid_velocity, CFL, corner_treatment, Lx, Ly)
e_rms_after_presmooth = compute_continuity_rms(fine)
print(f"Fine E_RMS after pre-smooth: {e_rms_after_presmooth:.6e}")
print(f"Change from SG step: {(e_rms_after_presmooth - e_rms_after_sg)/e_rms_after_sg*100:+.2f}%")

# Save fine solution after pre-smooth
u_h_old = fine.u.copy()
v_h_old = fine.v.copy()
p_h_old = fine.p.copy()

# Step 2: Compute fine grid residual
print("\n--- Step 2: Compute fine residual ---")
compute_residuals(fine, fine.u, fine.v, fine.p, Re, beta_squared)
print(f"Fine |R_u|_max: {np.max(np.abs(fine.R_u)):.6e}")
print(f"Fine |R_v|_max: {np.max(np.abs(fine.R_v)):.6e}")
print(f"Fine |R_p|_max: {np.max(np.abs(fine.R_p)):.6e}")

# Step 3: Restrict residual
print("\n--- Step 3: Restrict residual (FFT) ---")
I_r_u, I_r_v, I_r_p = restrict_residual(fine, coarse)
print(f"Restricted |I(r_u)|_max: {np.max(np.abs(I_r_u)):.6e}")
print(f"Restricted |I(r_v)|_max: {np.max(np.abs(I_r_v)):.6e}")
print(f"Restricted |I(r_p)|_max: {np.max(np.abs(I_r_p)):.6e}")

# Step 4: Restrict solution (injection)
print("\n--- Step 4: Restrict solution (injection) ---")
restrict_solution(fine, coarse)
enforce_boundary_conditions(coarse, coarse.u, coarse.v, lid_velocity, corner_treatment, Lx, Ly)
print(f"Coarse |u|_max: {np.max(np.abs(coarse.u)):.6e}")
print(f"Coarse |v|_max: {np.max(np.abs(coarse.v)):.6e}")
print(f"Coarse |p|_max: {np.max(np.abs(coarse.p)):.6e}")

u_H_old = coarse.u.copy()
v_H_old = coarse.v.copy()
p_H_old = coarse.p.copy()

# Step 5: Compute coarse residual at restricted solution
print("\n--- Step 5: Compute coarse residual ---")
compute_residuals(coarse, coarse.u, coarse.v, coarse.p, Re, beta_squared)

# Zero boundaries on coarse residual
R_u_2d = coarse.R_u.reshape(coarse.shape_full)
R_v_2d = coarse.R_v.reshape(coarse.shape_full)
R_u_2d[0, :] = 0.0; R_u_2d[-1, :] = 0.0; R_u_2d[:, 0] = 0.0; R_u_2d[:, -1] = 0.0
R_v_2d[0, :] = 0.0; R_v_2d[-1, :] = 0.0; R_v_2d[:, 0] = 0.0; R_v_2d[:, -1] = 0.0

print(f"Coarse |R_u|_max: {np.max(np.abs(coarse.R_u)):.6e}")
print(f"Coarse |R_v|_max: {np.max(np.abs(coarse.R_v)):.6e}")
print(f"Coarse |R_p|_max: {np.max(np.abs(coarse.R_p)):.6e}")

# Step 6: Compute tau
print("\n--- Step 6: Compute tau correction ---")
tau_u = I_r_u - coarse.R_u
tau_v = I_r_v - coarse.R_v
tau_p = I_r_p - coarse.R_p
print(f"|tau_u|_max: {np.max(np.abs(tau_u)):.6e}")
print(f"|tau_v|_max: {np.max(np.abs(tau_v)):.6e}")
print(f"|tau_p|_max: {np.max(np.abs(tau_p)):.6e}")

# Step 7: Coarse grid solve with tau
print("\n--- Step 7: Coarse grid solve (1 RK4 step with tau) ---")
# Test with tau
coarse.tau_u = tau_u
coarse.tau_v = tau_v
coarse.tau_p = tau_p

e_rms_coarse_before = compute_continuity_rms(coarse)
print(f"Coarse E_RMS before: {e_rms_coarse_before:.6e}")
print(f"Testing WITH tau first...")

fas_rk4_step(coarse, Re, beta_squared, lid_velocity, CFL, corner_treatment, Lx, Ly)

e_rms_coarse_after = compute_continuity_rms(coarse)
print(f"Coarse E_RMS after: {e_rms_coarse_after:.6e}")
print(f"Coarse |u|_max after: {np.max(np.abs(coarse.u)):.6e}")
print(f"Coarse |p|_max after: {np.max(np.abs(coarse.p)):.6e}")

# Step 8: Compute correction
print("\n--- Step 8: Compute coarse correction ---")
e_u = coarse.u - u_H_old
e_v = coarse.v - v_H_old
e_p = coarse.p - p_H_old
print(f"|e_u|_max: {np.max(np.abs(e_u)):.6e}")
print(f"|e_v|_max: {np.max(np.abs(e_v)):.6e}")
print(f"|e_p|_max: {np.max(np.abs(e_p)):.6e}")

# Step 9: Prolongate correction
print("\n--- Step 9: Prolongate correction (FFT) ---")
e_u_fine, e_v_fine, e_p_fine = prolongate_correction(coarse, fine, e_u, e_v, e_p)
print(f"|e_u_fine|_max: {np.max(np.abs(e_u_fine)):.6e}")
print(f"|e_v_fine|_max: {np.max(np.abs(e_v_fine)):.6e}")
print(f"|e_p_fine|_max: {np.max(np.abs(e_p_fine)):.6e}")

# Step 10: Apply correction
print("\n--- Step 10: Apply correction to fine grid ---")
fine.u[:] = u_h_old + e_u_fine
fine.v[:] = v_h_old + e_v_fine
fine.p[:] = p_h_old + e_p_fine
enforce_boundary_conditions(fine, fine.u, fine.v, lid_velocity, corner_treatment, Lx, Ly)

e_rms_after_vcycle = compute_continuity_rms(fine)
print(f"Fine E_RMS after V-cycle: {e_rms_after_vcycle:.6e}")
print(f"Change from pre-smooth: {(e_rms_after_vcycle - e_rms_after_presmooth)/e_rms_after_presmooth*100:+.2f}%")
print(f"Change from start of V-cycle: {(e_rms_after_vcycle - e_rms_after_sg)/e_rms_after_sg*100:+.2f}%")

# Clear tau
coarse.tau_u = None
coarse.tau_v = None
coarse.tau_p = None

print("\n" + "="*60)
print("=== COMPARISON: V-CYCLE vs 2 SG STEPS ===")
print("="*60)

# Reset to state before V-cycle
fine.u[:] = u_fine_before
fine.v[:] = v_fine_before
fine.p[:] = p_fine_before

# Do 2 SG steps (same total work as 1 pre-smooth + 1 coarse step)
fas_rk4_step(fine, Re, beta_squared, lid_velocity, CFL, corner_treatment, Lx, Ly)
fas_rk4_step(fine, Re, beta_squared, lid_velocity, CFL, corner_treatment, Lx, Ly)

e_rms_after_2sg = compute_continuity_rms(fine)
print(f"Fine E_RMS after 2 SG steps: {e_rms_after_2sg:.6e}")
print(f"Change from initial: {(e_rms_after_2sg - e_rms_after_sg)/e_rms_after_sg*100:+.2f}%")

print(f"\n>>> V-cycle improvement: {(e_rms_after_vcycle - e_rms_after_sg)/e_rms_after_sg*100:+.2f}%")
print(f">>> 2 SG improvement: {(e_rms_after_2sg - e_rms_after_sg)/e_rms_after_sg*100:+.2f}%")
print(f">>> V-cycle advantage: {(e_rms_after_2sg - e_rms_after_vcycle)/e_rms_after_2sg*100:+.2f}%")

# Additional test: V-cycle WITHOUT tau
print("\n" + "="*60)
print("=== TEST: V-CYCLE WITHOUT TAU ===")
print("="*60)

# Reset to state before V-cycle
fine.u[:] = u_fine_before
fine.v[:] = v_fine_before
fine.p[:] = p_fine_before

# Step 1: Pre-smoothing on fine (1 RK4 step)
fas_rk4_step(fine, Re, beta_squared, lid_velocity, CFL, corner_treatment, Lx, Ly)
u_h_old2 = fine.u.copy()
v_h_old2 = fine.v.copy()
p_h_old2 = fine.p.copy()

# Restrict solution
restrict_solution(fine, coarse)
enforce_boundary_conditions(coarse, coarse.u, coarse.v, lid_velocity, corner_treatment, Lx, Ly)
u_H_old2 = coarse.u.copy()
v_H_old2 = coarse.v.copy()
p_H_old2 = coarse.p.copy()

# Coarse solve WITHOUT tau
coarse.tau_u = None
coarse.tau_v = None
coarse.tau_p = None
fas_rk4_step(coarse, Re, beta_squared, lid_velocity, CFL, corner_treatment, Lx, Ly)

# Compute correction
e_u2 = coarse.u - u_H_old2
e_v2 = coarse.v - v_H_old2
e_p2 = coarse.p - p_H_old2

# Prolongate
e_u_fine2, e_v_fine2, e_p_fine2 = prolongate_correction(coarse, fine, e_u2, e_v2, e_p2)

# Apply correction
fine.u[:] = u_h_old2 + e_u_fine2
fine.v[:] = v_h_old2 + e_v_fine2
fine.p[:] = p_h_old2 + e_p_fine2
enforce_boundary_conditions(fine, fine.u, fine.v, lid_velocity, corner_treatment, Lx, Ly)

e_rms_no_tau = compute_continuity_rms(fine)
print(f"V-cycle WITHOUT tau E_RMS: {e_rms_no_tau:.6e}")
print(f">>> No-tau V-cycle improvement: {(e_rms_no_tau - e_rms_after_sg)/e_rms_after_sg*100:+.2f}%")
print(f">>> Comparison: WITH tau = {e_rms_after_vcycle:.6e}, WITHOUT tau = {e_rms_no_tau:.6e}")
print(f">>> Tau makes it {'better' if e_rms_after_vcycle < e_rms_no_tau else 'WORSE'} by {abs(e_rms_after_vcycle - e_rms_no_tau)/e_rms_no_tau*100:.2f}%")

# Test with MORE coarse iterations
print("\n" + "="*60)
print("=== TEST: VARYING COARSE ITERATIONS ===")
print("="*60)

for n_coarse in [1, 2, 3, 5, 10]:
    # Reset to state before V-cycle
    fine.u[:] = u_fine_before
    fine.v[:] = v_fine_before
    fine.p[:] = p_fine_before

    # Step 1: Pre-smoothing on fine (1 RK4 step)
    fas_rk4_step(fine, Re, beta_squared, lid_velocity, CFL, corner_treatment, Lx, Ly)
    u_h_old3 = fine.u.copy()
    v_h_old3 = fine.v.copy()
    p_h_old3 = fine.p.copy()

    # Compute fine residual
    compute_residuals(fine, fine.u, fine.v, fine.p, Re, beta_squared)
    I_r_u3, I_r_v3, I_r_p3 = restrict_residual(fine, coarse)

    # Restrict solution
    restrict_solution(fine, coarse)
    enforce_boundary_conditions(coarse, coarse.u, coarse.v, lid_velocity, corner_treatment, Lx, Ly)
    u_H_old3 = coarse.u.copy()
    v_H_old3 = coarse.v.copy()
    p_H_old3 = coarse.p.copy()

    # Compute tau
    compute_residuals(coarse, coarse.u, coarse.v, coarse.p, Re, beta_squared)
    R_u_2d = coarse.R_u.reshape(coarse.shape_full)
    R_v_2d = coarse.R_v.reshape(coarse.shape_full)
    R_u_2d[0, :] = 0.0; R_u_2d[-1, :] = 0.0; R_u_2d[:, 0] = 0.0; R_u_2d[:, -1] = 0.0
    R_v_2d[0, :] = 0.0; R_v_2d[-1, :] = 0.0; R_v_2d[:, 0] = 0.0; R_v_2d[:, -1] = 0.0
    tau_u3 = I_r_u3 - coarse.R_u
    tau_v3 = I_r_v3 - coarse.R_v
    tau_p3 = I_r_p3 - coarse.R_p

    # Coarse solve with tau for ALL iterations
    coarse.tau_u = tau_u3
    coarse.tau_v = tau_v3
    coarse.tau_p = tau_p3
    for _ in range(n_coarse):
        fas_rk4_step(coarse, Re, beta_squared, lid_velocity, CFL, corner_treatment, Lx, Ly)
    coarse.tau_u = None
    coarse.tau_v = None
    coarse.tau_p = None

    # Compute correction
    e_u3 = coarse.u - u_H_old3
    e_v3 = coarse.v - v_H_old3
    e_p3 = coarse.p - p_H_old3

    # Prolongate
    e_u_fine3, e_v_fine3, e_p_fine3 = prolongate_correction(coarse, fine, e_u3, e_v3, e_p3)

    # Apply correction
    fine.u[:] = u_h_old3 + e_u_fine3
    fine.v[:] = v_h_old3 + e_v_fine3
    fine.p[:] = p_h_old3 + e_p_fine3
    enforce_boundary_conditions(fine, fine.u, fine.v, lid_velocity, corner_treatment, Lx, Ly)

    e_rms_test = compute_continuity_rms(fine)
    improvement = (e_rms_test - e_rms_after_sg)/e_rms_after_sg*100
    print(f"{n_coarse} coarse iters (tau all): E_RMS={e_rms_test:.6e}, improvement={improvement:+.2f}%")

print("\n--- Now testing tau on FIRST coarse iteration only ---")
for n_coarse in [1, 2, 3, 5, 10]:
    # Reset to state before V-cycle
    fine.u[:] = u_fine_before
    fine.v[:] = v_fine_before
    fine.p[:] = p_fine_before

    # Step 1: Pre-smoothing on fine (1 RK4 step)
    fas_rk4_step(fine, Re, beta_squared, lid_velocity, CFL, corner_treatment, Lx, Ly)
    u_h_old4 = fine.u.copy()
    v_h_old4 = fine.v.copy()
    p_h_old4 = fine.p.copy()

    # Compute fine residual
    compute_residuals(fine, fine.u, fine.v, fine.p, Re, beta_squared)
    I_r_u4, I_r_v4, I_r_p4 = restrict_residual(fine, coarse)

    # Restrict solution
    restrict_solution(fine, coarse)
    enforce_boundary_conditions(coarse, coarse.u, coarse.v, lid_velocity, corner_treatment, Lx, Ly)
    u_H_old4 = coarse.u.copy()
    v_H_old4 = coarse.v.copy()
    p_H_old4 = coarse.p.copy()

    # Compute tau
    compute_residuals(coarse, coarse.u, coarse.v, coarse.p, Re, beta_squared)
    R_u_2d = coarse.R_u.reshape(coarse.shape_full)
    R_v_2d = coarse.R_v.reshape(coarse.shape_full)
    R_u_2d[0, :] = 0.0; R_u_2d[-1, :] = 0.0; R_u_2d[:, 0] = 0.0; R_u_2d[:, -1] = 0.0
    R_v_2d[0, :] = 0.0; R_v_2d[-1, :] = 0.0; R_v_2d[:, 0] = 0.0; R_v_2d[:, -1] = 0.0
    tau_u4 = I_r_u4 - coarse.R_u
    tau_v4 = I_r_v4 - coarse.R_v
    tau_p4 = I_r_p4 - coarse.R_p

    # First coarse iteration WITH tau
    coarse.tau_u = tau_u4
    coarse.tau_v = tau_v4
    coarse.tau_p = tau_p4
    fas_rk4_step(coarse, Re, beta_squared, lid_velocity, CFL, corner_treatment, Lx, Ly)
    coarse.tau_u = None
    coarse.tau_v = None
    coarse.tau_p = None

    # Remaining iterations WITHOUT tau
    for _ in range(n_coarse - 1):
        fas_rk4_step(coarse, Re, beta_squared, lid_velocity, CFL, corner_treatment, Lx, Ly)

    # Compute correction
    e_u4 = coarse.u - u_H_old4
    e_v4 = coarse.v - v_H_old4
    e_p4 = coarse.p - p_H_old4

    # Prolongate
    e_u_fine4, e_v_fine4, e_p_fine4 = prolongate_correction(coarse, fine, e_u4, e_v4, e_p4)

    # Apply correction
    fine.u[:] = u_h_old4 + e_u_fine4
    fine.v[:] = v_h_old4 + e_v_fine4
    fine.p[:] = p_h_old4 + e_p_fine4
    enforce_boundary_conditions(fine, fine.u, fine.v, lid_velocity, corner_treatment, Lx, Ly)

    e_rms_test2 = compute_continuity_rms(fine)
    improvement2 = (e_rms_test2 - e_rms_after_sg)/e_rms_after_sg*100
    print(f"{n_coarse} coarse iters (tau first only): E_RMS={e_rms_test2:.6e}, improvement={improvement2:+.2f}%")
