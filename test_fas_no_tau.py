"""Test FAS without tau correction - just add restricted fine residual.

The Zhang & Xi paper says:
    A_{k-1}(Q̃_{k-1}) = b_{k-1} + I_k^{k-1}(r_k)

This is NOT quite the same as the tau correction we implemented!

In FAS tau formulation:
    tau = I(r_h) - R_H(Î u_h)   (computed ONCE at restriction)
    Then solve: dQ/dt = R_H(Q) + tau

In Zhang & Xi's formulation (as I now understand it):
    Just add I(r_h) to the RHS!
    Solve: dQ/dt = R_H(Q) + I(r_h)

The difference is subtle but important:
- tau depends on R_H at the INITIAL restricted solution
- I(r_h) is a CONSTANT source term

Let's test this approach!
"""

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

# Run 100 SG iterations to establish flow
print("\n=== Running 100 SG iterations to establish flow ===")
for i in range(100):
    fas_rk4_step(fine, Re, beta_squared, lid_velocity, CFL, corner_treatment, Lx, Ly)

e_rms_initial = compute_continuity_rms(fine)
print(f"Fine E_RMS after 100 SG: {e_rms_initial:.6e}")

# Save initial state
u_initial = fine.u.copy()
v_initial = fine.v.copy()
p_initial = fine.p.copy()


def rk4_step_with_source(level, Re, beta_squared, lid_velocity, CFL, corner_treatment, Lx, Ly,
                          source_u, source_v, source_p):
    """RK4 step with CONSTANT source term added to residual."""
    from solvers.spectral.fas import compute_adaptive_timestep

    dt = compute_adaptive_timestep(level, Re, beta_squared, lid_velocity, CFL)

    rk4_coeffs = [0.25, 1.0/3.0, 0.5, 1.0]
    u_in, v_in, p_in = level.u, level.v, level.p

    for i, alpha in enumerate(rk4_coeffs):
        compute_residuals(level, u_in, v_in, p_in, Re, beta_squared)

        # Add CONSTANT source (not tau!)
        level.R_u[:] += source_u
        level.R_v[:] += source_v
        level.R_p[:] += source_p

        if i < 3:
            level.u_stage[:] = level.u + alpha * dt * level.R_u
            level.v_stage[:] = level.v + alpha * dt * level.R_v
            level.p_stage[:] = level.p + alpha * dt * level.R_p
            enforce_boundary_conditions(
                level, level.u_stage, level.v_stage,
                lid_velocity, corner_treatment, Lx, Ly
            )
            u_in, v_in, p_in = level.u_stage, level.v_stage, level.p_stage
        else:
            level.u[:] = level.u + alpha * dt * level.R_u
            level.v[:] = level.v + alpha * dt * level.R_v
            level.p[:] = level.p + alpha * dt * level.R_p
            enforce_boundary_conditions(
                level, level.u, level.v,
                lid_velocity, corner_treatment, Lx, Ly
            )


def vcycle_with_source(n_coarse_iters=1):
    """V-cycle using Zhang & Xi approach: add I(r_h) as source."""
    # Reset fine to initial state
    fine.u[:] = u_initial.copy()
    fine.v[:] = v_initial.copy()
    fine.p[:] = p_initial.copy()

    # Step 1: Pre-smoothing on fine (1 RK4 step)
    fas_rk4_step(fine, Re, beta_squared, lid_velocity, CFL, corner_treatment, Lx, Ly)

    # Save fine solution after pre-smooth
    u_h_old = fine.u.copy()
    v_h_old = fine.v.copy()
    p_h_old = fine.p.copy()

    # Step 2: Compute fine grid residual
    compute_residuals(fine, fine.u, fine.v, fine.p, Re, beta_squared)

    # Step 3: Restrict residual (this becomes our SOURCE term!)
    I_r_u, I_r_v, I_r_p = restrict_residual(fine, coarse)

    # Step 4: Restrict solution
    restrict_solution(fine, coarse)
    enforce_boundary_conditions(coarse, coarse.u, coarse.v, lid_velocity, corner_treatment, Lx, Ly)

    u_H_old = coarse.u.copy()
    v_H_old = coarse.v.copy()
    p_H_old = coarse.p.copy()

    # Step 5: Coarse solve with CONSTANT source I(r_h)
    # NO TAU! Just add the restricted residual as a source.
    for _ in range(n_coarse_iters):
        rk4_step_with_source(
            coarse, Re, beta_squared, lid_velocity, CFL, corner_treatment, Lx, Ly,
            I_r_u, I_r_v, I_r_p
        )

    # Step 6: Compute correction
    e_u = coarse.u - u_H_old
    e_v = coarse.v - v_H_old
    e_p = coarse.p - p_H_old

    # Step 7: Prolongate
    e_u_fine, e_v_fine, e_p_fine = prolongate_correction(coarse, fine, e_u, e_v, e_p)

    # Step 8: Apply correction
    fine.u[:] = u_h_old + e_u_fine
    fine.v[:] = v_h_old + e_v_fine
    fine.p[:] = p_h_old + e_p_fine
    enforce_boundary_conditions(fine, fine.u, fine.v, lid_velocity, corner_treatment, Lx, Ly)

    return compute_continuity_rms(fine)


print("\n" + "="*60)
print("=== TEST: V-CYCLE WITH I(r_h) AS SOURCE (NO TAU) ===")
print("="*60)

for n_coarse in [1, 2, 3, 5, 10]:
    erms = vcycle_with_source(n_coarse)
    improvement = (erms - e_rms_initial)/e_rms_initial*100
    print(f"{n_coarse} coarse iters (source): E_RMS={erms:.6e}, improvement={improvement:+.2f}%")


# Compare with tau approach
print("\n" + "="*60)
print("=== COMPARE: V-CYCLE WITH TAU (original implementation) ===")
print("="*60)

def vcycle_with_tau(n_coarse_iters=1):
    """Original tau-based V-cycle."""
    # Reset fine to initial state
    fine.u[:] = u_initial.copy()
    fine.v[:] = v_initial.copy()
    fine.p[:] = p_initial.copy()

    # Step 1: Pre-smoothing on fine
    fas_rk4_step(fine, Re, beta_squared, lid_velocity, CFL, corner_treatment, Lx, Ly)

    u_h_old = fine.u.copy()
    v_h_old = fine.v.copy()
    p_h_old = fine.p.copy()

    # Step 2: Fine residual
    compute_residuals(fine, fine.u, fine.v, fine.p, Re, beta_squared)

    # Step 3: Restrict residual
    I_r_u, I_r_v, I_r_p = restrict_residual(fine, coarse)

    # Step 4: Restrict solution
    restrict_solution(fine, coarse)
    enforce_boundary_conditions(coarse, coarse.u, coarse.v, lid_velocity, corner_treatment, Lx, Ly)

    u_H_old = coarse.u.copy()
    v_H_old = coarse.v.copy()
    p_H_old = coarse.p.copy()

    # Step 5: Compute tau
    compute_residuals(coarse, coarse.u, coarse.v, coarse.p, Re, beta_squared)
    R_u_2d = coarse.R_u.reshape(coarse.shape_full)
    R_v_2d = coarse.R_v.reshape(coarse.shape_full)
    R_u_2d[0, :] = 0.0; R_u_2d[-1, :] = 0.0; R_u_2d[:, 0] = 0.0; R_u_2d[:, -1] = 0.0
    R_v_2d[0, :] = 0.0; R_v_2d[-1, :] = 0.0; R_v_2d[:, 0] = 0.0; R_v_2d[:, -1] = 0.0
    tau_u = I_r_u - coarse.R_u
    tau_v = I_r_v - coarse.R_v
    tau_p = I_r_p - coarse.R_p

    # Step 6: Coarse solve with tau
    coarse.tau_u = tau_u
    coarse.tau_v = tau_v
    coarse.tau_p = tau_p
    for _ in range(n_coarse_iters):
        fas_rk4_step(coarse, Re, beta_squared, lid_velocity, CFL, corner_treatment, Lx, Ly)
    coarse.tau_u = None
    coarse.tau_v = None
    coarse.tau_p = None

    # Step 7-8: Correction
    e_u = coarse.u - u_H_old
    e_v = coarse.v - v_H_old
    e_p = coarse.p - p_H_old
    e_u_fine, e_v_fine, e_p_fine = prolongate_correction(coarse, fine, e_u, e_v, e_p)

    fine.u[:] = u_h_old + e_u_fine
    fine.v[:] = v_h_old + e_v_fine
    fine.p[:] = p_h_old + e_p_fine
    enforce_boundary_conditions(fine, fine.u, fine.v, lid_velocity, corner_treatment, Lx, Ly)

    return compute_continuity_rms(fine)

for n_coarse in [1, 2, 3, 5, 10]:
    erms = vcycle_with_tau(n_coarse)
    improvement = (erms - e_rms_initial)/e_rms_initial*100
    print(f"{n_coarse} coarse iters (tau): E_RMS={erms:.6e}, improvement={improvement:+.2f}%")


# And compare with simple SG
print("\n" + "="*60)
print("=== BASELINE: 2 SG STEPS ===")
print("="*60)
fine.u[:] = u_initial.copy()
fine.v[:] = v_initial.copy()
fine.p[:] = p_initial.copy()
fas_rk4_step(fine, Re, beta_squared, lid_velocity, CFL, corner_treatment, Lx, Ly)
fas_rk4_step(fine, Re, beta_squared, lid_velocity, CFL, corner_treatment, Lx, Ly)
erms_2sg = compute_continuity_rms(fine)
print(f"2 SG steps: E_RMS={erms_2sg:.6e}, improvement={(erms_2sg - e_rms_initial)/e_rms_initial*100:+.2f}%")
