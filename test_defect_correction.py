"""Test defect correction multigrid approach.

Instead of FAS (solving for full approximation), use defect correction:
1. Compute residual on fine: r_h = R(u_h)
2. Solve for CORRECTION on coarse: A_H e_H = I(r_h)
3. Prolongate: u_h = u_h + P(e_H)

For Navier-Stokes with pseudo time-stepping:
- The "A_H e_H = I(r_h)" becomes: find e such that R_linear(e) = I(r_h)
- We linearize around ZERO (since e is small correction)
- This gives: (u_adv · ∇)e + ∇p_e - (1/Re)∇²e = I(r_h)

Wait... but linearizing around zero doesn't make sense for advection!

Let's try a simpler approach: just use the coarse grid to smooth
the correction equation, starting from e=0.
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
    FASLevel,
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

fine = levels[1]
coarse = levels[0]

print(f"Fine: N={fine.n}, Coarse: N={coarse.n}")

# Initialize and run to establish flow
fine.u[:] = 0.0
fine.v[:] = 0.0
fine.p[:] = 0.0
enforce_boundary_conditions(fine, fine.u, fine.v, lid_velocity, corner_treatment, Lx, Ly)

print("Running 100 SG iterations...")
for i in range(100):
    fas_rk4_step(fine, Re, beta_squared, lid_velocity, CFL, corner_treatment, Lx, Ly)

e_rms_initial = compute_continuity_rms(fine)
print(f"Initial E_RMS: {e_rms_initial:.6e}")

u_initial = fine.u.copy()
v_initial = fine.v.copy()
p_initial = fine.p.copy()


def compute_linear_residuals_for_correction(
    level: FASLevel,
    e_u: np.ndarray,
    e_v: np.ndarray,
    e_p: np.ndarray,
    u_base: np.ndarray,
    v_base: np.ndarray,
    Re: float,
    beta_squared: float,
    source_u: np.ndarray,
    source_v: np.ndarray,
    source_p: np.ndarray,
) -> None:
    """Compute residuals for the correction equation.

    We want to solve: L(e) = source
    where L is the linearized NS operator around (u_base, v_base).

    For Picard linearization:
        L(e)_u = (u_base · ∇)e_u + ∇e_p - (1/Re)∇²e_u
        L(e)_v = (u_base · ∇)e_v + ∇e_p - (1/Re)∇²e_v
        L(e)_p = ∇ · e

    Residual = source - L(e)
    """
    # Reshape
    e_u_2d = e_u.reshape(level.shape_full)
    e_v_2d = e_v.reshape(level.shape_full)
    u_base_2d = u_base.reshape(level.shape_full)
    v_base_2d = v_base.reshape(level.shape_full)

    # Compute correction derivatives
    de_u_dx_2d = level.Dx_1d @ e_u_2d
    de_u_dy_2d = e_u_2d @ level.Dy_1d.T
    de_v_dx_2d = level.Dx_1d @ e_v_2d
    de_v_dy_2d = e_v_2d @ level.Dy_1d.T

    # Laplacians of corrections
    lap_e_u_2d = level.Dxx_1d @ e_u_2d + e_u_2d @ level.Dyy_1d.T
    lap_e_v_2d = level.Dxx_1d @ e_v_2d + e_v_2d @ level.Dyy_1d.T

    # Pressure gradient of correction
    e_p_inner_2d = e_p.reshape(level.shape_inner)
    e_p_full_2d = level.Interp_x @ e_p_inner_2d @ level.Interp_y.T
    de_p_dx = (level.Dx_1d @ e_p_full_2d).ravel()
    de_p_dy = (e_p_full_2d @ level.Dy_1d.T).ravel()

    # Linearized convection: (u_base · ∇)e
    conv_e_u = u_base * de_u_dx_2d.ravel() + v_base * de_u_dy_2d.ravel()
    conv_e_v = u_base * de_v_dx_2d.ravel() + v_base * de_v_dy_2d.ravel()

    nu = 1.0 / Re

    # L(e) = conv + ∇p - ν∇²e
    L_e_u = conv_e_u + de_p_dx - nu * lap_e_u_2d.ravel()
    L_e_v = conv_e_v + de_p_dy - nu * lap_e_v_2d.ravel()

    # Divergence of correction
    div_e_2d = de_u_dx_2d + de_v_dy_2d
    div_e_inner = div_e_2d[1:-1, 1:-1].ravel()
    L_e_p = beta_squared * div_e_inner

    # Residual = source - L(e)
    level.R_u[:] = source_u - L_e_u
    level.R_v[:] = source_v - L_e_v
    level.R_p[:] = source_p - L_e_p


def defect_correction_vcycle(n_coarse_iters=5):
    """Defect correction multigrid approach.

    1. Pre-smooth fine
    2. Compute fine residual r_h
    3. Restrict r_h to coarse
    4. Solve L_H(e_H) = I(r_h) on coarse (starting from e=0)
    5. Prolongate e_H
    6. Update fine: u_h += P(e_H)
    7. Post-smooth fine
    """
    from solvers.spectral.fas import compute_adaptive_timestep
    from solvers.spectral.operators.transfer_operators import InjectionRestriction

    # Reset
    fine.u[:] = u_initial.copy()
    fine.v[:] = v_initial.copy()
    fine.p[:] = p_initial.copy()

    # 1. Pre-smooth on fine
    fas_rk4_step(fine, Re, beta_squared, lid_velocity, CFL, corner_treatment, Lx, Ly)

    # 2. Compute fine residual
    compute_residuals(fine, fine.u, fine.v, fine.p, Re, beta_squared)

    # 3. Restrict residual to coarse (this is our source term)
    I_r_u, I_r_v, I_r_p = restrict_residual(fine, coarse)

    # Also restrict the base velocity for linearization
    injection = InjectionRestriction()
    u_base_fine_2d = fine.u.reshape(fine.shape_full)
    v_base_fine_2d = fine.v.reshape(fine.shape_full)
    u_base_coarse = injection.restrict_2d(u_base_fine_2d, coarse.shape_full).ravel()
    v_base_coarse = injection.restrict_2d(v_base_fine_2d, coarse.shape_full).ravel()

    # 4. Solve for correction on coarse: L_H(e_H) = I(r_h)
    # Start with e = 0
    e_u = np.zeros_like(coarse.u)
    e_v = np.zeros_like(coarse.v)
    e_p = np.zeros_like(coarse.p)

    # Use pseudo time-stepping to solve the correction equation
    for _ in range(n_coarse_iters):
        dt = compute_adaptive_timestep(coarse, Re, beta_squared, lid_velocity, CFL)

        # RK4 for correction equation
        rk4_coeffs = [0.25, 1.0/3.0, 0.5, 1.0]
        e_u_in, e_v_in, e_p_in = e_u, e_v, e_p

        for i, alpha in enumerate(rk4_coeffs):
            compute_linear_residuals_for_correction(
                coarse, e_u_in, e_v_in, e_p_in,
                u_base_coarse, v_base_coarse,
                Re, beta_squared,
                I_r_u, I_r_v, I_r_p
            )

            if i < 3:
                coarse.u_stage[:] = e_u + alpha * dt * coarse.R_u
                coarse.v_stage[:] = e_v + alpha * dt * coarse.R_v
                coarse.p_stage[:] = e_p + alpha * dt * coarse.R_p
                # BCs on correction: e = 0 on boundaries
                coarse.u_stage.reshape(coarse.shape_full)[0, :] = 0.0
                coarse.u_stage.reshape(coarse.shape_full)[-1, :] = 0.0
                coarse.u_stage.reshape(coarse.shape_full)[:, 0] = 0.0
                coarse.u_stage.reshape(coarse.shape_full)[:, -1] = 0.0
                coarse.v_stage.reshape(coarse.shape_full)[0, :] = 0.0
                coarse.v_stage.reshape(coarse.shape_full)[-1, :] = 0.0
                coarse.v_stage.reshape(coarse.shape_full)[:, 0] = 0.0
                coarse.v_stage.reshape(coarse.shape_full)[:, -1] = 0.0
                e_u_in, e_v_in, e_p_in = coarse.u_stage, coarse.v_stage, coarse.p_stage
            else:
                e_u[:] = e_u + alpha * dt * coarse.R_u
                e_v[:] = e_v + alpha * dt * coarse.R_v
                e_p[:] = e_p + alpha * dt * coarse.R_p
                # BCs
                e_u.reshape(coarse.shape_full)[0, :] = 0.0
                e_u.reshape(coarse.shape_full)[-1, :] = 0.0
                e_u.reshape(coarse.shape_full)[:, 0] = 0.0
                e_u.reshape(coarse.shape_full)[:, -1] = 0.0
                e_v.reshape(coarse.shape_full)[0, :] = 0.0
                e_v.reshape(coarse.shape_full)[-1, :] = 0.0
                e_v.reshape(coarse.shape_full)[:, 0] = 0.0
                e_v.reshape(coarse.shape_full)[:, -1] = 0.0

    # 5. Prolongate correction
    e_u_fine, e_v_fine, e_p_fine = prolongate_correction(coarse, fine, e_u, e_v, e_p)

    # 6. Apply correction
    fine.u[:] += e_u_fine
    fine.v[:] += e_v_fine
    fine.p[:] += e_p_fine
    enforce_boundary_conditions(fine, fine.u, fine.v, lid_velocity, corner_treatment, Lx, Ly)

    # 7. Post-smooth
    fas_rk4_step(fine, Re, beta_squared, lid_velocity, CFL, corner_treatment, Lx, Ly)

    return compute_continuity_rms(fine)


print("\n" + "="*60)
print("=== TEST: DEFECT CORRECTION (solve for e, not full solution) ===")
print("="*60)

for n_coarse in [1, 2, 5, 10, 20]:
    try:
        erms = defect_correction_vcycle(n_coarse)
        improvement = (erms - e_rms_initial)/e_rms_initial*100
        print(f"{n_coarse:2d} coarse iters: E_RMS={erms:.6e}, improvement={improvement:+.2f}%")
    except Exception as e:
        print(f"{n_coarse:2d} coarse iters: ERROR - {e}")


# Baseline
print("\n--- Baseline ---")
fine.u[:] = u_initial.copy()
fine.v[:] = v_initial.copy()
fine.p[:] = p_initial.copy()
fas_rk4_step(fine, Re, beta_squared, lid_velocity, CFL, corner_treatment, Lx, Ly)
fas_rk4_step(fine, Re, beta_squared, lid_velocity, CFL, corner_treatment, Lx, Ly)
fas_rk4_step(fine, Re, beta_squared, lid_velocity, CFL, corner_treatment, Lx, Ly)  # +1 for post-smooth equiv
erms_3sg = compute_continuity_rms(fine)
print(f"3 SG steps: E_RMS={erms_3sg:.6e}, improvement={(erms_3sg - e_rms_initial)/e_rms_initial*100:+.2f}%")
