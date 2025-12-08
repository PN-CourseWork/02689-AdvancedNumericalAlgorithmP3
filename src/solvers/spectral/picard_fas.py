"""Picard-FAS (Full Approximation Storage) V-Cycle Multigrid Solver.

This solver combines:
1. Picard iteration (fixed-point) for the nonlinear NS equations
2. FAS multigrid V-cycle to accelerate the linear solve at each Picard step

The key insight is that FAS with tau correction works well for LINEAR problems.
By linearizing the convection term (freezing the advecting velocity), each
Picard step becomes a linear solve that FAS can accelerate effectively.

Picard iteration for steady NS:
    Given u^k, solve for u^{k+1}:
    (u^k · ∇)u^{k+1} + ∇p^{k+1} - (1/Re)∇²u^{k+1} = 0
    ∇ · u^{k+1} = 0

This is a linear system in (u^{k+1}, p^{k+1}) with frozen advection velocity u^k.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .fas import (
    FASLevel,
    build_fas_hierarchy,
    enforce_boundary_conditions,
    restrict_solution,
    restrict_residual,
    prolongate_correction,
    compute_continuity_rms,
)
from .sg import SGSolver
from .operators.transfer_operators import (
    FFTProlongation,
    FFTRestriction,
    InjectionRestriction,
)

log = logging.getLogger(__name__)


# =============================================================================
# Picard-linearized residual computation
# =============================================================================


def compute_picard_residuals(
    level: FASLevel,
    u: np.ndarray,
    v: np.ndarray,
    p: np.ndarray,
    u_adv: np.ndarray,
    v_adv: np.ndarray,
    Re: float,
    beta_squared: float,
) -> None:
    """Compute residuals with FROZEN advection velocity (Picard linearization).

    The linearized momentum equations are:
        R_u = -(u_adv · ∇)u - ∇p + (1/Re)∇²u
        R_v = -(u_adv · ∇)v - ∇p + (1/Re)∇²v

    where u_adv, v_adv are FROZEN (from previous Picard iteration).

    This makes the system LINEAR in (u, v, p)!

    Stores results in level.R_u, level.R_v, level.R_p.
    """
    # Reshape to 2D for tensor product operations
    u_2d = u.reshape(level.shape_full)
    v_2d = v.reshape(level.shape_full)
    u_adv_2d = u_adv.reshape(level.shape_full)
    v_adv_2d = v_adv.reshape(level.shape_full)

    # Compute velocity derivatives using tensor products
    du_dx_2d = level.Dx_1d @ u_2d
    du_dy_2d = u_2d @ level.Dy_1d.T
    dv_dx_2d = level.Dx_1d @ v_2d
    dv_dy_2d = v_2d @ level.Dy_1d.T

    # Store flattened derivatives
    level.du_dx[:] = du_dx_2d.ravel()
    level.du_dy[:] = du_dy_2d.ravel()
    level.dv_dx[:] = dv_dx_2d.ravel()
    level.dv_dy[:] = dv_dy_2d.ravel()

    # Compute Laplacians
    lap_u_2d = level.Dxx_1d @ u_2d + u_2d @ level.Dyy_1d.T
    lap_v_2d = level.Dxx_1d @ v_2d + v_2d @ level.Dyy_1d.T
    level.lap_u[:] = lap_u_2d.ravel()
    level.lap_v[:] = lap_v_2d.ravel()

    # Pressure gradient
    p_inner_2d = p.reshape(level.shape_inner)
    p_full_2d = level.Interp_x @ p_inner_2d @ level.Interp_y.T
    level.dp_dx[:] = (level.Dx_1d @ p_full_2d).ravel()
    level.dp_dy[:] = (p_full_2d @ level.Dy_1d.T).ravel()

    # PICARD linearized convection: (u_adv · ∇)u (frozen advection!)
    conv_u = u_adv * level.du_dx + v_adv * level.du_dy
    conv_v = u_adv * level.dv_dx + v_adv * level.dv_dy

    nu = 1.0 / Re

    # Momentum residuals (LINEAR in u, v, p)
    level.R_u[:] = -conv_u - level.dp_dx + nu * level.lap_u
    level.R_v[:] = -conv_v - level.dp_dy + nu * level.lap_v

    # Add FAS tau correction if set
    if level.tau_u is not None:
        level.R_u[:] += level.tau_u
    if level.tau_v is not None:
        level.R_v[:] += level.tau_v

    # Continuity residual
    divergence_2d = du_dx_2d + dv_dy_2d
    divergence_inner = divergence_2d[1:-1, 1:-1].ravel()
    level.R_p[:] = -beta_squared * divergence_inner

    if level.tau_p is not None:
        level.R_p[:] += level.tau_p


def compute_adaptive_timestep_picard(
    level: FASLevel,
    u_adv: np.ndarray,
    v_adv: np.ndarray,
    Re: float,
    beta_squared: float,
    lid_velocity: float,
    CFL: float,
) -> float:
    """Compute adaptive timestep based on FROZEN advection velocity."""
    u_max = max(np.max(np.abs(u_adv)), lid_velocity)
    v_max = max(np.max(np.abs(v_adv)), 1e-10)
    nu = 1.0 / Re

    lambda_x = (u_max + np.sqrt(u_max**2 + beta_squared)) / level.dx_min + nu / level.dx_min**2
    lambda_y = (v_max + np.sqrt(v_max**2 + beta_squared)) / level.dy_min + nu / level.dy_min**2

    return CFL / (lambda_x + lambda_y)


def picard_rk4_step(
    level: FASLevel,
    u_adv: np.ndarray,
    v_adv: np.ndarray,
    Re: float,
    beta_squared: float,
    lid_velocity: float,
    CFL: float,
    corner_treatment,
    Lx: float,
    Ly: float,
) -> None:
    """Perform one RK4 pseudo time-step with FROZEN advection (Picard).

    This is a smoother for the LINEAR system arising from Picard linearization.
    """
    dt = compute_adaptive_timestep_picard(level, u_adv, v_adv, Re, beta_squared, lid_velocity, CFL)

    rk4_coeffs = [0.25, 1.0/3.0, 0.5, 1.0]
    u_in, v_in, p_in = level.u, level.v, level.p

    for i, alpha in enumerate(rk4_coeffs):
        compute_picard_residuals(level, u_in, v_in, p_in, u_adv, v_adv, Re, beta_squared)

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


# =============================================================================
# Picard-FAS V-Cycle
# =============================================================================


def restrict_advection_velocity(
    fine: FASLevel,
    coarse: FASLevel,
    u_adv_fine: np.ndarray,
    v_adv_fine: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Restrict the frozen advection velocity to coarse grid.

    Uses injection (same as solution restriction) for consistency.
    """
    injection = InjectionRestriction()

    u_adv_fine_2d = u_adv_fine.reshape(fine.shape_full)
    v_adv_fine_2d = v_adv_fine.reshape(fine.shape_full)

    u_adv_coarse = injection.restrict_2d(u_adv_fine_2d, coarse.shape_full).ravel()
    v_adv_coarse = injection.restrict_2d(v_adv_fine_2d, coarse.shape_full).ravel()

    return u_adv_coarse, v_adv_coarse


def picard_fas_vcycle(
    levels: List[FASLevel],
    level_idx: int,
    u_adv_list: List[np.ndarray],
    v_adv_list: List[np.ndarray],
    Re: float,
    beta_squared: float,
    lid_velocity: float,
    CFL: float,
    corner_treatment,
    Lx: float,
    Ly: float,
    pre_smooth: int = 2,
    post_smooth: int = 2,
    coarse_solve_iters: int = 10,
) -> None:
    """Perform one Picard-FAS V-cycle with frozen advection velocity.

    This V-cycle solves the LINEAR system arising from Picard linearization.
    The tau correction now makes perfect sense!

    Parameters
    ----------
    levels : List[FASLevel]
        Grid hierarchy (index 0 = coarsest)
    level_idx : int
        Current level index
    u_adv_list, v_adv_list : List[np.ndarray]
        Frozen advection velocities for each level
    pre_smooth, post_smooth : int
        Smoothing iterations (now can use more since it's a linear system!)
    coarse_solve_iters : int
        Iterations on coarsest grid (can solve more thoroughly)
    """
    level = levels[level_idx]
    u_adv = u_adv_list[level_idx]
    v_adv = v_adv_list[level_idx]

    # 1. Pre-smoothing
    for _ in range(pre_smooth):
        picard_rk4_step(level, u_adv, v_adv, Re, beta_squared, lid_velocity, CFL, corner_treatment, Lx, Ly)

    # 2. Coarse grid correction (if not coarsest)
    if level_idx > 0:
        coarse = levels[level_idx - 1]
        u_adv_coarse = u_adv_list[level_idx - 1]
        v_adv_coarse = v_adv_list[level_idx - 1]

        # 2a. Save current fine solution
        u_h_old = level.u.copy()
        v_h_old = level.v.copy()
        p_h_old = level.p.copy()

        # 2b. Compute fine grid residual (with Picard linearization)
        compute_picard_residuals(level, level.u, level.v, level.p, u_adv, v_adv, Re, beta_squared)

        # 2c. Restrict residual (FFT + truncation)
        I_r_u, I_r_v, I_r_p = restrict_residual(level, coarse)

        # 2d. Restrict solution (direct injection)
        restrict_solution(level, coarse)
        enforce_boundary_conditions(coarse, coarse.u, coarse.v, lid_velocity, corner_treatment, Lx, Ly)

        u_H_old = coarse.u.copy()
        v_H_old = coarse.v.copy()
        p_H_old = coarse.p.copy()

        # 2e. Compute coarse residual at restricted solution
        compute_picard_residuals(coarse, coarse.u, coarse.v, coarse.p, u_adv_coarse, v_adv_coarse, Re, beta_squared)

        # Zero coarse residual at boundaries
        R_u_2d = coarse.R_u.reshape(coarse.shape_full)
        R_v_2d = coarse.R_v.reshape(coarse.shape_full)
        R_u_2d[0, :] = 0.0; R_u_2d[-1, :] = 0.0; R_u_2d[:, 0] = 0.0; R_u_2d[:, -1] = 0.0
        R_v_2d[0, :] = 0.0; R_v_2d[-1, :] = 0.0; R_v_2d[:, 0] = 0.0; R_v_2d[:, -1] = 0.0

        # 2f. Compute tau correction: tau = I(r_h) - r_H'
        tau_u = I_r_u - coarse.R_u
        tau_v = I_r_v - coarse.R_v
        tau_p = I_r_p - coarse.R_p

        # 2g. Set tau and recurse
        coarse.tau_u = tau_u
        coarse.tau_v = tau_v
        coarse.tau_p = tau_p

        picard_fas_vcycle(
            levels, level_idx - 1,
            u_adv_list, v_adv_list,
            Re, beta_squared, lid_velocity, CFL, corner_treatment, Lx, Ly,
            pre_smooth, post_smooth, coarse_solve_iters
        )

        # Clear tau
        coarse.tau_u = None
        coarse.tau_v = None
        coarse.tau_p = None

        # 2h. Compute correction
        e_u = coarse.u - u_H_old
        e_v = coarse.v - v_H_old
        e_p = coarse.p - p_H_old

        # 2i. Prolongate correction
        e_u_fine, e_v_fine, e_p_fine = prolongate_correction(coarse, level, e_u, e_v, e_p)

        # 2j. Apply correction (no damping)
        level.u[:] = u_h_old + e_u_fine
        level.v[:] = v_h_old + e_v_fine
        level.p[:] = p_h_old + e_p_fine

        # 2k. Re-enforce boundary conditions
        enforce_boundary_conditions(level, level.u, level.v, lid_velocity, corner_treatment, Lx, Ly)

    else:
        # On coarsest grid: do more iterations (solve more thoroughly)
        for _ in range(coarse_solve_iters - pre_smooth):
            picard_rk4_step(level, u_adv, v_adv, Re, beta_squared, lid_velocity, CFL, corner_treatment, Lx, Ly)

    # 3. Post-smoothing
    for _ in range(post_smooth):
        picard_rk4_step(level, u_adv, v_adv, Re, beta_squared, lid_velocity, CFL, corner_treatment, Lx, Ly)


# =============================================================================
# Picard-FAS Solver Class
# =============================================================================


class PicardFASSolver(SGSolver):
    """Picard-FAS Multigrid Solver.

    Combines Picard iteration (outer loop) with FAS V-cycle (inner loop).

    Outer loop: Picard iteration
        - Freeze advection velocity u_adv = u^k
        - Solve linearized NS for u^{k+1}

    Inner loop: FAS V-cycle
        - Accelerate the linear solve using multigrid
        - tau correction works correctly for linear systems!

    Parameters
    ----------
    All parameters from SGSolver, plus:
        n_levels : int
            Number of multigrid levels
        pre_smooth, post_smooth : int
            Smoothing iterations per V-cycle
        coarse_solve_iters : int
            Iterations on coarsest grid
        vcycles_per_picard : int
            Number of V-cycles per Picard iteration
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Picard-FAS parameters
        self.n_levels = getattr(self.params, 'n_levels', 3)
        self.pre_smooth = getattr(self.params, 'pre_smooth', 2)
        self.post_smooth = getattr(self.params, 'post_smooth', 2)
        self.coarse_solve_iters = getattr(self.params, 'coarse_solve_iters', 10)
        self.vcycles_per_picard = getattr(self.params, 'vcycles_per_picard', 5)
        self.coarsest_n = getattr(self.params, 'coarsest_n', 12)

        # Build hierarchy
        self._levels = build_fas_hierarchy(
            n_fine=self.params.nx,
            n_levels=self.n_levels,
            basis_x=self.basis_x,
            basis_y=self.basis_y,
            Lx=self.params.Lx,
            Ly=self.params.Ly,
            coarsest_n=self.coarsest_n,
        )

        log.info(
            f"PicardFASSolver initialized: {len(self._levels)} levels, "
            f"N = {[l.n for l in self._levels]}, "
            f"pre_smooth={self.pre_smooth}, post_smooth={self.post_smooth}, "
            f"vcycles_per_picard={self.vcycles_per_picard}"
        )

    def _build_advection_hierarchy(self, u_adv: np.ndarray, v_adv: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Build restricted advection velocities for all levels."""
        u_adv_list = [None] * len(self._levels)
        v_adv_list = [None] * len(self._levels)

        # Finest level
        u_adv_list[-1] = u_adv.copy()
        v_adv_list[-1] = v_adv.copy()

        # Restrict to coarser levels
        for i in range(len(self._levels) - 2, -1, -1):
            fine = self._levels[i + 1]
            coarse = self._levels[i]
            u_adv_list[i], v_adv_list[i] = restrict_advection_velocity(
                fine, coarse, u_adv_list[i + 1], v_adv_list[i + 1]
            )

        return u_adv_list, v_adv_list

    def solve(self, tolerance: float = None, max_iter: int = None):
        """Solve using Picard iteration with FAS V-cycle acceleration.

        Parameters
        ----------
        tolerance : float, optional
            Convergence tolerance (E_RMS). Default from params.
        max_iter : int, optional
            Maximum Picard iterations. Default from params.
        """
        import time

        if tolerance is None:
            tolerance = self.params.tolerance
        if max_iter is None:
            max_iter = self.params.max_iterations

        # Initialize finest level
        finest = self._levels[-1]
        finest.u[:] = 0.0
        finest.v[:] = 0.0
        finest.p[:] = 0.0

        # Initialize lid BC
        enforce_boundary_conditions(
            finest, finest.u, finest.v,
            self.params.lid_velocity, self.corner_treatment,
            self.params.Lx, self.params.Ly
        )

        log.info(f"Picard-FAS solve: tol={tolerance:.2e}, max_picard_iter={max_iter}")

        time_start = time.time()
        converged = False
        total_vcycles = 0

        for picard_iter in range(max_iter):
            # Freeze advection velocity for this Picard iteration
            u_adv = finest.u.copy()
            v_adv = finest.v.copy()

            # Build advection velocity hierarchy
            u_adv_list, v_adv_list = self._build_advection_hierarchy(u_adv, v_adv)

            # Perform V-cycles to solve the linearized system
            for vcycle in range(self.vcycles_per_picard):
                picard_fas_vcycle(
                    self._levels,
                    level_idx=len(self._levels) - 1,
                    u_adv_list=u_adv_list,
                    v_adv_list=v_adv_list,
                    Re=self.params.Re,
                    beta_squared=self.params.beta_squared,
                    lid_velocity=self.params.lid_velocity,
                    CFL=self.params.CFL,
                    corner_treatment=self.corner_treatment,
                    Lx=self.params.Lx,
                    Ly=self.params.Ly,
                    pre_smooth=self.pre_smooth,
                    post_smooth=self.post_smooth,
                    coarse_solve_iters=self.coarse_solve_iters,
                )
                total_vcycles += 1

            # Check convergence (using full nonlinear residual)
            erms = compute_continuity_rms(finest)

            if picard_iter % 10 == 0 or erms < tolerance:
                log.info(f"Picard iter {picard_iter + 1}: E_RMS = {erms:.6e}, total V-cycles = {total_vcycles}")

            if erms < tolerance:
                converged = True
                log.info(f"Picard-FAS converged in {picard_iter + 1} Picard iterations ({total_vcycles} V-cycles), E_RMS = {erms:.6e}")
                break

        time_end = time.time()
        wall_time = time_end - time_start

        if not converged:
            log.warning(f"Picard-FAS did not converge after {max_iter} Picard iterations, E_RMS = {erms:.6e}")

        # Copy solution to output arrays
        self.arrays.u[:] = finest.u
        self.arrays.v[:] = finest.v
        p_inner_2d = finest.p.reshape(finest.shape_inner)
        p_full_2d = finest.Interp_x @ p_inner_2d @ finest.Interp_y.T
        self.arrays.p[:] = p_full_2d.ravel()[:len(self.arrays.p)]

        # Store results
        self._compute_residuals(self.arrays.u, self.arrays.v, self.arrays.p)
        final_residuals = self._compute_algebraic_residuals()

        residual_history = [{
            "rel_iter": erms,
            "u_eq": final_residuals["u_residual"],
            "v_eq": final_residuals["v_residual"],
            "continuity": final_residuals["continuity_residual"],
        }]

        self._store_results(
            residual_history=residual_history,
            final_iter_count=total_vcycles,
            is_converged=converged,
            wall_time=wall_time,
            energy_history=[self._compute_energy()],
            enstrophy_history=[self._compute_enstrophy()],
            palinstrophy_history=[self._compute_palinstrophy()],
        )

        log.info(f"Picard-FAS completed in {wall_time:.2f}s")
