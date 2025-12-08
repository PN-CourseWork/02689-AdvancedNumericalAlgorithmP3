"""FAS (Full Approximation Storage) V-Cycle Multigrid Solver.

Implementation following Zhang & Xi (2010):
"An explicit Chebyshev pseudospectral multigrid method for incompressible
Navier-Stokes equations", Computers & Fluids 39 (2010) 178-188.

Key features (paper-faithful):
- Pre-smoothing: 1 RK4 step per level (configurable)
- Post-smoothing: 0 steps (paper says "unnecessary" and "sometimes brings instabilities")
- No damping on coarse grid correction (damping = 1.0)
- Convergence criterion: RMS of continuity residual (divergence)
- Transfer operators:
  - Restrict variables: Direct injection (coarse GLL nodes are subset of fine)
  - Restrict residuals: FFT + high-frequency truncation
  - Prolongate corrections: FFT (Chebyshev interpolation)

Usage:
    solver = FASSolver(nx=64, ny=64, Re=100, n_levels=3)
    solver.solve()
"""

import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np

from .sg import SGSolver
from .operators.transfer_operators import (
    FFTProlongation,
    FFTRestriction,
    InjectionRestriction,
)
from .operators.corner import create_corner_treatment

log = logging.getLogger(__name__)


# =============================================================================
# FASLevel: Clean data structure for one multigrid level
# =============================================================================


@dataclass
class FASLevel:
    """Data structure for one FAS multigrid level.

    Minimal and clean - only what's needed for FAS V-cycle.

    Attributes
    ----------
    n : int
        Polynomial order (n+1 nodes per dimension)
    level_idx : int
        Level index (0 = coarsest, higher = finer)
    """

    # Grid info
    n: int
    level_idx: int
    shape_full: Tuple[int, int]   # (n+1, n+1) for velocities
    shape_inner: Tuple[int, int]  # (n-1, n-1) for pressure

    # 1D node arrays
    x_nodes: np.ndarray
    y_nodes: np.ndarray

    # Grid spacing for CFL
    dx_min: float
    dy_min: float

    # 2D meshgrids
    X: np.ndarray  # Full grid
    Y: np.ndarray

    # Differentiation matrices (2D Kronecker form)
    Dx: np.ndarray      # d/dx on full grid
    Dy: np.ndarray      # d/dy on full grid
    Laplacian: np.ndarray

    # Interpolation from inner to full grid (for pressure gradient)
    Interp_x: np.ndarray
    Interp_y: np.ndarray

    # Solution arrays (flattened)
    u: np.ndarray       # Full grid
    v: np.ndarray       # Full grid
    p: np.ndarray       # Inner grid

    # RK4 staging arrays
    u_stage: np.ndarray
    v_stage: np.ndarray
    p_stage: np.ndarray

    # Residuals
    R_u: np.ndarray     # Full grid
    R_v: np.ndarray     # Full grid
    R_p: np.ndarray     # Inner grid (continuity)

    # Work buffers for derivatives
    du_dx: np.ndarray
    du_dy: np.ndarray
    dv_dx: np.ndarray
    dv_dy: np.ndarray
    lap_u: np.ndarray
    lap_v: np.ndarray
    dp_dx: np.ndarray   # Full grid (interpolated)
    dp_dy: np.ndarray

    # FAS tau correction (set during coarse grid solve, None otherwise)
    tau_u: Optional[np.ndarray] = None
    tau_v: Optional[np.ndarray] = None
    tau_p: Optional[np.ndarray] = None


def _build_interpolation_matrix_1d(nodes_inner: np.ndarray, nodes_full: np.ndarray) -> np.ndarray:
    """Build spectral interpolation matrix from inner to full grid.

    Uses Chebyshev polynomial interpolation for spectral accuracy.
    """
    from numpy.polynomial.chebyshev import chebvander

    n_inner = len(nodes_inner)

    # Map to [-1, 1] for Chebyshev
    a, b = nodes_full[0], nodes_full[-1]
    xi_inner = 2 * (nodes_inner - a) / (b - a) - 1
    xi_full = 2 * (nodes_full - a) / (b - a) - 1

    # Vandermonde matrices
    V_inner = chebvander(xi_inner, n_inner - 1)
    V_full = chebvander(xi_full, n_inner - 1)

    # Interpolation matrix: f_full = Interp @ f_inner
    Interp = V_full @ np.linalg.solve(V_inner, np.eye(n_inner))

    return Interp


def build_fas_level(
    n: int,
    level_idx: int,
    basis_x,
    basis_y,
    Lx: float = 1.0,
    Ly: float = 1.0,
) -> FASLevel:
    """Build a single FAS level with all operators and arrays.

    Parameters
    ----------
    n : int
        Polynomial order (n+1 nodes per dimension)
    level_idx : int
        Level index in hierarchy (0 = coarsest)
    basis_x, basis_y : Basis objects
        Spectral basis (Chebyshev or Legendre Lobatto)
    Lx, Ly : float
        Domain dimensions

    Returns
    -------
    FASLevel
        Fully initialized level
    """
    shape_full = (n + 1, n + 1)
    shape_inner = (n - 1, n - 1)
    n_full = shape_full[0] * shape_full[1]
    n_inner = shape_inner[0] * shape_inner[1]

    # 1D nodes
    x_nodes = basis_x.nodes(n + 1)
    y_nodes = basis_y.nodes(n + 1)

    # Grid spacing
    dx_min = np.min(np.diff(x_nodes))
    dy_min = np.min(np.diff(y_nodes))

    # 2D meshgrid
    X, Y = np.meshgrid(x_nodes, y_nodes, indexing="ij")

    # Differentiation matrices
    Dx_1d = basis_x.diff_matrix(x_nodes)
    Dy_1d = basis_y.diff_matrix(y_nodes)

    Ix = np.eye(n + 1)
    Iy = np.eye(n + 1)
    Dx = np.kron(Dx_1d, Iy)
    Dy = np.kron(Ix, Dy_1d)

    Dxx_1d = Dx_1d @ Dx_1d
    Dyy_1d = Dy_1d @ Dy_1d
    Dxx = np.kron(Dxx_1d, Iy)
    Dyy = np.kron(Ix, Dyy_1d)
    Laplacian = Dxx + Dyy

    # Interpolation matrices for pressure gradient
    x_inner = x_nodes[1:-1]
    y_inner = y_nodes[1:-1]
    Interp_x = _build_interpolation_matrix_1d(x_inner, x_nodes)
    Interp_y = _build_interpolation_matrix_1d(y_inner, y_nodes)

    return FASLevel(
        n=n,
        level_idx=level_idx,
        shape_full=shape_full,
        shape_inner=shape_inner,
        x_nodes=x_nodes,
        y_nodes=y_nodes,
        dx_min=dx_min,
        dy_min=dy_min,
        X=X,
        Y=Y,
        Dx=Dx,
        Dy=Dy,
        Laplacian=Laplacian,
        Interp_x=Interp_x,
        Interp_y=Interp_y,
        # Solution arrays
        u=np.zeros(n_full),
        v=np.zeros(n_full),
        p=np.zeros(n_inner),
        u_stage=np.zeros(n_full),
        v_stage=np.zeros(n_full),
        p_stage=np.zeros(n_inner),
        # Residuals
        R_u=np.zeros(n_full),
        R_v=np.zeros(n_full),
        R_p=np.zeros(n_inner),
        # Work buffers
        du_dx=np.zeros(n_full),
        du_dy=np.zeros(n_full),
        dv_dx=np.zeros(n_full),
        dv_dy=np.zeros(n_full),
        lap_u=np.zeros(n_full),
        lap_v=np.zeros(n_full),
        dp_dx=np.zeros(n_full),
        dp_dy=np.zeros(n_full),
    )


def build_fas_hierarchy(
    n_fine: int,
    n_levels: int,
    basis_x,
    basis_y,
    Lx: float = 1.0,
    Ly: float = 1.0,
    coarsest_n: int = 12,
) -> List[FASLevel]:
    """Build FAS multigrid hierarchy.

    Uses full coarsening: N_coarse = N_fine / 2

    Parameters
    ----------
    n_fine : int
        Polynomial order on finest grid
    n_levels : int
        Number of multigrid levels (2 or 3 typically)
    basis_x, basis_y : Basis objects
        Spectral basis
    Lx, Ly : float
        Domain dimensions
    coarsest_n : int
        Minimum polynomial order for coarsest grid (default 12).
        Paper uses N=12 as minimum to resolve physics adequately.

    Returns
    -------
    List[FASLevel]
        Hierarchy with index 0 = coarsest, index -1 = finest
    """
    # Compute polynomial orders for each level
    orders = [n_fine]
    n = n_fine
    for _ in range(n_levels - 1):
        n = n // 2
        if n < coarsest_n:  # Minimum grid to resolve physics
            log.warning(f"Cannot create {n_levels} levels from N={n_fine} with coarsest_n={coarsest_n}, stopping at N={n*2}")
            break
        orders.append(n)

    # Reverse so coarsest is first
    orders = orders[::-1]

    log.info(f"FAS hierarchy: {len(orders)} levels, N = {orders}")

    # Build levels
    levels = []
    for idx, n in enumerate(orders):
        level = build_fas_level(n, idx, basis_x, basis_y, Lx, Ly)
        levels.append(level)

    return levels


# =============================================================================
# Core FAS Operations (Testable in Isolation)
# =============================================================================


def compute_residuals(
    level: FASLevel,
    u: np.ndarray,
    v: np.ndarray,
    p: np.ndarray,
    Re: float,
    beta_squared: float,
) -> None:
    """Compute residuals R_u, R_v, R_p for pseudo time-stepping.

    Stores results in level.R_u, level.R_v, level.R_p.

    Uses standard convection (no subtraction method - using smoothing for corners).

    Parameters
    ----------
    level : FASLevel
        Level data structure (modified in place)
    u, v : np.ndarray
        Velocity fields on full grid
    p : np.ndarray
        Pressure on inner grid
    Re : float
        Reynolds number
    beta_squared : float
        Artificial compressibility parameter
    """
    # Velocity derivatives
    level.du_dx[:] = level.Dx @ u
    level.du_dy[:] = level.Dy @ u
    level.dv_dx[:] = level.Dx @ v
    level.dv_dy[:] = level.Dy @ v

    # Laplacians
    level.lap_u[:] = level.Laplacian @ u
    level.lap_v[:] = level.Laplacian @ v

    # Pressure gradient: interpolate from inner to full grid, then differentiate
    p_inner_2d = p.reshape(level.shape_inner)
    p_full_2d = level.Interp_x @ p_inner_2d @ level.Interp_y.T
    p_full = p_full_2d.ravel()
    level.dp_dx[:] = level.Dx @ p_full
    level.dp_dy[:] = level.Dy @ p_full

    # Standard convection: (u·∇)u
    conv_u = u * level.du_dx + v * level.du_dy
    conv_v = u * level.dv_dx + v * level.dv_dy

    nu = 1.0 / Re

    # Momentum residuals
    level.R_u[:] = -conv_u - level.dp_dx + nu * level.lap_u
    level.R_v[:] = -conv_v - level.dp_dy + nu * level.lap_v

    # Add FAS tau correction if set
    if level.tau_u is not None:
        level.R_u[:] += level.tau_u
    if level.tau_v is not None:
        level.R_v[:] += level.tau_v

    # Continuity residual on inner grid: R_p = -β²(∂u/∂x + ∂v/∂y)
    divergence_full = level.du_dx + level.dv_dy
    divergence_2d = divergence_full.reshape(level.shape_full)
    divergence_inner = divergence_2d[1:-1, 1:-1].ravel()
    level.R_p[:] = -beta_squared * divergence_inner

    if level.tau_p is not None:
        level.R_p[:] += level.tau_p


def compute_adaptive_timestep(
    level: FASLevel,
    Re: float,
    beta_squared: float,
    lid_velocity: float,
    CFL: float,
) -> float:
    """Compute adaptive timestep based on CFL condition.

    From paper Eq. 8.
    """
    u_max = max(np.max(np.abs(level.u)), lid_velocity)
    v_max = max(np.max(np.abs(level.v)), 1e-10)
    nu = 1.0 / Re

    lambda_x = (u_max + np.sqrt(u_max**2 + beta_squared)) / level.dx_min + nu / level.dx_min**2
    lambda_y = (v_max + np.sqrt(v_max**2 + beta_squared)) / level.dy_min + nu / level.dy_min**2

    return CFL / (lambda_x + lambda_y)


def enforce_boundary_conditions(
    level: FASLevel,
    u: np.ndarray,
    v: np.ndarray,
    lid_velocity: float,
    corner_treatment,
    Lx: float,
    Ly: float,
) -> None:
    """Enforce boundary conditions on velocity fields.

    Modifies u, v in place.
    """
    u_2d = u.reshape(level.shape_full)
    v_2d = v.reshape(level.shape_full)

    # Get wall velocities (0 for smoothing method)
    # West
    u_wall, v_wall = corner_treatment.get_wall_velocity(
        level.X[0, :], level.Y[0, :], Lx, Ly
    )
    u_2d[0, :] = u_wall
    v_2d[0, :] = v_wall

    # East
    u_wall, v_wall = corner_treatment.get_wall_velocity(
        level.X[-1, :], level.Y[-1, :], Lx, Ly
    )
    u_2d[-1, :] = u_wall
    v_2d[-1, :] = v_wall

    # South
    u_wall, v_wall = corner_treatment.get_wall_velocity(
        level.X[:, 0], level.Y[:, 0], Lx, Ly
    )
    u_2d[:, 0] = u_wall
    v_2d[:, 0] = v_wall

    # North (lid)
    u_lid, v_lid = corner_treatment.get_lid_velocity(
        level.X[:, -1], level.Y[:, -1],
        lid_velocity=lid_velocity,
        Lx=Lx, Ly=Ly,
    )
    u_2d[:, -1] = u_lid
    v_2d[:, -1] = v_lid


def fas_rk4_step(
    level: FASLevel,
    Re: float,
    beta_squared: float,
    lid_velocity: float,
    CFL: float,
    corner_treatment,
    Lx: float,
    Ly: float,
) -> None:
    """Perform one RK4 pseudo time-step on a single level.

    4-stage explicit RK4 (paper Eq. 7):
        φ^(1) = φ^n + (1/4) Δτ R(φ^n)
        φ^(2) = φ^n + (1/3) Δτ R(φ^(1))
        φ^(3) = φ^n + (1/2) Δτ R(φ^(2))
        φ^(n+1) = φ^n + Δτ R(φ^(3))

    Modifies level.u, level.v, level.p in place.
    """
    dt = compute_adaptive_timestep(level, Re, beta_squared, lid_velocity, CFL)

    rk4_coeffs = [0.25, 1.0/3.0, 0.5, 1.0]
    u_in, v_in, p_in = level.u, level.v, level.p

    for i, alpha in enumerate(rk4_coeffs):
        compute_residuals(level, u_in, v_in, p_in, Re, beta_squared)

        if i < 3:
            # Intermediate stages: write to staging arrays
            level.u_stage[:] = level.u + alpha * dt * level.R_u
            level.v_stage[:] = level.v + alpha * dt * level.R_v
            level.p_stage[:] = level.p + alpha * dt * level.R_p
            enforce_boundary_conditions(
                level, level.u_stage, level.v_stage,
                lid_velocity, corner_treatment, Lx, Ly
            )
            u_in, v_in, p_in = level.u_stage, level.v_stage, level.p_stage
        else:
            # Final stage: write to main arrays
            level.u[:] = level.u + alpha * dt * level.R_u
            level.v[:] = level.v + alpha * dt * level.R_v
            level.p[:] = level.p + alpha * dt * level.R_p
            enforce_boundary_conditions(
                level, level.u, level.v,
                lid_velocity, corner_treatment, Lx, Ly
            )


def compute_continuity_rms(level: FASLevel) -> float:
    """Compute RMS of continuity residual (paper's E_RMS convergence criterion).

    E_RMS = sqrt( Σᵢⱼ (∂u/∂x + ∂v/∂y)²ᵢⱼ / ((Nx-1)(Ny-1)) )

    Computed on inner grid only.
    """
    # Compute divergence on full grid
    du_dx = level.Dx @ level.u
    dv_dy = level.Dy @ level.v
    div_full = du_dx + dv_dy

    # Extract inner grid
    div_2d = div_full.reshape(level.shape_full)
    div_inner = div_2d[1:-1, 1:-1].ravel()

    # RMS
    n_inner = len(div_inner)
    return np.sqrt(np.sum(div_inner**2) / n_inner)


# =============================================================================
# Transfer Operators (Using existing implementations)
# =============================================================================


def restrict_solution(
    fine: FASLevel,
    coarse: FASLevel,
) -> None:
    """Restrict solution from fine to coarse using direct injection.

    Paper Section 3.3: "the restriction of variables is accomplished
    simply by direct injection"

    Modifies coarse.u, coarse.v, coarse.p in place.
    """
    injection = InjectionRestriction()

    # Velocities (full grid)
    u_fine_2d = fine.u.reshape(fine.shape_full)
    v_fine_2d = fine.v.reshape(fine.shape_full)
    coarse.u[:] = injection.restrict_2d(u_fine_2d, coarse.shape_full).ravel()
    coarse.v[:] = injection.restrict_2d(v_fine_2d, coarse.shape_full).ravel()

    # Pressure (inner grid)
    p_fine_2d = fine.p.reshape(fine.shape_inner)
    coarse.p[:] = injection.restrict_2d(p_fine_2d, coarse.shape_inner).ravel()


def restrict_residual(
    fine: FASLevel,
    coarse: FASLevel,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Restrict residuals from fine to coarse using FFT + truncation.

    Paper Section 3.3: Uses Chebyshev coefficients with high-frequency truncation.

    IMPORTANT: Zero boundaries before restriction (BCs enforced separately).

    Returns
    -------
    I_r_u, I_r_v, I_r_p : np.ndarray
        Restricted residuals
    """
    restriction = FFTRestriction()

    # Copy and zero boundaries for momentum residuals
    R_u_2d = fine.R_u.reshape(fine.shape_full).copy()
    R_v_2d = fine.R_v.reshape(fine.shape_full).copy()

    R_u_2d[0, :] = 0.0
    R_u_2d[-1, :] = 0.0
    R_u_2d[:, 0] = 0.0
    R_u_2d[:, -1] = 0.0
    R_v_2d[0, :] = 0.0
    R_v_2d[-1, :] = 0.0
    R_v_2d[:, 0] = 0.0
    R_v_2d[:, -1] = 0.0

    # Restrict
    I_r_u_2d = restriction.restrict_2d(R_u_2d, coarse.shape_full)
    I_r_v_2d = restriction.restrict_2d(R_v_2d, coarse.shape_full)

    # Zero coarse boundaries too (belt and suspenders)
    I_r_u_2d[0, :] = 0.0
    I_r_u_2d[-1, :] = 0.0
    I_r_u_2d[:, 0] = 0.0
    I_r_u_2d[:, -1] = 0.0
    I_r_v_2d[0, :] = 0.0
    I_r_v_2d[-1, :] = 0.0
    I_r_v_2d[:, 0] = 0.0
    I_r_v_2d[:, -1] = 0.0

    # Pressure residual (inner grid - no boundary issues)
    R_p_2d = fine.R_p.reshape(fine.shape_inner)
    I_r_p_2d = restriction.restrict_2d(R_p_2d, coarse.shape_inner)

    return I_r_u_2d.ravel(), I_r_v_2d.ravel(), I_r_p_2d.ravel()


def prolongate_correction(
    coarse: FASLevel,
    fine: FASLevel,
    e_u: np.ndarray,
    e_v: np.ndarray,
    e_p: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prolongate correction from coarse to fine using FFT.

    Paper Eq. 10-11: Chebyshev interpolation.

    IMPORTANT: Zero boundaries before and after (BCs enforced separately).

    Returns
    -------
    e_u_fine, e_v_fine, e_p_fine : np.ndarray
        Prolongated corrections
    """
    prolongation = FFTProlongation()

    # Zero coarse boundaries before prolongation
    e_u_2d = e_u.reshape(coarse.shape_full).copy()
    e_v_2d = e_v.reshape(coarse.shape_full).copy()

    e_u_2d[0, :] = 0.0
    e_u_2d[-1, :] = 0.0
    e_u_2d[:, 0] = 0.0
    e_u_2d[:, -1] = 0.0
    e_v_2d[0, :] = 0.0
    e_v_2d[-1, :] = 0.0
    e_v_2d[:, 0] = 0.0
    e_v_2d[:, -1] = 0.0

    # Prolongate
    e_u_fine_2d = prolongation.prolongate_2d(e_u_2d, fine.shape_full)
    e_v_fine_2d = prolongation.prolongate_2d(e_v_2d, fine.shape_full)

    # Zero fine boundaries after prolongation
    e_u_fine_2d[0, :] = 0.0
    e_u_fine_2d[-1, :] = 0.0
    e_u_fine_2d[:, 0] = 0.0
    e_u_fine_2d[:, -1] = 0.0
    e_v_fine_2d[0, :] = 0.0
    e_v_fine_2d[-1, :] = 0.0
    e_v_fine_2d[:, 0] = 0.0
    e_v_fine_2d[:, -1] = 0.0

    # Pressure (inner grid)
    e_p_2d = e_p.reshape(coarse.shape_inner)
    e_p_fine_2d = prolongation.prolongate_2d(e_p_2d, fine.shape_inner)

    return e_u_fine_2d.ravel(), e_v_fine_2d.ravel(), e_p_fine_2d.ravel()


# =============================================================================
# FAS V-Cycle
# =============================================================================


def fas_vcycle(
    levels: List[FASLevel],
    level_idx: int,
    Re: float,
    beta_squared: float,
    lid_velocity: float,
    CFL: float,
    corner_treatment,
    Lx: float,
    Ly: float,
    pre_smooth: int = 1,
    post_smooth: int = 0,
    level_smoothing: Optional[List[int]] = None,
) -> None:
    """Perform one FAS V-cycle starting at level_idx.

    Recursive implementation of FAS algorithm (paper Section 3.1).

    Parameters
    ----------
    levels : List[FASLevel]
        Grid hierarchy (index 0 = coarsest)
    level_idx : int
        Current level index
    pre_smooth : int
        Pre-smoothing RK4 steps (default 1). Used if level_smoothing is None.
    post_smooth : int
        Post-smoothing RK4 steps (default 0, paper says unnecessary)
    level_smoothing : List[int], optional
        Per-level smoothing steps. Index 0 = coarsest level.
        Paper VMG-123 means [3, 2, 1] (coarsest to finest).
        If None, uses uniform pre_smooth on all levels.
    """
    level = levels[level_idx]

    # Determine smoothing for this level
    if level_smoothing is not None:
        n_smooth = level_smoothing[level_idx]
    else:
        n_smooth = pre_smooth

    # 1. Pre-smoothing
    for _ in range(n_smooth):
        fas_rk4_step(level, Re, beta_squared, lid_velocity, CFL, corner_treatment, Lx, Ly)

    # 2. Coarse grid correction (if not coarsest)
    if level_idx > 0:
        coarse = levels[level_idx - 1]

        # 2a. Save current fine solution
        u_h_old = level.u.copy()
        v_h_old = level.v.copy()
        p_h_old = level.p.copy()

        # 2b. Compute fine grid residual
        compute_residuals(level, level.u, level.v, level.p, Re, beta_squared)

        # 2c. Restrict residual (FFT + truncation)
        I_r_u, I_r_v, I_r_p = restrict_residual(level, coarse)

        # 2d. Restrict solution (direct injection)
        restrict_solution(level, coarse)
        u_H_old = coarse.u.copy()
        v_H_old = coarse.v.copy()
        p_H_old = coarse.p.copy()

        # 2e. Compute coarse residual at restricted solution
        compute_residuals(coarse, coarse.u, coarse.v, coarse.p, Re, beta_squared)

        # 2f. Compute tau correction: tau = I(r_h) - r_H'
        tau_u = I_r_u - coarse.R_u
        tau_v = I_r_v - coarse.R_v
        tau_p = I_r_p - coarse.R_p

        # Zero tau at boundaries
        tau_u_2d = tau_u.reshape(coarse.shape_full)
        tau_v_2d = tau_v.reshape(coarse.shape_full)
        tau_u_2d[0, :] = 0.0
        tau_u_2d[-1, :] = 0.0
        tau_u_2d[:, 0] = 0.0
        tau_u_2d[:, -1] = 0.0
        tau_v_2d[0, :] = 0.0
        tau_v_2d[-1, :] = 0.0
        tau_v_2d[:, 0] = 0.0
        tau_v_2d[:, -1] = 0.0

        # 2g. Set tau and recurse
        coarse.tau_u = tau_u
        coarse.tau_v = tau_v
        coarse.tau_p = tau_p

        fas_vcycle(
            levels, level_idx - 1,
            Re, beta_squared, lid_velocity, CFL, corner_treatment, Lx, Ly,
            pre_smooth, post_smooth, level_smoothing
        )

        # Clear tau
        coarse.tau_u = None
        coarse.tau_v = None
        coarse.tau_p = None

        # 2h. Compute correction: e_H = v_H - u_H
        e_u = coarse.u - u_H_old
        e_v = coarse.v - v_H_old
        e_p = coarse.p - p_H_old

        # 2i. Prolongate correction
        e_u_fine, e_v_fine, e_p_fine = prolongate_correction(coarse, level, e_u, e_v, e_p)

        # 2j. Apply correction (no damping, as per paper)
        level.u[:] = u_h_old + e_u_fine
        level.v[:] = v_h_old + e_v_fine
        level.p[:] = p_h_old + e_p_fine

        # 2k. Re-enforce boundary conditions
        enforce_boundary_conditions(level, level.u, level.v, lid_velocity, corner_treatment, Lx, Ly)

    # 3. Post-smoothing (paper says this is unnecessary and sometimes harmful)
    for _ in range(post_smooth):
        fas_rk4_step(level, Re, beta_squared, lid_velocity, CFL, corner_treatment, Lx, Ly)


# =============================================================================
# FASSolver Class
# =============================================================================


class FASSolver(SGSolver):
    """FAS V-Cycle Multigrid Solver following Zhang & Xi (2010).

    Achieves ~90% time reduction (10x speedup) with 3 levels using
    paper-faithful parameters:
    - pre_smooth = 1
    - post_smooth = 0
    - no damping on coarse correction
    - continuity RMS convergence criterion

    Parameters
    ----------
    All parameters from SGSolver, plus:
        n_levels : int
            Number of multigrid levels (2 or 3 recommended)
        pre_smooth : int
            Pre-smoothing RK4 steps per level (default 1)
        post_smooth : int
            Post-smoothing RK4 steps per level (default 0)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # FAS parameters
        self.n_levels = getattr(self.params, 'n_levels', 3)
        self.pre_smooth = getattr(self.params, 'pre_smooth', 1)
        self.post_smooth = getattr(self.params, 'post_smooth', 0)
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
            f"FASSolver initialized: {len(self._levels)} levels, "
            f"N = {[l.n for l in self._levels]}, "
            f"pre_smooth={self.pre_smooth}, post_smooth={self.post_smooth}"
        )

    def solve(self, tolerance: float = None, max_iter: int = None):
        """Solve using FAS V-cycle multigrid.

        Parameters
        ----------
        tolerance : float, optional
            Convergence tolerance (E_RMS). Default from params.
        max_iter : int, optional
            Maximum V-cycles. Default from params.
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

        log.info(f"FAS solve: tol={tolerance:.2e}, max_cycles={max_iter}")

        time_start = time.time()
        converged = False

        for cycle in range(max_iter):
            # Perform one V-cycle
            fas_vcycle(
                self._levels,
                level_idx=len(self._levels) - 1,
                Re=self.params.Re,
                beta_squared=self.params.beta_squared,
                lid_velocity=self.params.lid_velocity,
                CFL=self.params.CFL,
                corner_treatment=self.corner_treatment,
                Lx=self.params.Lx,
                Ly=self.params.Ly,
                pre_smooth=self.pre_smooth,
                post_smooth=self.post_smooth,
            )

            # Check convergence
            erms = compute_continuity_rms(finest)

            log.debug(f"V-cycle {cycle + 1}: E_RMS = {erms:.6e}")

            if erms < tolerance:
                converged = True
                log.info(f"FAS converged in {cycle + 1} V-cycles, E_RMS = {erms:.6e}")
                break

        time_end = time.time()
        wall_time = time_end - time_start

        if not converged:
            log.warning(f"FAS did not converge after {max_iter} V-cycles, E_RMS = {erms:.6e}")

        # Copy solution to output arrays
        self.arrays.u[:] = finest.u
        self.arrays.v[:] = finest.v
        # Pressure needs interpolation to full grid for output
        p_inner_2d = finest.p.reshape(finest.shape_inner)
        p_full_2d = finest.Interp_x @ p_inner_2d @ finest.Interp_y.T
        self.arrays.p[:] = p_full_2d.ravel()[:len(self.arrays.p)]  # Match size

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
            final_iter_count=cycle + 1,
            is_converged=converged,
            wall_time=wall_time,
            energy_history=[self._compute_energy()],
            enstrophy_history=[self._compute_enstrophy()],
            palinstrophy_history=[self._compute_palinstrophy()],
        )

        log.info(f"FAS completed in {wall_time:.2f}s")
