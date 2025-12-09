"""Spectral multigrid implementation for lid-driven cavity solver.

Based on Zhang & Xi (2010): "An explicit Chebyshev pseudospectral multigrid
method for incompressible Navier-Stokes equations"

Implements:
- FSG (Full Single Grid): Sequential solve from coarse to fine
- FMG (Full Multigrid): Coming in Phase 3

Transfer operators (prolongation/restriction) are pluggable via Hydra config.

Performance optimizations:
- Numba JIT-compiled kernels for derivatives and residuals
- Cached transposed matrices to avoid repeated transposition
- Pre-computed boundary conditions
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from numba import njit, prange

from solvers.spectral.operators.transfer_operators import (
    TransferOperators,
    create_transfer_operators,
    InjectionRestriction,
)
from solvers.spectral.operators.corner import (
    CornerTreatment,
    create_corner_treatment,
)

log = logging.getLogger(__name__)


# =============================================================================
# Numba JIT-compiled kernels for performance-critical operations
# =============================================================================


@njit(cache=True, fastmath=True)
def _compute_derivatives_and_laplacian(
    u_2d: np.ndarray,
    v_2d: np.ndarray,
    Dx_1d: np.ndarray,
    Dy_1d_T: np.ndarray,
    Dxx_1d: np.ndarray,
    Dyy_1d_T: np.ndarray,
    du_dx: np.ndarray,
    du_dy: np.ndarray,
    dv_dx: np.ndarray,
    dv_dy: np.ndarray,
    lap_u: np.ndarray,
    lap_v: np.ndarray,
):
    """JIT-compiled computation of all velocity derivatives and Laplacians.

    Computes:
    - du/dx, du/dy, dv/dx, dv/dy (first derivatives)
    - lap_u = d²u/dx² + d²u/dy², lap_v = d²v/dx² + d²v/dy² (Laplacians)

    All outputs are flattened 1D arrays.
    """
    n = u_2d.shape[0]

    # Compute all matrix products
    # du/dx = Dx @ u, du/dy = u @ Dy.T
    du_dx_2d = Dx_1d @ u_2d
    du_dy_2d = u_2d @ Dy_1d_T
    dv_dx_2d = Dx_1d @ v_2d
    dv_dy_2d = v_2d @ Dy_1d_T

    # Laplacians: lap_u = Dxx @ u + u @ Dyy.T
    lap_u_2d = Dxx_1d @ u_2d + u_2d @ Dyy_1d_T
    lap_v_2d = Dxx_1d @ v_2d + v_2d @ Dyy_1d_T

    # Flatten to output arrays
    idx = 0
    for i in range(n):
        for j in range(n):
            du_dx[idx] = du_dx_2d[i, j]
            du_dy[idx] = du_dy_2d[i, j]
            dv_dx[idx] = dv_dx_2d[i, j]
            dv_dy[idx] = dv_dy_2d[i, j]
            lap_u[idx] = lap_u_2d[i, j]
            lap_v[idx] = lap_v_2d[i, j]
            idx += 1

    return du_dx_2d, dv_dy_2d  # Return these for divergence computation


@njit(cache=True, fastmath=True)
def _compute_momentum_residuals(
    u: np.ndarray,
    v: np.ndarray,
    du_dx: np.ndarray,
    du_dy: np.ndarray,
    dv_dx: np.ndarray,
    dv_dy: np.ndarray,
    lap_u: np.ndarray,
    lap_v: np.ndarray,
    dp_dx: np.ndarray,
    dp_dy: np.ndarray,
    nu: float,
    R_u: np.ndarray,
    R_v: np.ndarray,
):
    """JIT-compiled momentum residual computation.

    Computes: R_u = -conv_u - dp/dx + nu*lap_u
              R_v = -conv_v - dp/dy + nu*lap_v
    where conv_u = u*du/dx + v*du/dy, conv_v = u*dv/dx + v*dv/dy
    """
    n = u.shape[0]
    for i in range(n):
        conv_u = u[i] * du_dx[i] + v[i] * du_dy[i]
        conv_v = u[i] * dv_dx[i] + v[i] * dv_dy[i]
        R_u[i] = -conv_u - dp_dx[i] + nu * lap_u[i]
        R_v[i] = -conv_v - dp_dy[i] + nu * lap_v[i]


@njit(cache=True, fastmath=True)
def _compute_continuity_residual(
    du_dx_2d: np.ndarray,
    dv_dy_2d: np.ndarray,
    beta_squared: float,
    R_p: np.ndarray,
):
    """JIT-compiled continuity residual on inner grid.

    Computes: R_p = -beta² * (du/dx + dv/dy) on interior points only
    """
    n = du_dx_2d.shape[0]
    idx = 0
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            R_p[idx] = -beta_squared * (du_dx_2d[i, j] + dv_dy_2d[i, j])
            idx += 1


@njit(cache=True, fastmath=True)
def _interpolate_and_differentiate_pressure(
    p_inner: np.ndarray,
    Interp_x: np.ndarray,
    Interp_y_T: np.ndarray,
    Dx_1d: np.ndarray,
    Dy_1d_T: np.ndarray,
    dp_dx: np.ndarray,
    dp_dy: np.ndarray,
    shape_inner: tuple,
):
    """JIT-compiled pressure interpolation and gradient computation.

    Combines:
    1. Interpolation from inner to full grid: p_full = Interp_x @ p_inner @ Interp_y.T
    2. Gradient computation: dp/dx = Dx @ p_full, dp/dy = p_full @ Dy.T
    """
    # Reshape to 2D
    ni, nj = shape_inner
    p_inner_2d = p_inner.reshape((ni, nj))

    # Interpolate: p_full = Interp_x @ p_inner @ Interp_y.T
    p_full_2d = Interp_x @ p_inner_2d @ Interp_y_T

    # Compute gradients
    dp_dx_2d = Dx_1d @ p_full_2d
    dp_dy_2d = p_full_2d @ Dy_1d_T

    # Flatten to output
    n = dp_dx_2d.shape[0]
    idx = 0
    for i in range(n):
        for j in range(n):
            dp_dx[idx] = dp_dx_2d[i, j]
            dp_dy[idx] = dp_dy_2d[i, j]
            idx += 1


def _build_interpolation_matrix_1d(nodes_inner, nodes_full):
    """Build interpolation matrix from inner grid to full grid.

    Uses Chebyshev polynomial interpolation for spectral accuracy.
    Given values f_inner at nodes_inner, computes f_full = Interp @ f_inner.

    Parameters
    ----------
    nodes_inner : np.ndarray
        Inner grid nodes (excludes boundary points)
    nodes_full : np.ndarray
        Full grid nodes (includes boundary points)

    Returns
    -------
    Interp : np.ndarray
        Interpolation matrix of shape (n_full, n_inner)
    """
    from numpy.polynomial.chebyshev import chebvander

    n_inner = len(nodes_inner)

    # Map physical domain to [-1, 1] for Chebyshev polynomials
    a, b = nodes_full[0], nodes_full[-1]
    xi_inner = 2 * (nodes_inner - a) / (b - a) - 1
    xi_full = 2 * (nodes_full - a) / (b - a) - 1

    # Vandermonde matrices: V[i,k] = T_k(xi[i])
    V_inner = chebvander(xi_inner, n_inner - 1)  # (n_inner, n_inner)
    V_full = chebvander(xi_full, n_inner - 1)    # (n_full, n_inner)

    # Interpolation: f_full = V_full @ coeffs, where coeffs = V_inner^{-1} @ f_inner
    # So: f_full = (V_full @ V_inner^{-1}) @ f_inner = Interp @ f_inner
    Interp = V_full @ np.linalg.solve(V_inner, np.eye(n_inner))

    return Interp


# =============================================================================
# SpectralLevel: Data structure for one multigrid level
# =============================================================================


@dataclass
class SpectralLevel:
    """Data structure holding all arrays and operators for one multigrid level.

    Attributes
    ----------
    n : int
        Polynomial order (gives n+1 nodes per dimension)
    level_idx : int
        Level index (0 = coarsest, increasing = finer)
    """

    # Grid info
    n: int  # polynomial order
    level_idx: int

    # 1D node arrays
    x_nodes: np.ndarray
    y_nodes: np.ndarray

    # 2D meshgrid arrays (full grid for velocities)
    X: np.ndarray
    Y: np.ndarray

    # 2D meshgrid arrays (inner grid for pressure)
    X_inner: np.ndarray
    Y_inner: np.ndarray

    # Grid shapes
    shape_full: Tuple[int, int]
    shape_inner: Tuple[int, int]

    # Minimum grid spacing for CFL
    dx_min: float
    dy_min: float

    # 1D differentiation matrices (for O(N³) tensor product operations)
    Dx_1d: np.ndarray  # 1D d/dx matrix (N+1) × (N+1)
    Dy_1d: np.ndarray  # 1D d/dy matrix (N+1) × (N+1)
    Dxx_1d: np.ndarray  # 1D d²/dx² matrix (N+1) × (N+1)
    Dyy_1d: np.ndarray  # 1D d²/dy² matrix (N+1) × (N+1)

    # 2D Kronecker form (kept for compatibility, but prefer 1D for performance)
    Dx: np.ndarray  # d/dx on full grid
    Dy: np.ndarray  # d/dy on full grid
    Dxx: np.ndarray  # d²/dx² on full grid
    Dyy: np.ndarray  # d²/dy² on full grid
    Laplacian: np.ndarray  # ∇² on full grid
    Dx_inner: np.ndarray  # d/dx on inner grid (deprecated, kept for compatibility)
    Dy_inner: np.ndarray  # d/dy on inner grid (deprecated, kept for compatibility)

    # Interpolation matrices from inner to full grid (for pressure gradient)
    Interp_x: np.ndarray  # 1D interpolation in x direction
    Interp_y: np.ndarray  # 1D interpolation in y direction

    # Solution arrays (flattened)
    u: np.ndarray  # velocity u on full grid
    v: np.ndarray  # velocity v on full grid
    p: np.ndarray  # pressure on inner grid

    # Previous iteration (for convergence)
    u_prev: np.ndarray
    v_prev: np.ndarray

    # RK4 stage buffers
    u_stage: np.ndarray
    v_stage: np.ndarray
    p_stage: np.ndarray

    # Residual arrays
    R_u: np.ndarray
    R_v: np.ndarray
    R_p: np.ndarray

    # Work buffers for derivatives
    du_dx: np.ndarray
    du_dy: np.ndarray
    dv_dx: np.ndarray
    dv_dy: np.ndarray
    lap_u: np.ndarray
    lap_v: np.ndarray
    dp_dx: np.ndarray
    dp_dy: np.ndarray
    dp_dx_inner: np.ndarray
    dp_dy_inner: np.ndarray

    @property
    def n_nodes_full(self) -> int:
        return self.shape_full[0] * self.shape_full[1]

    @property
    def n_nodes_inner(self) -> int:
        return self.shape_inner[0] * self.shape_inner[1]


def build_spectral_level(
    n: int,
    level_idx: int,
    basis_x,
    basis_y,
    Lx: float = 1.0,
    Ly: float = 1.0,
) -> SpectralLevel:
    """Construct a SpectralLevel with all operators and arrays.

    Parameters
    ----------
    n : int
        Polynomial order (n+1 nodes per dimension)
    level_idx : int
        Level index in hierarchy
    basis_x, basis_y : Basis objects
        Spectral basis (Chebyshev or Legendre Lobatto)
    Lx, Ly : float
        Domain dimensions

    Returns
    -------
    SpectralLevel
        Fully initialized level
    """
    # Grid shapes
    shape_full = (n + 1, n + 1)
    shape_inner = (n - 1, n - 1)
    n_full = shape_full[0] * shape_full[1]
    n_inner = shape_inner[0] * shape_inner[1]

    # 1D nodes
    x_nodes = basis_x.nodes(n + 1)
    y_nodes = basis_y.nodes(n + 1)

    # 2D meshgrids
    X, Y = np.meshgrid(x_nodes, y_nodes, indexing="ij")
    x_inner = x_nodes[1:-1]
    y_inner = y_nodes[1:-1]
    X_inner, Y_inner = np.meshgrid(x_inner, y_inner, indexing="ij")

    # Grid spacing
    dx_min = np.min(np.diff(x_nodes))
    dy_min = np.min(np.diff(y_nodes))

    # Build 1D differentiation matrices (stored for O(N³) tensor product operations)
    Dx_1d = basis_x.diff_matrix(x_nodes)
    Dy_1d = basis_y.diff_matrix(y_nodes)
    Dxx_1d = Dx_1d @ Dx_1d
    Dyy_1d = Dy_1d @ Dy_1d

    # Build 2D Kronecker matrices (kept for compatibility)
    Ix = np.eye(n + 1)
    Iy = np.eye(n + 1)
    Dx = np.kron(Dx_1d, Iy)
    Dy = np.kron(Ix, Dy_1d)
    Dxx = np.kron(Dxx_1d, Iy)
    Dyy = np.kron(Ix, Dyy_1d)
    Laplacian = Dxx + Dyy

    # Inner grid diff matrices (deprecated, but kept for compatibility)
    Dx_inner_1d = basis_x.diff_matrix(x_inner)
    Dy_inner_1d = basis_y.diff_matrix(y_inner)
    Ix_inner = np.eye(n - 1)
    Iy_inner = np.eye(n - 1)
    Dx_inner = np.kron(Dx_inner_1d, Iy_inner)
    Dy_inner = np.kron(Ix_inner, Dy_inner_1d)

    # Build interpolation matrices from inner to full grid (for pressure gradient)
    Interp_x = _build_interpolation_matrix_1d(x_inner, x_nodes)
    Interp_y = _build_interpolation_matrix_1d(y_inner, y_nodes)

    # Allocate solution and work arrays
    return SpectralLevel(
        n=n,
        level_idx=level_idx,
        x_nodes=x_nodes,
        y_nodes=y_nodes,
        X=X,
        Y=Y,
        X_inner=X_inner,
        Y_inner=Y_inner,
        shape_full=shape_full,
        shape_inner=shape_inner,
        dx_min=dx_min,
        dy_min=dy_min,
        # 1D matrices for tensor product operations
        Dx_1d=Dx_1d,
        Dy_1d=Dy_1d,
        Dxx_1d=Dxx_1d,
        Dyy_1d=Dyy_1d,
        # 2D Kronecker matrices (kept for compatibility)
        Dx=Dx,
        Dy=Dy,
        Dxx=Dxx,
        Dyy=Dyy,
        Laplacian=Laplacian,
        Dx_inner=Dx_inner,
        Dy_inner=Dy_inner,
        Interp_x=Interp_x,
        Interp_y=Interp_y,
        # Solution arrays
        u=np.zeros(n_full),
        v=np.zeros(n_full),
        p=np.zeros(n_inner),
        u_prev=np.zeros(n_full),
        v_prev=np.zeros(n_full),
        u_stage=np.zeros(n_full),
        v_stage=np.zeros(n_full),
        p_stage=np.zeros(n_inner),
        R_u=np.zeros(n_full),
        R_v=np.zeros(n_full),
        R_p=np.zeros(n_inner),
        du_dx=np.zeros(n_full),
        du_dy=np.zeros(n_full),
        dv_dx=np.zeros(n_full),
        dv_dy=np.zeros(n_full),
        lap_u=np.zeros(n_full),
        lap_v=np.zeros(n_full),
        dp_dx=np.zeros(n_full),
        dp_dy=np.zeros(n_full),
        dp_dx_inner=np.zeros(n_inner),
        dp_dy_inner=np.zeros(n_inner),
    )


# =============================================================================
# Grid Hierarchy
# =============================================================================


def build_hierarchy(
    n_fine: int,
    n_levels: int,
    basis_x,
    basis_y,
    Lx: float = 1.0,
    Ly: float = 1.0,
    coarsest_n: int = 12,
) -> List[SpectralLevel]:
    """Build multigrid hierarchy from fine to coarse.

    Parameters
    ----------
    n_fine : int
        Polynomial order on finest grid
    n_levels : int
        Maximum number of multigrid levels (may use fewer if coarsest_n limit reached)
    basis_x, basis_y : Basis objects
        Spectral basis objects
    Lx, Ly : float
        Domain dimensions
    coarsest_n : int
        Minimum polynomial order for coarsest grid (default 12).
        Coarse grids need sufficient resolution to capture physics.

    Returns
    -------
    List[SpectralLevel]
        List of levels, index 0 = coarsest, index -1 = finest
    """
    # Compute polynomial orders for each level (full coarsening: N/2)
    # Stop when next coarsening would go below coarsest_n
    orders = []
    n = n_fine
    for _ in range(n_levels):
        orders.append(n)
        n_next = n // 2
        if n_next < coarsest_n:
            break
        n = n_next

    # Reverse so coarsest is first
    orders = orders[::-1]

    log.info(f"Building {len(orders)}-level hierarchy: N = {orders}")

    # Verify coarse nodes are subset of fine nodes (for Lobatto grids)
    # This is automatic for N_c = N_f / 2 with Lobatto nodes

    levels = []
    for idx, n in enumerate(orders):
        level = build_spectral_level(n, idx, basis_x, basis_y, Lx, Ly)
        levels.append(level)

    return levels


# =============================================================================
# Prolongation (Coarse to Fine Interpolation)
# =============================================================================


def prolongate_solution(
    level_coarse: SpectralLevel,
    level_fine: SpectralLevel,
    transfer_ops: TransferOperators,
    lid_velocity: float = 1.0,
) -> None:
    """Prolongate solution (u, v, p) from coarse level to fine level.

    Modifies level_fine.u, level_fine.v, level_fine.p in place.

    IMPORTANT: After spectral interpolation, boundary conditions are
    re-enforced explicitly to avoid Gibbs-type oscillations at boundaries.

    Parameters
    ----------
    level_coarse : SpectralLevel
        Source (coarse) level with converged solution
    level_fine : SpectralLevel
        Target (fine) level to receive interpolated solution
    transfer_ops : TransferOperators
        Configured transfer operators for prolongation
    lid_velocity : float
        Lid velocity for boundary condition (default: 1.0)
    """
    # Prolongate velocities (full grid)
    u_coarse_2d = level_coarse.u.reshape(level_coarse.shape_full)
    v_coarse_2d = level_coarse.v.reshape(level_coarse.shape_full)

    u_fine_2d = transfer_ops.prolongation.prolongate_2d(
        u_coarse_2d, level_fine.shape_full
    )
    v_fine_2d = transfer_ops.prolongation.prolongate_2d(
        v_coarse_2d, level_fine.shape_full
    )

    # Re-enforce boundary conditions after interpolation
    # (spectral interpolation can introduce Gibbs oscillations at boundaries)
    # Bottom: u=0, v=0
    u_fine_2d[0, :] = 0.0
    v_fine_2d[0, :] = 0.0
    # Top: u=lid_velocity, v=0
    u_fine_2d[-1, :] = lid_velocity
    v_fine_2d[-1, :] = 0.0
    # Left: u=0, v=0
    u_fine_2d[:, 0] = 0.0
    v_fine_2d[:, 0] = 0.0
    # Right: u=0, v=0
    u_fine_2d[:, -1] = 0.0
    v_fine_2d[:, -1] = 0.0

    level_fine.u[:] = u_fine_2d.ravel()
    level_fine.v[:] = v_fine_2d.ravel()

    # Prolongate pressure (inner grid - no boundary conditions needed)
    p_coarse_2d = level_coarse.p.reshape(level_coarse.shape_inner)
    p_fine_2d = transfer_ops.prolongation.prolongate_2d(
        p_coarse_2d, level_fine.shape_inner
    )
    level_fine.p[:] = p_fine_2d.ravel()

    log.debug(
        f"Prolongated solution from level {level_coarse.level_idx} "
        f"(N={level_coarse.n}) to level {level_fine.level_idx} (N={level_fine.n})"
    )


# =============================================================================
# Restriction (Fine to Coarse)
# =============================================================================


def restrict_solution(
    level_fine: SpectralLevel,
    level_coarse: SpectralLevel,
    transfer_ops: TransferOperators,
) -> None:
    """Restrict solution (u, v, p) from fine level to coarse level.

    Uses direct injection for variables (FAS scheme requirement).
    This is critical: coarse GLL points are subsets of fine GLL points,
    so injection preserves the exact solution values.

    Parameters
    ----------
    level_fine : SpectralLevel
        Source (fine) level
    level_coarse : SpectralLevel
        Target (coarse) level
    transfer_ops : TransferOperators
        Configured transfer operators (not used - always uses injection)
    """
    # FAS requires direct injection for solution restriction
    # (coarse GLL points are subsets of fine GLL points)
    injection = InjectionRestriction()

    # Restrict velocities (full grid)
    u_fine_2d = level_fine.u.reshape(level_fine.shape_full)
    v_fine_2d = level_fine.v.reshape(level_fine.shape_full)

    u_coarse_2d = injection.restrict_2d(u_fine_2d, level_coarse.shape_full)
    v_coarse_2d = injection.restrict_2d(v_fine_2d, level_coarse.shape_full)

    level_coarse.u[:] = u_coarse_2d.ravel()
    level_coarse.v[:] = v_coarse_2d.ravel()

    # Restrict pressure (inner grid)
    p_fine_2d = level_fine.p.reshape(level_fine.shape_inner)
    p_coarse_2d = injection.restrict_2d(p_fine_2d, level_coarse.shape_inner)
    level_coarse.p[:] = p_coarse_2d.ravel()

    log.debug(
        f"Restricted solution from level {level_fine.level_idx} "
        f"(N={level_fine.n}) to level {level_coarse.level_idx} (N={level_coarse.n})"
    )


def restrict_residual(
    level_fine: SpectralLevel,
    level_coarse: SpectralLevel,
    transfer_ops: TransferOperators,
) -> None:
    """Restrict residuals (R_u, R_v, R_p) from fine to coarse level.

    Uses FFT-based restriction for residuals (spectral truncation).

    Per Zhang & Xi (2010), Section 3.3:
    "In the PN − PN−2 method, the boundary values are already known for
    velocities and unnecessary for pressure, so the residuals and corrections
    on the boundary points are all set to zero."

    Parameters
    ----------
    level_fine : SpectralLevel
        Source (fine) level with computed residuals
    level_coarse : SpectralLevel
        Target (coarse) level to receive restricted residuals
    transfer_ops : TransferOperators
        Configured transfer operators
    """
    # Restrict momentum residuals (full grid)
    # Per paper Section 3.3: Use FFT-based restriction with high-frequency truncation
    #
    # IMPORTANT: Zero FINE grid boundaries BEFORE restriction!
    # The residuals at boundary nodes are garbage (BCs are enforced separately).
    # If we don't zero them before FFT restriction, they pollute interior values
    # through spectral truncation.
    R_u_fine_2d = level_fine.R_u.reshape(level_fine.shape_full).copy()
    R_v_fine_2d = level_fine.R_v.reshape(level_fine.shape_full).copy()

    # Zero fine grid boundaries BEFORE restriction
    R_u_fine_2d[0, :] = 0.0
    R_u_fine_2d[-1, :] = 0.0
    R_u_fine_2d[:, 0] = 0.0
    R_u_fine_2d[:, -1] = 0.0
    R_v_fine_2d[0, :] = 0.0
    R_v_fine_2d[-1, :] = 0.0
    R_v_fine_2d[:, 0] = 0.0
    R_v_fine_2d[:, -1] = 0.0

    R_u_coarse_2d = transfer_ops.restriction.restrict_2d(
        R_u_fine_2d, level_coarse.shape_full
    )
    R_v_coarse_2d = transfer_ops.restriction.restrict_2d(
        R_v_fine_2d, level_coarse.shape_full
    )

    # Also zero coarse grid boundaries after restriction (belt and suspenders)
    # "residuals and corrections on the boundary points are all set to zero"
    R_u_coarse_2d[0, :] = 0.0
    R_u_coarse_2d[-1, :] = 0.0
    R_u_coarse_2d[:, 0] = 0.0
    R_u_coarse_2d[:, -1] = 0.0
    R_v_coarse_2d[0, :] = 0.0
    R_v_coarse_2d[-1, :] = 0.0
    R_v_coarse_2d[:, 0] = 0.0
    R_v_coarse_2d[:, -1] = 0.0

    level_coarse.R_u[:] = R_u_coarse_2d.ravel()
    level_coarse.R_v[:] = R_v_coarse_2d.ravel()

    # Restrict continuity residual (inner grid - already excludes boundaries)
    R_p_fine_2d = level_fine.R_p.reshape(level_fine.shape_inner)
    R_p_coarse_2d = transfer_ops.restriction.restrict_2d(
        R_p_fine_2d, level_coarse.shape_inner
    )
    level_coarse.R_p[:] = R_p_coarse_2d.ravel()


# =============================================================================
# Level-Specific Solver Routines
# =============================================================================


class MultigridSmoother:
    """Performs RK4 smoothing iterations on a single level.

    Encapsulates the time-stepping logic for one multigrid level.
    Supports FAS scheme by allowing external forcing terms (tau correction).
    """

    def __init__(
        self,
        level: SpectralLevel,
        Re: float,
        beta_squared: float,
        lid_velocity: float,
        CFL: float,
        corner_treatment: CornerTreatment,
        Lx: float = 1.0,
        Ly: float = 1.0,
    ):
        self.level = level
        self.Re = Re
        self.nu = 1.0 / Re  # Cache viscosity
        self.beta_squared = beta_squared
        self.lid_velocity = lid_velocity
        self.CFL = CFL
        self.Lx = Lx
        self.Ly = Ly

        # FAS forcing terms (tau correction from fine grid)
        # These are ADDED to the computed residuals during coarse grid solve
        self.tau_u = None
        self.tau_v = None
        self.tau_p = None

        self.corner_treatment = corner_treatment

        # ========== OPTIMIZATION: Cache boundary conditions ==========
        # Pre-compute and cache lid velocity (it never changes during solve)
        x_lid = level.X[:, -1]
        y_lid = level.Y[:, -1]
        self._cached_u_lid, self._cached_v_lid = corner_treatment.get_lid_velocity(
            x_lid, y_lid, lid_velocity=lid_velocity, Lx=Lx, Ly=Ly
        )

        # Pre-compute wall velocities (zeros)
        # West boundary
        self._cached_u_west, self._cached_v_west = corner_treatment.get_wall_velocity(
            level.X[0, :], level.Y[0, :], Lx, Ly
        )
        # East boundary
        self._cached_u_east, self._cached_v_east = corner_treatment.get_wall_velocity(
            level.X[-1, :], level.Y[-1, :], Lx, Ly
        )
        # South boundary
        self._cached_u_south, self._cached_v_south = corner_treatment.get_wall_velocity(
            level.X[:, 0], level.Y[:, 0], Lx, Ly
        )

        # ========== OPTIMIZATION: Cache transposed matrices ==========
        self._Dy_1d_T = level.Dy_1d.T.copy()
        self._Dyy_1d_T = level.Dyy_1d.T.copy()
        self._Interp_y_T = level.Interp_y.T.copy()

    def _apply_lid_boundary(self, u_2d: np.ndarray, v_2d: np.ndarray):
        """Apply lid boundary condition using cached values."""
        # OPTIMIZATION: Use pre-computed lid velocities instead of calling corner_treatment
        u_2d[:, -1] = self._cached_u_lid
        v_2d[:, -1] = self._cached_v_lid

    def _extrapolate_to_full_grid(self, inner_2d: np.ndarray) -> np.ndarray:
        """Extrapolate from inner grid to full grid."""
        full_2d = np.zeros(self.level.shape_full)
        full_2d[1:-1, 1:-1] = inner_2d

        # Linear extrapolation to boundaries
        full_2d[0, 1:-1] = 2 * full_2d[1, 1:-1] - full_2d[2, 1:-1]
        full_2d[-1, 1:-1] = 2 * full_2d[-2, 1:-1] - full_2d[-3, 1:-1]
        full_2d[1:-1, 0] = 2 * full_2d[1:-1, 1] - full_2d[1:-1, 2]
        full_2d[1:-1, -1] = 2 * full_2d[1:-1, -2] - full_2d[1:-1, -3]

        # Corners
        full_2d[0, 0] = 0.5 * (full_2d[0, 1] + full_2d[1, 0])
        full_2d[0, -1] = 0.5 * (full_2d[0, -2] + full_2d[1, -1])
        full_2d[-1, 0] = 0.5 * (full_2d[-1, 1] + full_2d[-2, 0])
        full_2d[-1, -1] = 0.5 * (full_2d[-1, -2] + full_2d[-2, -1])

        return full_2d

    def _interpolate_pressure_gradient(self, p: np.ndarray):
        """Compute pressure gradient on full grid from inner-grid pressure.

        Uses spectral interpolation (Chebyshev polynomial fit) to extend
        pressure from inner grid to full grid before differentiation.
        OPTIMIZED: Uses JIT-compiled kernel for combined interpolation and differentiation.

        Parameters
        ----------
        p : np.ndarray
            Pressure array on inner grid (passed directly to avoid copy)
        """
        lvl = self.level

        # JIT-compiled combined interpolation and gradient computation
        _interpolate_and_differentiate_pressure(
            p,
            lvl.Interp_x,
            self._Interp_y_T,
            lvl.Dx_1d,
            self._Dy_1d_T,
            lvl.dp_dx,
            lvl.dp_dy,
            lvl.shape_inner,
        )

    def _compute_residuals(self, u: np.ndarray, v: np.ndarray, p: np.ndarray):
        """Compute RHS residuals for RK4 pseudo time-stepping.

        Uses O(N³) tensor product operations instead of O(N⁴) Kronecker products.
        OPTIMIZED: Uses Numba JIT-compiled kernels for performance.
        """
        lvl = self.level

        # Reshape to 2D for tensor product operations (views, no copy)
        u_2d = u.reshape(lvl.shape_full)
        v_2d = v.reshape(lvl.shape_full)

        # JIT-compiled: Compute all derivatives and Laplacians in one call
        du_dx_2d, dv_dy_2d = _compute_derivatives_and_laplacian(
            u_2d, v_2d,
            lvl.Dx_1d, self._Dy_1d_T,
            lvl.Dxx_1d, self._Dyy_1d_T,
            lvl.du_dx, lvl.du_dy,
            lvl.dv_dx, lvl.dv_dy,
            lvl.lap_u, lvl.lap_v,
        )

        # Compute pressure gradient
        self._interpolate_pressure_gradient(p)

        # JIT-compiled: Compute momentum residuals
        _compute_momentum_residuals(
            u, v,
            lvl.du_dx, lvl.du_dy,
            lvl.dv_dx, lvl.dv_dy,
            lvl.lap_u, lvl.lap_v,
            lvl.dp_dx, lvl.dp_dy,
            self.nu,
            lvl.R_u, lvl.R_v,
        )

        # JIT-compiled: Continuity residual on inner grid
        _compute_continuity_residual(du_dx_2d, dv_dy_2d, self.beta_squared, lvl.R_p)

        # Add FAS tau correction if set (for coarse grid solves in V-cycle)
        if self.tau_u is not None:
            lvl.R_u += self.tau_u
        if self.tau_v is not None:
            lvl.R_v += self.tau_v
        if self.tau_p is not None:
            lvl.R_p += self.tau_p

    def _enforce_boundary_conditions(self, u: np.ndarray, v: np.ndarray):
        """Enforce boundary conditions using cached values."""
        u_2d = u.reshape(self.level.shape_full)
        v_2d = v.reshape(self.level.shape_full)

        # OPTIMIZATION: Use pre-computed wall velocities instead of calling corner_treatment
        # West boundary
        u_2d[0, :] = self._cached_u_west
        v_2d[0, :] = self._cached_v_west

        # East boundary
        u_2d[-1, :] = self._cached_u_east
        v_2d[-1, :] = self._cached_v_east

        # South boundary
        u_2d[:, 0] = self._cached_u_south
        v_2d[:, 0] = self._cached_v_south

        # North boundary (moving lid) - also uses cached values
        u_2d[:, -1] = self._cached_u_lid
        v_2d[:, -1] = self._cached_v_lid

    def _compute_adaptive_timestep(self) -> float:
        """Compute adaptive timestep based on CFL."""
        lvl = self.level
        u_max = max(np.max(np.abs(lvl.u)), self.lid_velocity)
        v_max = max(np.max(np.abs(lvl.v)), 1e-10)
        nu = 1.0 / self.Re

        lambda_x = (
            u_max + np.sqrt(u_max**2 + self.beta_squared)
        ) / lvl.dx_min + nu / lvl.dx_min**2
        lambda_y = (
            v_max + np.sqrt(v_max**2 + self.beta_squared)
        ) / lvl.dy_min + nu / lvl.dy_min**2

        return self.CFL / (lambda_x + lambda_y)

    def initialize_lid(self):
        """Initialize lid velocity boundary condition using corner treatment."""
        u_2d = self.level.u.reshape(self.level.shape_full)
        v_2d = self.level.v.reshape(self.level.shape_full)
        self._apply_lid_boundary(u_2d, v_2d)

    def step(self) -> Tuple[float, float]:
        """Perform one RK4 pseudo time-step.

        Returns
        -------
        tuple
            (u_residual, v_residual) - L2 norms of velocity change
        """
        lvl = self.level

        # Save previous for convergence check
        lvl.u_prev[:] = lvl.u
        lvl.v_prev[:] = lvl.v

        dt = self._compute_adaptive_timestep()

        # 4-stage RK4
        rk4_coeffs = [0.25, 1.0 / 3.0, 0.5, 1.0]
        u_in, v_in, p_in = lvl.u, lvl.v, lvl.p

        for i, alpha in enumerate(rk4_coeffs):
            self._compute_residuals(u_in, v_in, p_in)

            if i < 3:
                lvl.u_stage[:] = lvl.u + alpha * dt * lvl.R_u
                lvl.v_stage[:] = lvl.v + alpha * dt * lvl.R_v
                lvl.p_stage[:] = lvl.p + alpha * dt * lvl.R_p
                self._enforce_boundary_conditions(lvl.u_stage, lvl.v_stage)
                u_in, v_in, p_in = lvl.u_stage, lvl.v_stage, lvl.p_stage
            else:
                lvl.u[:] = lvl.u + alpha * dt * lvl.R_u
                lvl.v[:] = lvl.v + alpha * dt * lvl.R_v
                lvl.p[:] = lvl.p + alpha * dt * lvl.R_p
                self._enforce_boundary_conditions(lvl.u, lvl.v)

        # Compute RELATIVE residuals for convergence check
        u_res = np.linalg.norm(lvl.u - lvl.u_prev) / (np.linalg.norm(lvl.u_prev) + 1e-12)
        v_res = np.linalg.norm(lvl.v - lvl.v_prev) / (np.linalg.norm(lvl.v_prev) + 1e-12)

        return u_res, v_res

    def smooth(self, n_steps: int) -> Tuple[float, float]:
        """Perform multiple RK4 smoothing steps.

        Parameters
        ----------
        n_steps : int
            Number of RK4 steps

        Returns
        -------
        tuple
            Final (u_residual, v_residual)
        """
        u_res, v_res = 0.0, 0.0
        for _ in range(n_steps):
            u_res, v_res = self.step()
        return u_res, v_res

    def get_continuity_residual(self) -> float:
        """Get L2 norm of continuity residual."""
        return np.linalg.norm(self.level.R_p)

    def set_tau_correction(
        self,
        tau_u: np.ndarray,
        tau_v: np.ndarray,
        tau_p: np.ndarray,
    ):
        """Set FAS tau correction terms for coarse grid solve.

        These terms are added to the computed residuals during RK4 steps.
        Call clear_tau_correction() after the coarse grid solve is complete.

        Parameters
        ----------
        tau_u, tau_v : np.ndarray
            Momentum tau corrections (full grid size)
        tau_p : np.ndarray
            Pressure tau correction (inner grid size)
        """
        self.tau_u = tau_u
        self.tau_v = tau_v
        self.tau_p = tau_p

    def clear_tau_correction(self):
        """Clear FAS tau correction terms."""
        self.tau_u = None
        self.tau_v = None
        self.tau_p = None


# =============================================================================
# FSG Driver (Full Single Grid)
# =============================================================================


def solve_fsg(
    levels: List[SpectralLevel],
    Re: float,
    beta_squared: float,
    lid_velocity: float,
    CFL: float,
    tolerance: float,
    max_iterations: int,
    transfer_ops: Optional[TransferOperators] = None,
    corner_treatment: Optional[CornerTreatment] = None,
    Lx: float = 1.0,
    Ly: float = 1.0,
    coarse_tolerance_factor: float = 1.0,
) -> Tuple[SpectralLevel, int, bool]:
    """Solve using Full Single Grid (FSG) multigrid.

    Solves sequentially from coarsest to finest level, using the converged
    solution on each level as initial guess for the next finer level.

    Per Zhang & Xi (2010): Uses the SAME tolerance on ALL levels.

    Parameters
    ----------
    levels : List[SpectralLevel]
        Grid hierarchy (index 0 = coarsest)
    Re, beta_squared, lid_velocity, CFL : float
        Solver parameters
    tolerance : float
        Global convergence tolerance (used on ALL levels)
    max_iterations : int
        Max iterations per level
    transfer_ops : TransferOperators, optional
        Configured transfer operators. If None, uses default FFT operators.
    corner_treatment : CornerTreatment, optional
        Corner treatment handler. If None, uses default smoothing.
    Lx, Ly : float
        Domain dimensions
    coarse_tolerance_factor : float
        Factor to loosen tolerance on coarser levels

    Returns
    -------
    tuple
        (finest_level, total_iterations, converged)
    """
    # Create default transfer operators if not provided
    if transfer_ops is None:
        transfer_ops = create_transfer_operators(
            prolongation_method="fft",
            restriction_method="fft",
        )

    # Create default corner treatment if not provided
    if corner_treatment is None:
        corner_treatment = create_corner_treatment(method="smoothing")

    total_iterations = 0
    n_levels = len(levels)

    for level_idx, level in enumerate(levels):
        is_finest = level_idx == n_levels - 1

        # Use LOOSER tolerance on coarser levels for efficiency
        # coarse_tolerance_factor=10 means: coarsest gets 100x looser (for 3 levels)
        levels_from_finest = n_levels - 1 - level_idx
        level_tol = tolerance * (coarse_tolerance_factor ** levels_from_finest)

        log.info(
            f"FSG Level {level_idx}/{n_levels - 1}: N={level.n}, "
            f"tolerance={level_tol:.2e}"
        )

        # Initialize from previous level or zeros
        if level_idx == 0:
            # Coarsest level: start from zeros
            level.u[:] = 0.0
            level.v[:] = 0.0
            level.p[:] = 0.0
        else:
            # Prolongate from previous (coarser) level
            prolongate_solution(levels[level_idx - 1], level, transfer_ops, lid_velocity)

        # Create smoother for this level
        smoother = MultigridSmoother(
            level=level,
            Re=Re,
            beta_squared=beta_squared,
            lid_velocity=lid_velocity,
            CFL=CFL,
            corner_treatment=corner_treatment,
            Lx=Lx,
            Ly=Ly,
        )
        smoother.initialize_lid()

        # Solve on this level
        converged = False
        level_iters = 0

        for iteration in range(max_iterations):
            u_res, v_res = smoother.step()
            level_iters += 1
            total_iterations += 1

            # Check convergence
            max_res = max(u_res, v_res)
            if max_res < level_tol:
                converged = True
                cont_res = smoother.get_continuity_residual()
                log.info(
                    f"  Level {level_idx} converged in {level_iters} iterations, "
                    f"residual={max_res:.2e}, continuity={cont_res:.2e}"
                )
                break

            # Logging every 100 iterations
            if iteration > 0 and iteration % 100 == 0:
                cont_res = smoother.get_continuity_residual()
                log.debug(
                    f"  Level {level_idx} iter {iteration}: "
                    f"u_res={u_res:.2e}, v_res={v_res:.2e}, cont={cont_res:.2e}"
                )

        if not converged and not is_finest:
            log.warning(
                f"  Level {level_idx} did not converge after {level_iters} iterations, "
                f"continuing to next level..."
            )
        elif not converged and is_finest:
            log.warning(
                f"  Finest level did not converge after {level_iters} iterations"
            )

    finest_level = levels[-1]
    final_converged = converged

    log.info(
        f"FSG completed: {total_iterations} total iterations, converged={final_converged}"
    )

    return finest_level, total_iterations, final_converged


