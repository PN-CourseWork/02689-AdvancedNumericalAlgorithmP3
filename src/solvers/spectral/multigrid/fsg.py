"""Spectral multigrid implementation for lid-driven cavity solver.

Based on Zhang & Xi (2010): "An explicit Chebyshev pseudospectral multigrid
method for incompressible Navier-Stokes equations"

Implements:
- FSG (Full Single Grid): Sequential solve from coarse to fine
- FMG (Full Multigrid): Coming in Phase 3

Transfer operators (prolongation/restriction) are pluggable via Hydra config.
Corner singularity treatment is also pluggable.
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

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
    corner_treatment: CornerTreatment,
    lid_velocity: float,
    Lx: float,
    Ly: float,
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
    corner_treatment : CornerTreatment
        Corner singularity treatment handler for boundary conditions
    lid_velocity : float
        Lid velocity magnitude
    Lx, Ly : float
        Domain dimensions
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

    # West boundary
    u_wall, v_wall = corner_treatment.get_wall_velocity(
        level_fine.X[0, :], level_fine.Y[0, :], Lx, Ly
    )
    u_fine_2d[0, :] = u_wall
    v_fine_2d[0, :] = v_wall

    # East boundary
    u_wall, v_wall = corner_treatment.get_wall_velocity(
        level_fine.X[-1, :], level_fine.Y[-1, :], Lx, Ly
    )
    u_fine_2d[-1, :] = u_wall
    v_fine_2d[-1, :] = v_wall

    # South boundary
    u_wall, v_wall = corner_treatment.get_wall_velocity(
        level_fine.X[:, 0], level_fine.Y[:, 0], Lx, Ly
    )
    u_fine_2d[:, 0] = u_wall
    v_fine_2d[:, 0] = v_wall

    # North boundary (moving lid)
    x_lid = level_fine.X[:, -1]
    y_lid = level_fine.Y[:, -1]
    u_lid, v_lid = corner_treatment.get_lid_velocity(
        x_lid, y_lid, lid_velocity, Lx, Ly
    )
    u_fine_2d[:, -1] = u_lid
    v_fine_2d[:, -1] = v_lid

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

        # Use subtraction method on all levels with corner exclusion for stability
        self.corner_treatment = corner_treatment
        self._uses_modified_convection = corner_treatment.uses_modified_convection()

        if self._uses_modified_convection:
            # Cache singular velocity and derivatives at this level's grid points
            X_flat = level.X.ravel()
            Y_flat = level.Y.ravel()

            u_s, v_s = corner_treatment.get_singular_velocity(X_flat, Y_flat, Lx, Ly)
            self._u_s = u_s.ravel()
            self._v_s = v_s.ravel()

            dus_dx, dus_dy, dvs_dx, dvs_dy = (
                corner_treatment.get_singular_velocity_derivatives(
                    X_flat, Y_flat, Lx, Ly
                )
            )

            # Create corner exclusion mask: don't apply modified convection very close to corners
            # The singular derivatives go like r^(λ-2) ≈ r^(-0.45) which blows up at corners
            # Using a cutoff radius based on grid spacing ensures numerical stability
            corner_radius = 2.5 * level.dx_min  # Exclude points within ~2.5 grid spacings

            # Distance from each corner
            r_left = np.sqrt(X_flat**2 + (Y_flat - Ly)**2)  # Distance from (0, Ly)
            r_right = np.sqrt((X_flat - Lx)**2 + (Y_flat - Ly)**2)  # Distance from (Lx, Ly)

            # Mask: 1.0 where we apply full terms, 0.0 near corners
            corner_mask = np.ones_like(X_flat)
            corner_mask = np.where(r_left < corner_radius, 0.0, corner_mask)
            corner_mask = np.where(r_right < corner_radius, 0.0, corner_mask)
            self._corner_mask = corner_mask.ravel()

            # Apply mask to singular velocities and derivatives
            self._u_s = self._u_s * self._corner_mask
            self._v_s = self._v_s * self._corner_mask
            self._dus_dx = dus_dx.ravel() * self._corner_mask
            self._dus_dy = dus_dy.ravel() * self._corner_mask
            self._dvs_dx = dvs_dx.ravel() * self._corner_mask
            self._dvs_dy = dvs_dy.ravel() * self._corner_mask

            # Precompute constant term: u_s·∇u_s (singular self-advection)
            self._conv_us_us = self._u_s * self._dus_dx + self._v_s * self._dus_dy
            self._conv_vs_vs = self._u_s * self._dvs_dx + self._v_s * self._dvs_dy

    def _apply_lid_boundary(self, u_2d: np.ndarray, v_2d: np.ndarray):
        """Apply lid boundary condition using corner treatment."""
        x_lid = self.level.X[:, -1]
        y_lid = self.level.Y[:, -1]

        u_lid, v_lid = self.corner_treatment.get_lid_velocity(
            x_lid,
            y_lid,
            lid_velocity=self.lid_velocity,
            Lx=self.Lx,
            Ly=self.Ly,
        )

        u_2d[:, -1] = u_lid
        v_2d[:, -1] = v_lid

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

    def _interpolate_pressure_gradient(self):
        """Compute pressure gradient on full grid from inner-grid pressure.

        Uses spectral interpolation (Chebyshev polynomial fit) to extend
        pressure from inner grid to full grid before differentiation.
        Uses O(N³) tensor product operations instead of O(N⁴) Kronecker products.
        """
        lvl = self.level

        # Step 1: Interpolate pressure from inner grid to full grid using spectral interpolation
        # 2D interpolation via tensor product: p_full = Interp_x @ p_inner @ Interp_y.T
        p_inner_2d = lvl.p.reshape(lvl.shape_inner)
        p_full_2d = lvl.Interp_x @ p_inner_2d @ lvl.Interp_y.T

        # Step 2: Compute pressure gradient using tensor products (O(N³) instead of O(N⁴))
        # d/dx: apply Dx_1d along axis 0 (rows)
        # d/dy: apply Dy_1d along axis 1 (columns via transpose trick)
        lvl.dp_dx[:] = (lvl.Dx_1d @ p_full_2d).ravel()
        lvl.dp_dy[:] = (p_full_2d @ lvl.Dy_1d.T).ravel()

    def _compute_residuals(self, u: np.ndarray, v: np.ndarray, p: np.ndarray):
        """Compute RHS residuals for RK4 pseudo time-stepping.

        Uses O(N³) tensor product operations instead of O(N⁴) Kronecker products.
        """
        lvl = self.level

        # Reshape to 2D for tensor product operations
        u_2d = u.reshape(lvl.shape_full)
        v_2d = v.reshape(lvl.shape_full)

        # Compute velocity derivatives using tensor products (O(N³) instead of O(N⁴))
        # d/dx: apply Dx_1d along axis 0
        # d/dy: apply Dy_1d along axis 1 (via transpose trick)
        du_dx_2d = lvl.Dx_1d @ u_2d
        du_dy_2d = u_2d @ lvl.Dy_1d.T
        dv_dx_2d = lvl.Dx_1d @ v_2d
        dv_dy_2d = v_2d @ lvl.Dy_1d.T

        # Store flattened derivatives
        lvl.du_dx[:] = du_dx_2d.ravel()
        lvl.du_dy[:] = du_dy_2d.ravel()
        lvl.dv_dx[:] = dv_dx_2d.ravel()
        lvl.dv_dy[:] = dv_dy_2d.ravel()

        # Compute Laplacians using tensor products: ∇²u = d²u/dx² + d²u/dy²
        lap_u_2d = lvl.Dxx_1d @ u_2d + u_2d @ lvl.Dyy_1d.T
        lap_v_2d = lvl.Dxx_1d @ v_2d + v_2d @ lvl.Dyy_1d.T
        lvl.lap_u[:] = lap_u_2d.ravel()
        lvl.lap_v[:] = lap_v_2d.ravel()

        # Pressure gradient (needs p array set first)
        old_p = lvl.p.copy()
        lvl.p[:] = p
        self._interpolate_pressure_gradient()
        lvl.p[:] = old_p

        # Momentum residuals - convection terms
        # Term 1: u_c·∇u_c (computational velocity advecting computational gradient)
        conv_u = u * lvl.du_dx + v * lvl.du_dy
        conv_v = u * lvl.dv_dx + v * lvl.dv_dy

        if self._uses_modified_convection:
            # Additional terms for subtraction method (Botella & Peyret 1998)
            # Term 2: u_s·∇u_c (singular velocity advecting computational gradient)
            conv_u = conv_u + self._u_s * lvl.du_dx + self._v_s * lvl.du_dy
            conv_v = conv_v + self._u_s * lvl.dv_dx + self._v_s * lvl.dv_dy

            # Term 3: u_c·∇u_s (computational velocity advecting singular gradient)
            conv_u = conv_u + u * self._dus_dx + v * self._dus_dy
            conv_v = conv_v + u * self._dvs_dx + v * self._dvs_dy

            # Term 4: u_s·∇u_s (precomputed constant)
            conv_u = conv_u + self._conv_us_us
            conv_v = conv_v + self._conv_vs_vs

        nu = 1.0 / self.Re

        lvl.R_u[:] = -conv_u - lvl.dp_dx + nu * lvl.lap_u
        lvl.R_v[:] = -conv_v - lvl.dp_dy + nu * lvl.lap_v

        # Continuity residual (on inner grid)
        divergence_full = lvl.du_dx + lvl.dv_dy
        divergence_2d = divergence_full.reshape(lvl.shape_full)
        divergence_inner = divergence_2d[1:-1, 1:-1].ravel()
        lvl.R_p[:] = -self.beta_squared * divergence_inner

        # Add FAS tau correction if set (for coarse grid solves in V-cycle)
        if self.tau_u is not None:
            lvl.R_u[:] += self.tau_u
        if self.tau_v is not None:
            lvl.R_v[:] += self.tau_v
        if self.tau_p is not None:
            lvl.R_p[:] += self.tau_p

    def _enforce_boundary_conditions(self, u: np.ndarray, v: np.ndarray):
        """Enforce boundary conditions using corner treatment."""
        u_2d = u.reshape(self.level.shape_full)
        v_2d = v.reshape(self.level.shape_full)

        # Get wall velocities from corner treatment (0 for smoothing, -u_s for subtraction)
        # West boundary
        u_wall, v_wall = self.corner_treatment.get_wall_velocity(
            self.level.X[0, :], self.level.Y[0, :], self.Lx, self.Ly
        )
        u_2d[0, :] = u_wall
        v_2d[0, :] = v_wall

        # East boundary
        u_wall, v_wall = self.corner_treatment.get_wall_velocity(
            self.level.X[-1, :], self.level.Y[-1, :], self.Lx, self.Ly
        )
        u_2d[-1, :] = u_wall
        v_2d[-1, :] = v_wall

        # South boundary
        u_wall, v_wall = self.corner_treatment.get_wall_velocity(
            self.level.X[:, 0], self.level.Y[:, 0], self.Lx, self.Ly
        )
        u_2d[:, 0] = u_wall
        v_2d[:, 0] = v_wall

        # North boundary (moving lid)
        self._apply_lid_boundary(u_2d, v_2d)

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

    For subtraction corner treatment: Uses smoothing on coarse levels (N<8) for
    stability, then transitions to full subtraction on finer levels. This hybrid
    approach avoids overflow from extreme singular derivatives on coarse grids.

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
        Corner singularity treatment handler. If None, uses default smoothing.
    Lx, Ly : float
        Domain dimensions

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

    # For subtraction method, also create smoothing treatment for coarse levels
    # This avoids numerical instability from extreme singular derivatives on coarse grids
    uses_subtraction = corner_treatment.uses_modified_convection()
    if uses_subtraction:
        smoothing_treatment = create_corner_treatment(method="smoothing")
        # Minimum N for subtraction method (below this, use smoothing)
        min_n_for_subtraction = 8

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

        # Select corner treatment for this level
        # For subtraction: use smoothing on coarse levels for stability
        if uses_subtraction and level.n < min_n_for_subtraction:
            level_corner_treatment = smoothing_treatment
            log.debug(f"  Level {level_idx} (N={level.n}): using smoothing (N < {min_n_for_subtraction})")
        else:
            level_corner_treatment = corner_treatment

        # Initialize from previous level or zeros
        if level_idx == 0:
            # Coarsest level: start from zeros
            level.u[:] = 0.0
            level.v[:] = 0.0
            level.p[:] = 0.0
        else:
            # Prolongate from previous (coarser) level
            prolongate_solution(
                levels[level_idx - 1],
                level,
                transfer_ops,
                level_corner_treatment,
                lid_velocity,
                Lx,
                Ly,
            )

        # Create smoother for this level
        smoother = MultigridSmoother(
            level=level,
            Re=Re,
            beta_squared=beta_squared,
            lid_velocity=lid_velocity,
            CFL=CFL,
            corner_treatment=level_corner_treatment,
            Lx=Lx,
            Ly=Ly,
        )
        smoother.initialize_lid()

        # Solve on this level
        converged = False
        level_iters = 0

        diverged = False
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

            # Early exit on NaN/Inf (diverged)
            if not np.isfinite(max_res):
                diverged = True
                log.warning(
                    f"  Level {level_idx} diverged (NaN/Inf) at iteration {level_iters}, exiting early"
                )
                break

            # Logging every 100 iterations
            if iteration > 0 and iteration % 100 == 0:
                cont_res = smoother.get_continuity_residual()
                log.debug(
                    f"  Level {level_idx} iter {iteration}: "
                    f"u_res={u_res:.2e}, v_res={v_res:.2e}, cont={cont_res:.2e}"
                )

        # Exit early if diverged (NaN/Inf detected)
        if diverged:
            break

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
    final_converged = converged and not diverged

    log.info(
        f"FSG completed: {total_iterations} total iterations, converged={final_converged}"
    )

    return finest_level, total_iterations, final_converged


