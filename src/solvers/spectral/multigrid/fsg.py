"""Spectral multigrid implementation for lid-driven cavity solver.

Based on Zhang & Xi (2010): "An explicit Chebyshev pseudospectral multigrid
method for incompressible Navier-Stokes equations"

Implements:
- FSG (Full Single Grid): Sequential solve from coarse to fine
- VMG (V-cycle Multigrid with FAS): Coming in Phase 2
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
)
from solvers.spectral.operators.corner import (
    CornerTreatment,
    create_corner_treatment,
)

log = logging.getLogger(__name__)


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

    # Differentiation matrices (2D Kronecker form)
    Dx: np.ndarray  # d/dx on full grid
    Dy: np.ndarray  # d/dy on full grid
    Dxx: np.ndarray  # d²/dx² on full grid
    Dyy: np.ndarray  # d²/dy² on full grid
    Laplacian: np.ndarray  # ∇² on full grid
    Dx_inner: np.ndarray  # d/dx on inner grid
    Dy_inner: np.ndarray  # d/dy on inner grid

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

    # Build differentiation matrices
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

    # Inner grid diff matrices
    Dx_inner_1d = basis_x.diff_matrix(x_inner)
    Dy_inner_1d = basis_y.diff_matrix(y_inner)
    Ix_inner = np.eye(n - 1)
    Iy_inner = np.eye(n - 1)
    Dx_inner = np.kron(Dx_inner_1d, Iy_inner)
    Dy_inner = np.kron(Ix_inner, Dy_inner_1d)

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
        Dx=Dx,
        Dy=Dy,
        Dxx=Dxx,
        Dyy=Dyy,
        Laplacian=Laplacian,
        Dx_inner=Dx_inner,
        Dy_inner=Dy_inner,
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
) -> List[SpectralLevel]:
    """Build multigrid hierarchy from fine to coarse.

    Parameters
    ----------
    n_fine : int
        Polynomial order on finest grid
    n_levels : int
        Number of multigrid levels
    basis_x, basis_y : Basis objects
        Spectral basis objects

    Returns
    -------
    List[SpectralLevel]
        List of levels, index 0 = coarsest, index -1 = finest
    """
    # Compute polynomial orders for each level (full coarsening: N/2)
    orders = []
    n = n_fine
    for _ in range(n_levels):
        orders.append(n)
        n = n // 2
        if n < 3:  # Minimum usable grid
            break

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
) -> None:
    """Prolongate solution (u, v, p) from coarse level to fine level.

    Modifies level_fine.u, level_fine.v, level_fine.p in place.

    Parameters
    ----------
    level_coarse : SpectralLevel
        Source (coarse) level with converged solution
    level_fine : SpectralLevel
        Target (fine) level to receive interpolated solution
    transfer_ops : TransferOperators
        Configured transfer operators for prolongation
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

    level_fine.u[:] = u_fine_2d.ravel()
    level_fine.v[:] = v_fine_2d.ravel()

    # Prolongate pressure (inner grid)
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
# Level-Specific Solver Routines
# =============================================================================


class MultigridSmoother:
    """Performs RK4 smoothing iterations on a single level.

    Encapsulates the time-stepping logic for one multigrid level.
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
        self.corner_treatment = corner_treatment
        self.Lx = Lx
        self.Ly = Ly

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
        """Compute pressure gradient on inner grid and extrapolate to full."""
        lvl = self.level

        # Compute on inner grid
        lvl.dp_dx_inner[:] = lvl.Dx_inner @ lvl.p
        lvl.dp_dy_inner[:] = lvl.Dy_inner @ lvl.p

        # Extrapolate to full grid
        dp_dx_inner_2d = lvl.dp_dx_inner.reshape(lvl.shape_inner)
        dp_dy_inner_2d = lvl.dp_dy_inner.reshape(lvl.shape_inner)
        dp_dx_2d = self._extrapolate_to_full_grid(dp_dx_inner_2d)
        dp_dy_2d = self._extrapolate_to_full_grid(dp_dy_inner_2d)

        lvl.dp_dx[:] = dp_dx_2d.ravel()
        lvl.dp_dy[:] = dp_dy_2d.ravel()

    def _compute_residuals(self, u: np.ndarray, v: np.ndarray, p: np.ndarray):
        """Compute RHS residuals for RK4 pseudo time-stepping."""
        lvl = self.level

        # Velocity derivatives
        lvl.du_dx[:] = lvl.Dx @ u
        lvl.du_dy[:] = lvl.Dy @ u
        lvl.dv_dx[:] = lvl.Dx @ v
        lvl.dv_dy[:] = lvl.Dy @ v

        # Laplacians
        lvl.lap_u[:] = lvl.Laplacian @ u
        lvl.lap_v[:] = lvl.Laplacian @ v

        # Pressure gradient (needs p array set first)
        old_p = lvl.p.copy()
        lvl.p[:] = p
        self._interpolate_pressure_gradient()
        lvl.p[:] = old_p

        # Momentum residuals
        conv_u = u * lvl.du_dx + v * lvl.du_dy
        conv_v = u * lvl.dv_dx + v * lvl.dv_dy
        nu = 1.0 / self.Re

        lvl.R_u[:] = -conv_u - lvl.dp_dx + nu * lvl.lap_u
        lvl.R_v[:] = -conv_v - lvl.dp_dy + nu * lvl.lap_v

        # Continuity residual (on inner grid)
        divergence_full = lvl.du_dx + lvl.dv_dy
        divergence_2d = divergence_full.reshape(lvl.shape_full)
        divergence_inner = divergence_2d[1:-1, 1:-1].ravel()
        lvl.R_p[:] = -self.beta_squared * divergence_inner

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

        # Compute residuals for convergence check
        u_res = np.linalg.norm(lvl.u - lvl.u_prev)
        v_res = np.linalg.norm(lvl.v - lvl.v_prev)

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
    coarse_tolerance_factor: float = 1.0,  # Not used anymore, kept for API compat
) -> Tuple[SpectralLevel, int, bool]:
    """Solve using Full Single Grid (FSG) multigrid.

    Solves sequentially from coarsest to finest level, using the converged
    solution on each level as initial guess for the next finer level.

    Per Zhang & Xi (2010): Each level converges to the SAME global tolerance
    before prolongating to the next finer level.

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

    total_iterations = 0
    n_levels = len(levels)

    for level_idx, level in enumerate(levels):
        is_finest = level_idx == n_levels - 1

        # Use same tolerance on ALL levels (per Zhang & Xi 2010)
        level_tol = tolerance

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
            prolongate_solution(levels[level_idx - 1], level, transfer_ops)

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
