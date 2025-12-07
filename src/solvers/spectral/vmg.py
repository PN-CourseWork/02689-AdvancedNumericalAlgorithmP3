"""V-cycle MultiGrid (VMG) spectral solver for lid-driven cavity.

VMG extends the base SG solver with V-cycle multigrid acceleration using
FAS (Full Approximation Storage) scheme for the nonlinear NS equations.

Based on Zhang & Xi (2010): "An explicit Chebyshev pseudospectral multigrid
method for incompressible Navier-Stokes equations"
"""

import logging
import time

from .sg import SGSolver
from solvers.spectral.multigrid.fsg import build_hierarchy, solve_vmg
from solvers.spectral.operators.transfer_operators import create_transfer_operators

log = logging.getLogger(__name__)


class VMGSolver(SGSolver):
    """V-cycle MultiGrid (VMG) spectral solver.

    Extends the base Single Grid solver with V-cycle multigrid acceleration.
    Uses FAS (Full Approximation Storage) scheme for the nonlinear problem.

    Parameters
    ----------
    All parameters inherited from SGSolver, plus:
        n_levels : int
            Number of multigrid levels
        pre_smoothing : list of int
            Presmoothing iterations per level (coarse to fine order)
        post_smoothing : list of int
            Postsmoothing iterations per level
        prolongation_method : str
            Transfer operator for coarse-to-fine ('fft' or 'polynomial')
        restriction_method : str
            Transfer operator for fine-to-coarse ('fft' or 'injection')
    """

    def __init__(self, **kwargs):
        """Initialize VMG solver with multigrid hierarchy."""
        super().__init__(**kwargs)

        # Create transfer operators from config
        self._transfer_ops = create_transfer_operators(
            prolongation_method=self.params.prolongation_method,
            restriction_method=self.params.restriction_method,
        )

        # Build grid hierarchy (setup cost, excluded from solve wall time)
        self._levels = build_hierarchy(
            n_fine=self.params.nx,
            n_levels=self.params.n_levels,
            basis_x=self.basis_x,
            basis_y=self.basis_y,
            Lx=self.params.Lx,
            Ly=self.params.Ly,
        )

        log.info(f"VMG initialized with {self.params.n_levels} levels")

    def solve(self, tolerance: float = None, max_iter: int = None):
        """Solve using V-cycle MultiGrid (VMG).

        Parameters
        ----------
        tolerance : float, optional
            Convergence tolerance. If None, uses params.tolerance.
        max_iter : int, optional
            Maximum V-cycles. If None, uses params.max_iterations.
        """
        if tolerance is None:
            tolerance = self.params.tolerance
        if max_iter is None:
            max_iter = self.params.max_iterations

        log.info(f"Using VMG with {self.params.n_levels} levels")
        log.info(
            f"Transfer operators: prolongation={self.params.prolongation_method}, "
            f"restriction={self.params.restriction_method}"
        )

        # Parse smoothing iterations from config
        pre_smoothing = getattr(self.params, 'pre_smoothing', None)
        post_smoothing = getattr(self.params, 'post_smoothing', None)
        correction_damping = getattr(self.params, 'correction_damping', 0.2)

        if pre_smoothing is not None:
            log.info(f"Pre-smoothing iterations: {pre_smoothing}")
        if post_smoothing is not None:
            log.info(f"Post-smoothing iterations: {post_smoothing}")
        log.info(f"Correction damping: {correction_damping}")

        # Solve using VMG (wall time starts here, after setup)
        time_start = time.time()
        finest_level, total_iters, converged = solve_vmg(
            levels=self._levels,
            Re=self.params.Re,
            beta_squared=self.params.beta_squared,
            lid_velocity=self.params.lid_velocity,
            CFL=self.params.CFL,
            tolerance=tolerance,
            max_iterations=max_iter,
            transfer_ops=self._transfer_ops,
            corner_treatment=self.corner_treatment,
            Lx=self.params.Lx,
            Ly=self.params.Ly,
            pre_smoothing=pre_smoothing,
            post_smoothing=post_smoothing,
            correction_damping=correction_damping,
        )

        time_end = time.time()
        wall_time = time_end - time_start

        # Copy solution from finest level to solver arrays
        self.arrays.u[:] = finest_level.u
        self.arrays.v[:] = finest_level.v
        self.arrays.p[:] = finest_level.p

        # Compute final residuals
        self._compute_residuals(self.arrays.u, self.arrays.v, self.arrays.p)

        # Store results using base class machinery
        final_residuals = self._compute_algebraic_residuals()
        residual_history = [
            {
                "rel_iter": tolerance if converged else tolerance * 10,
                "u_eq": final_residuals["u_residual"],
                "v_eq": final_residuals["v_residual"],
                "continuity": final_residuals["continuity_residual"],
            }
        ]

        self._store_results(
            residual_history=residual_history,
            final_iter_count=total_iters,
            is_converged=converged,
            wall_time=wall_time,
            energy_history=[self._compute_energy()],
            enstrophy_history=[self._compute_enstrophy()],
            palinstrophy_history=[self._compute_palinstrophy()],
        )

        log.info(
            f"VMG completed in {wall_time:.2f}s: {total_iters} iterations, "
            f"converged={converged}"
        )
