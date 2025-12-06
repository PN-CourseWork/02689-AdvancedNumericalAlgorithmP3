"""Full Single Grid (FSG) multigrid spectral solver for lid-driven cavity.

FSG extends the base SG solver with multigrid acceleration using:
- Grid hierarchy with nested Gauss-Lobatto grids
- FFT-based or polynomial-based transfer operators
- Coarse-to-fine solution sequence
"""

import logging
import time

from .sg import SGSolver
from solvers.spectral.multigrid.fsg import build_hierarchy, solve_fsg
from solvers.spectral.operators.transfer_operators import create_transfer_operators

log = logging.getLogger(__name__)


class FSGSolver(SGSolver):
    """Full Single Grid (FSG) multigrid spectral solver.

    Extends the base Single Grid solver with FSG multigrid acceleration.
    Solves on a sequence of grids from coarse to fine, using the coarse
    grid solution as initial guess for the next finer grid.

    Parameters
    ----------
    All parameters inherited from SGSolver, plus:
        n_levels : int
            Number of multigrid levels
        coarse_tolerance_factor : float
            Tolerance multiplier for coarse grids
        prolongation_method : str
            Transfer operator for coarse-to-fine ('fft' or 'polynomial')
        restriction_method : str
            Transfer operator for fine-to-coarse ('fft' or 'polynomial')
    """

    def solve(self, tolerance: float = None, max_iter: int = None):
        """Solve using Full Single Grid (FSG) multigrid.

        Parameters
        ----------
        tolerance : float, optional
            Convergence tolerance. If None, uses params.tolerance.
        max_iter : int, optional
            Maximum iterations. If None, uses params.max_iterations.
        """
        if tolerance is None:
            tolerance = self.params.tolerance
        if max_iter is None:
            max_iter = self.params.max_iterations

        log.info(f"Using FSG multigrid with {self.params.n_levels} levels")
        log.info(
            f"Transfer operators: prolongation={self.params.prolongation_method}, "
            f"restriction={self.params.restriction_method}"
        )

        # Create transfer operators from config
        transfer_ops = create_transfer_operators(
            prolongation_method=self.params.prolongation_method,
            restriction_method=self.params.restriction_method,
        )

        # Build grid hierarchy
        time_start = time.time()
        levels = build_hierarchy(
            n_fine=self.params.nx,
            n_levels=self.params.n_levels,
            basis_x=self.basis_x,
            basis_y=self.basis_y,
            Lx=self.params.Lx,
            Ly=self.params.Ly,
        )

        # Solve using FSG
        finest_level, total_iters, converged = solve_fsg(
            levels=levels,
            Re=self.params.Re,
            beta_squared=self.params.beta_squared,
            lid_velocity=self.params.lid_velocity,
            CFL=self.params.CFL,
            tolerance=tolerance,
            max_iterations=max_iter,
            transfer_ops=transfer_ops,
            corner_treatment=self.corner_treatment,
            Lx=self.params.Lx,
            Ly=self.params.Ly,
            coarse_tolerance_factor=self.params.coarse_tolerance_factor,
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
        # Create minimal residual history for compatibility
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
            f"FSG completed in {wall_time:.2f}s: {total_iters} iterations, "
            f"converged={converged}"
        )
