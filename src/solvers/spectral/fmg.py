"""Full MultiGrid (FMG) spectral solver for lid-driven cavity.

FMG will extend the base SG solver with Full MultiGrid acceleration.
Currently not implemented.
"""

import logging

from .sg import SGSolver

log = logging.getLogger(__name__)


class FMGSolver(SGSolver):
    """Full MultiGrid (FMG) spectral solver.

    Extends the base Single Grid solver with Full MultiGrid acceleration.

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
        """Solve using Full MultiGrid (FMG).

        Parameters
        ----------
        tolerance : float, optional
            Convergence tolerance. If None, uses params.tolerance.
        max_iter : int, optional
            Maximum iterations. If None, uses params.max_iterations.
        """
        raise NotImplementedError(
            "Full MultiGrid (FMG) solver not yet implemented. "
            "Use FSG (Full Single Grid) instead."
        )
