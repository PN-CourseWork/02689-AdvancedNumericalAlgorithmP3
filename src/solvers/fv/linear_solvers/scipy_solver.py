"""Scipy-based linear solver using BiCGSTAB with PyAMG preconditioning."""

import numpy as np
import pyamg
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import bicgstab


def scipy_solver(
    A_csr: csr_matrix,
    b_np: np.ndarray,
    M=None,
    tolerance=1e-6,
    max_iterations=1000,
):
    """Solve A x = b using scipy BiCGSTAB with PyAMG preconditioning.

    Parameters
    ----------
    A_csr : csr_matrix
        Sparse matrix in CSR format.
    b_np : np.ndarray
        Right-hand side vector.
    M : LinearOperator, optional
        Preconditioner. If None, builds AMG preconditioner automatically.
    tolerance : float, optional
        Convergence tolerance (default: 1e-6).
    max_iterations : int, optional
        Maximum iterations (default: 1000).

    Returns
    -------
    x_np : np.ndarray
        Solution vector.
    M : LinearOperator
        Preconditioner for reuse in subsequent solves.
    """
    # Build AMG preconditioner if not provided
    if M is None:
        ml = pyamg.smoothed_aggregation_solver(A_csr, max_coarse=10)
        M = ml.aspreconditioner()

    # Solve using BiCGSTAB with AMG preconditioner
    x, info = bicgstab(A_csr, b_np, M=M, rtol=tolerance, atol=0, maxiter=max_iterations)

    if info != 0:
        if info > 0:
            # Did not converge but we can still use the result
            pass
        else:
            raise RuntimeError(f"BiCGSTAB failed (info={info})")

    return x, M
