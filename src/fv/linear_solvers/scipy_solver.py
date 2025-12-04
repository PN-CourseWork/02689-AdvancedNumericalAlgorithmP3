"""Scipy-based linear solver using BiCGSTAB."""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import bicgstab


def scipy_solver(
    A_csr: csr_matrix,
    b_np: np.ndarray,
    M=None,  # Unused, kept for API compatibility
    tolerance=1e-6,
    max_iterations=1000,
    remove_nullspace=False,
):
    """Solve A x = b using scipy BiCGSTAB.

    Parameters
    ----------
    A_csr : csr_matrix
        Sparse matrix in CSR format.
    b_np : np.ndarray
        Right-hand side vector.
    M : unused
        Kept for API compatibility, ignored.
    tolerance : float, optional
        Convergence tolerance (default: 1e-6).
    max_iterations : int, optional
        Maximum iterations (default: 1000).
    remove_nullspace : bool, optional
        If True, removes the mean from RHS and solution (for pressure eq).

    Returns
    -------
    x_np : np.ndarray
        Solution vector.
    None
        Placeholder for API compatibility.
    """
    # Handle nullspace if requested (for pressure Poisson equation)
    b = b_np.copy()
    if remove_nullspace:
        b = b - np.mean(b)

    # Solve using BiCGSTAB
    x, info = bicgstab(A_csr, b, rtol=tolerance, atol=0, maxiter=max_iterations)

    if info != 0:
        if info > 0:
            # Did not converge but we can still use the result
            pass
        else:
            raise RuntimeError(f"BiCGSTAB failed (info={info})")

    # Remove nullspace component from solution if requested
    if remove_nullspace:
        x = x - np.mean(x)

    return x, None
