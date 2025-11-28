"""Scipy-based linear solver using BiCGSTAB with PyAMG preconditioner."""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import bicgstab


def scipy_solver(A_csr: csr_matrix, b_np: np.ndarray, use_cg: bool = False):
    """Solve A x = b using BiCGSTAB with PyAMG preconditioner.

    Parameters
    ----------
    A_csr : csr_matrix
        Coefficient matrix in CSR format
    b_np : np.ndarray
        Right-hand side vector
    use_cg : bool, optional
        Unused parameter kept for API compatibility

    Returns
    -------
    np.ndarray
        Solution vector
    """
    # Solve using BiCGSTAB without preconditioner
    # PyAMG preconditioner can cause numerical issues on early iterations
    x, info = bicgstab(A_csr, b_np, rtol=1e-6, atol=0)

    if info != 0:
        raise RuntimeError(f"BiCGSTAB failed to converge (info={info})")

    return x
