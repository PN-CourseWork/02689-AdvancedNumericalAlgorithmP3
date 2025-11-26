"""Scipy-based linear solver using BiCGSTAB with PyAMG preconditioner."""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import bicgstab
import pyamg


def scipy_solver(A_csr: csr_matrix, b_np: np.ndarray):
    """Solve A x = b using BiCGSTAB with PyAMG smoothed aggregation preconditioner.

    Parameters
    ----------
    A_csr : csr_matrix
        Coefficient matrix in CSR format
    b_np : np.ndarray
        Right-hand side vector

    Returns
    -------
    np.ndarray
        Solution vector
    """
    # Create PyAMG smoothed aggregation preconditioner
    #ml = pyamg.smoothed_aggregation_solver(A_csr)
    #M = ml.aspreconditioner()

    # Solve using BiCGSTAB with preconditioner
    #x, info = bicgstab(A_csr, b_np, M=M, rtol=1e-8, atol=1e-8)
    x, info = bicgstab(A_csr, b_np,rtol=1e-6)

    if info != 0:
        raise RuntimeError(f"BiCGSTAB failed to converge (info={info})")

    return x
