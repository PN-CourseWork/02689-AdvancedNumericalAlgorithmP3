"""Scipy-based linear solver using direct method (spsolve)."""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


def scipy_solver(A_csr: csr_matrix, b_np: np.ndarray):
    """Solve A x = b using SciPy sparse direct solver (spsolve)."""
    return spsolve(A_csr, b_np)
