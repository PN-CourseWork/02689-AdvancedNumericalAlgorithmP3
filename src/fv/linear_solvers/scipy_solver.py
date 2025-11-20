"""Scipy-based linear solver using direct method (spsolve)."""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


def scipy_solver(
    A_csr: csr_matrix,
    b_np: np.ndarray,
    remove_nullspace: bool = False,
    **kwargs  # Accept and ignore extra parameters for flexible calling
):
    """
    Solve A x = b using SciPy sparse direct solver (spsolve).

    """

    # Handle nullspace if requested (pressure correction equation)
    if remove_nullspace:
        # Pin first value to zero to remove nullspace
        A_modified = A_csr.tolil()  # Convert to LIL for efficient row modification
        A_modified[0, :] = 0.0
        A_modified[0, 0] = 1.0
        A_work = A_modified.tocsr()

        b_modified = b_np.copy()
        b_modified[0] = 0.0
        b_work = b_modified
    else:
        A_work = A_csr
        b_work = b_np

    # Solve 
    x_np = spsolve(A_work, b_work)

    return x_np 
