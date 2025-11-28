import numpy as np
from numba import njit, prange


@njit(parallel=True)
def velocity_correction(
    mesh, grad_p_prime_cell, bold_D_cell, u_prime=None, v_prime=None
):
    """
    Apply velocity correction: U' = -D_U * grad(p')
    Returns separate u_prime and v_prime components.
    """
    n_cells = mesh.cell_centers.shape[0]
    if u_prime is None:
        u_prime = np.zeros(n_cells)
    if v_prime is None:
        v_prime = np.zeros(n_cells)

    for i in prange(n_cells):
        correction = -bold_D_cell[i] * grad_p_prime_cell[i]
        u_prime[i] = correction[0]
        v_prime[i] = correction[1]

    return u_prime, v_prime
