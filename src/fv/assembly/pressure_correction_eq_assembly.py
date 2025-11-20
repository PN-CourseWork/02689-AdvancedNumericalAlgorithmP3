import numpy as np
from numba import njit


@njit()
def assemble_pressure_correction_matrix(mesh, rho):
    """
    Assemble pressure correction equation matrix.
    Optimized for memory access patterns with pre-fetched static data.
    """
    n_internal = mesh.internal_faces.shape[0]

    # Pessimistic non-zero count
    max_nnz = 4 * n_internal
    row = np.zeros(max_nnz, dtype=np.int64)
    col = np.zeros(max_nnz, dtype=np.int64)
    data = np.zeros(max_nnz, dtype=np.float64)

    idx = 0

    # ═══ PRE-FETCH STATIC DATA FOR MEMORY OPTIMIZATION ═══
    internal_faces = mesh.internal_faces
    owner_cells = mesh.owner_cells
    neighbor_cells = mesh.neighbor_cells
    vector_E_f = mesh.vector_S_f
    vector_d_CE = mesh.vector_d_CE

    # ––– internal faces (OPTIMIZED LOOP) –––––––––––––––––––––––––––––––––––
    for i in range(n_internal):
        f = internal_faces[i]
        P = owner_cells[f]
        N = neighbor_cells[f]

        # Pre-fetch vector components
        E_f = vector_E_f[f]
        d_CE = vector_d_CE[f]

        # Compute vector norms
        E_mag = np.linalg.norm(E_f)
        d_mag = np.linalg.norm(d_CE)

        # Compute conductance
        D_f = rho * E_mag / d_mag

        # Matrix coefficients
        row[idx] = P
        col[idx] = P
        data[idx] = D_f
        idx += 1
        row[idx] = P
        col[idx] = N
        data[idx] = -D_f
        idx += 1
        row[idx] = N
        col[idx] = N
        data[idx] = D_f
        idx += 1
        row[idx] = N
        col[idx] = P
        data[idx] = -D_f
        idx += 1

    return row[:idx], col[:idx], data[:idx]
