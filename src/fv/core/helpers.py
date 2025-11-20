import numpy as np
from numba import njit, prange

@njit(parallel=False, cache=True)
def relax_momentum_equation(rhs, A_diag, phi, alpha):
    """
    In-place Patankar-style under-relaxation of a momentum equation system.
    Modifies `rhs` in-place, writes the relaxed diagonal into `A_diag`.
    """
    inv_alpha = 1.0 / alpha
    scale = (1.0 - alpha) / alpha
    n = rhs.shape[0]
    relaxed_diagonal = np.zeros(n, dtype=np.float64)
    relaxed_rhs = np.zeros(n, dtype=np.float64)

    for i in prange(n):
        a = A_diag[i]
        a_relaxed = a * inv_alpha
        relaxed_diagonal[i] = a_relaxed
        relaxed_rhs[i] = rhs[i] + scale * a * phi[i]

    return relaxed_diagonal, relaxed_rhs


@njit(parallel=True, cache=True)
def interpolate_velocity_to_face(mesh, u, v, out=None):
    """
    Optimized velocity interpolation that takes separate u and v components.
    Returns combined (n_faces, 2) array without intermediate column stacking.
    """
    n_faces = mesh.face_areas.shape[0]
    n_internal = mesh.internal_faces.shape[0]
    n_boundary = mesh.boundary_faces.shape[0]

    if out is None:
        U_face = np.zeros((n_faces, 2), dtype=np.float64)
    else:
        U_face = out

    # Process internal faces
    for i in prange(n_internal):
        f = mesh.internal_faces[i]
        P = mesh.owner_cells[f]
        N = mesh.neighbor_cells[f]
        gf = mesh.face_interp_factors[f]

        U_face[f, 0] = gf * u[N] + (1.0 - gf) * u[P]
        U_face[f, 1] = gf * v[N] + (1.0 - gf) * v[P]

    # Process boundary faces
    for i in prange(n_boundary):
        f = mesh.boundary_faces[i]
        P = mesh.owner_cells[f]
        U_face[f, 0] = u[P]
        U_face[f, 1] = v[P]

    return U_face


@njit(parallel=True, cache=True)
def interpolate_to_face(mesh, quantity, out=None):
    """
    Optimized interpolation to faces with better memory access patterns.
    Handles both scalar and vector quantities efficiently.
    """
    n_faces = mesh.face_areas.shape[0]
    n_internal = mesh.internal_faces.shape[0]
    n_boundary = mesh.boundary_faces.shape[0]

    if quantity.ndim == 1:
        # Scalar field
        if out is None:
            interpolated_quantity = np.zeros((n_faces, 1), dtype=np.float64)
        else:
            interpolated_quantity = out

        # Process internal faces
        for i in prange(n_internal):
            f = mesh.internal_faces[i]
            P = mesh.owner_cells[f]
            N = mesh.neighbor_cells[f]
            gf = mesh.face_interp_factors[f]
            interpolated_quantity[f, 0] = gf * quantity[N] + (1.0 - gf) * quantity[P]

        # Process boundary faces
        for i in prange(n_boundary):
            f = mesh.boundary_faces[i]
            P = mesh.owner_cells[f]
            interpolated_quantity[f, 0] = quantity[P]

    else:
        # Vector field - optimized for common 2D case
        n_components = quantity.shape[1]
        if out is None:
            interpolated_quantity = np.zeros((n_faces, n_components), dtype=np.float64)
        else:
            interpolated_quantity = out

        if n_components == 2:
            # Optimized 2D vector case with manual unrolling
            for i in prange(n_internal):
                f = mesh.internal_faces[i]
                P = mesh.owner_cells[f]
                N = mesh.neighbor_cells[f]
                gf = mesh.face_interp_factors[f]

                interpolated_quantity[f, 0] = (
                    gf * quantity[N, 0] + (1.0 - gf) * quantity[P, 0]
                )
                interpolated_quantity[f, 1] = (
                    gf * quantity[N, 1] + (1.0 - gf) * quantity[P, 1]
                )

            for i in prange(n_boundary):
                f = mesh.boundary_faces[i]
                P = mesh.owner_cells[f]
                interpolated_quantity[f, 0] = quantity[P, 0]
                interpolated_quantity[f, 1] = quantity[P, 1]
        else:
            # General vector case
            for i in prange(n_internal):
                f = mesh.internal_faces[i]
                P = mesh.owner_cells[f]
                N = mesh.neighbor_cells[f]
                gf = mesh.face_interp_factors[f]

                for c in range(n_components):
                    interpolated_quantity[f, c] = (
                        gf * quantity[N, c] + (1.0 - gf) * quantity[P, c]
                    )

            for i in prange(n_boundary):
                f = mesh.boundary_faces[i]
                P = mesh.owner_cells[f]
                for c in range(n_components):
                    interpolated_quantity[f, c] = quantity[P, c]

    return interpolated_quantity


@njit(parallel=False, cache=True)
def bold_Dv_calculation(mesh, A_u_diag, A_v_diag, out=None):
    n_cells = mesh.cell_volumes.shape[0]
    if out is None:
        bold_Dv = np.zeros((n_cells, 2), dtype=np.float64)
    else:
        bold_Dv = out

    for i in prange(n_cells):
        bold_Dv[i, 0] = mesh.cell_volumes[i] / (A_u_diag[i] + 1e-14)  # D_u
        bold_Dv[i, 1] = mesh.cell_volumes[i] / (A_v_diag[i] + 1e-14)  # D_v

    return bold_Dv
