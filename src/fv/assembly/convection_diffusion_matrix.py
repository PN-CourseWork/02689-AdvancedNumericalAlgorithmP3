import numpy as np
from numba import njit

from fv.discretization.convection.upwind import compute_convective_stencil


@njit(
    cache=True,
    fastmath=True,
    nogil=True,
    parallel=False,
    boundscheck=False,
    error_model="numpy",
)
def assemble_diffusion_convection_matrix(
    mesh, mdot, grad_phi, rho, mu, component_idx, phi, scheme="Upwind", limiter=None
):
    """Assemble sparse matrix and RHS for steady-state collocated FV discretisation.

    Optimized for memory access patterns with pre-fetched static data.
    """

    n_cells = mesh.cell_volumes.shape[0]
    n_internal = mesh.internal_faces.shape[0]
    n_boundary = mesh.boundary_faces.shape[0]

    # ––– pessimistic non-zero count ––––––––––––––––––––––––––––––––––––––––
    max_nnz = 8 * n_internal + 3 * n_boundary
    row = np.zeros(max_nnz, dtype=np.int64)
    col = np.zeros(max_nnz, dtype=np.int64)
    data = np.zeros(max_nnz, dtype=np.float64)

    idx = 0  # running write position
    b = np.zeros(n_cells, dtype=np.float64)

    # ═══ PRE-FETCH STATIC DATA (HUGE MEMORY ACCESS OPTIMIZATION) ═══
    # Internal face connectivity (static throughout simulation)
    internal_faces = mesh.internal_faces
    owner_cells = mesh.owner_cells
    neighbor_cells = mesh.neighbor_cells

    # Boundary face data (static)
    boundary_faces = mesh.boundary_faces
    boundary_values = mesh.boundary_values

    # ––– internal faces (OPTIMIZED MEMORY ACCESS) –––––––––––––––––––––––––
    for i in range(n_internal):
        f = internal_faces[i]
        P = owner_cells[f]
        N = neighbor_cells[f]

        # —— convection term (upwind) ——
        convFlux_P_f, convFlux_N_f, convDC = compute_convective_stencil(
            f,
            mesh,
            rho,
            mdot,
            grad_phi,
            component_idx,
            phi,
            scheme=scheme,
            limiter=limiter,
        )

        # —— orthogonal diffusion (inlined for clarity) ——
        vector_E_f = mesh.vector_S_f[f]
        vector_d_CE = mesh.vector_d_CE[f]
        E_mag = np.sqrt(vector_E_f[0] ** 2 + vector_E_f[1] ** 2)
        d_mag = np.sqrt(vector_d_CE[0] ** 2 + vector_d_CE[1] ** 2)
        geoDiff = E_mag / d_mag
        diffFlux_P_f = mu * geoDiff
        diffFlux_N_f = -mu * geoDiff

        # —— face fluxes —— Moukalled 15.72 ——
        Flux_P_f = convFlux_P_f + diffFlux_P_f
        Flux_N_f = convFlux_N_f + diffFlux_N_f
        Flux_V_f = convDC  # diffDC is always 0 for orthogonal grids

        # Matrix assembly (using pre-fetched P, N)
        row[idx] = P
        col[idx] = P
        data[idx] = Flux_P_f
        idx += 1
        row[idx] = P
        col[idx] = N
        data[idx] = Flux_N_f
        idx += 1
        row[idx] = N
        col[idx] = N
        data[idx] = -Flux_N_f
        idx += 1
        row[idx] = N
        col[idx] = P
        data[idx] = -Flux_P_f
        idx += 1

        b[P] -= Flux_V_f
        b[N] += Flux_V_f

    # ––– boundary faces –––––––––––––––––––––––––––––––––––––––––––––––––––––
    for i in range(n_boundary):
        f = boundary_faces[i]
        bc_val = boundary_values[f, component_idx]
        P = owner_cells[f]

        # Diffusion flux (Dirichlet BC)
        E_f = mesh.vector_S_f[f]
        E_mag = np.sqrt(E_f[0] ** 2 + E_f[1] ** 2)
        d_PB = mesh.d_Cb[f]
        diffFlux_P_b = mu * E_mag / d_PB
        diffFlux_N_b = -diffFlux_P_b * bc_val

        # Convection flux (Dirichlet BC)
        convFlux_P_b = mdot[f]
        convFlux_N_b = -mdot[f] * bc_val

        row[idx] = P
        col[idx] = P
        data[idx] = diffFlux_P_b + convFlux_P_b
        idx += 1
        b[P] -= diffFlux_N_b + convFlux_N_b

    # ––– trim overallocation –––––––––––––––––––––––––––––––––––––––––––––––
    return row[:idx], col[:idx], data[:idx], b
