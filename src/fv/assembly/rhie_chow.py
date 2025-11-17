import numpy as np
from numba import njit, prange


@njit(inline="always", cache=True, fastmath=True, nogil=True)
def rhie_chow_velocity_internal_faces(mesh, U_star, grad_p_bar, grad_p, bold_D_bar):
    """
    Compute Rhie-Chow velocity at internal faces.
    Optimized for memory access patterns with pre-fetched static data.
    """
    n_internal = mesh.internal_faces.shape[0] 
    n_total_faces = mesh.vector_S_f.shape[0]
    U_faces = np.zeros((n_total_faces, 2), dtype=np.float64)

    # ═══ PRE-FETCH STATIC DATA FOR MEMORY OPTIMIZATION ═══
    internal_faces = mesh.internal_faces
    owner_cells = mesh.owner_cells
    neighbor_cells = mesh.neighbor_cells
    vector_S_f = mesh.vector_S_f
    face_interp_factors = mesh.face_interp_factors

    # ––– internal faces (OPTIMIZED LOOP) –––––––––––––––––––––––––––––––––––
    for i in range(n_internal):
        f = internal_faces[i]
        P = owner_cells[f]
        N = neighbor_cells[f]
        
        # Pre-fetch interpolation factor (single access)
        g = face_interp_factors[f]
        
        # Pre-fetch face area vector components
        S_f = vector_S_f[f]
        S_f_0 = S_f[0]
        S_f_1 = S_f[1]

        # Manual norm calculation
        mag_S_f = np.sqrt(S_f_0 * S_f_0 + S_f_1 * S_f_1)
        n_f_0 = S_f_0 / mag_S_f
        n_f_1 = S_f_1 / mag_S_f

        # Pre-fetch velocity and pressure gradient data (better cache locality)
        U_star_P = U_star[P]
        U_star_N = U_star[N]
        grad_p_P = grad_p[P]
        grad_p_N = grad_p[N]
        bold_D_P = bold_D_bar[f]  # Already at face

        # Velocity interpolation with pre-fetched data
        U_f_0 = (1.0 - g) * U_star_P[0] + g * U_star_N[0]
        U_f_1 = (1.0 - g) * U_star_P[1] + g * U_star_N[1]

        # Pressure gradient interpolation with cached components
        grad_p_f_0 = (1.0 - g) * grad_p_P[0] + g * grad_p_N[0]
        grad_p_f_1 = (1.0 - g) * grad_p_P[1] + g * grad_p_N[1]

        # Face-centered gradient correction (using pre-fetched gradient)
        grad_p_bar_f = grad_p_bar[f]
        grad_p_f_corr_0 = grad_p_bar_f[0] - grad_p_f_0
        grad_p_f_corr_1 = grad_p_bar_f[1] - grad_p_f_1

        # Rhie-Chow correction with manual operations
        correction_0 = bold_D_P[0] * grad_p_f_corr_0
        correction_1 = bold_D_P[1] * grad_p_f_corr_1

        # Final velocity with optimization
        U_faces[f, 0] = U_f_0 - correction_0
        U_faces[f, 1] = U_f_1 - correction_1

    return U_faces


@njit(inline="always", cache=True, fastmath=True, nogil=True)
def rhie_chow_velocity_boundary_faces(mesh, U_faces):
    """
    Apply boundary conditions to Rhie-Chow velocity.
    Optimized memory access patterns with pre-fetched boundary data.
    """
    n_boundary = mesh.boundary_faces.shape[0]

    # ═══ PRE-FETCH STATIC BOUNDARY DATA ═══
    boundary_faces = mesh.boundary_faces
    boundary_values = mesh.boundary_values

    # ––– boundary faces (OPTIMIZED LOOP) –––––––––––––––––––––––––––––––––––
    for i in range(n_boundary):
        f = boundary_faces[i]

        # All velocity boundaries use Dirichlet BC: fixed velocity at boundary
        boundary_vel = boundary_values[f]
        U_faces[f, 0] = boundary_vel[0]
        U_faces[f, 1] = boundary_vel[1]

    return U_faces


@njit(cache=True, fastmath=True, nogil=True)
def mdot_calculation(mesh, rho, U_f):
    """
    Calculate mass flux through faces: mdot = rho * U_f · S_f
    Optimized for memory access patterns with pre-fetched static data.
    """
    n_internal = mesh.internal_faces.shape[0]
    n_boundary = mesh.boundary_faces.shape[0]
    n_faces = n_internal + n_boundary
    
    mdot = np.zeros(n_faces, dtype=np.float64)

    # ═══ PRE-FETCH STATIC DATA FOR MEMORY OPTIMIZATION ═══
    internal_faces = mesh.internal_faces
    boundary_faces = mesh.boundary_faces
    vector_S_f = mesh.vector_S_f

    # ––– internal faces (OPTIMIZED LOOP) –––––––––––––––––––––––––––––––––––
    for i in range(n_internal):
        f = internal_faces[i]

        # Pre-fetch velocity and area vector components (single memory access each)
        U_f_vec = U_f[f]
        S_f_vec = vector_S_f[f]
        U_f_0 = U_f_vec[0]
        U_f_1 = U_f_vec[1]
        S_f_0 = S_f_vec[0]
        S_f_1 = S_f_vec[1]

        # Manual dot product (avoid np.dot allocation)
        mdot[f] = rho * (U_f_0 * S_f_0 + U_f_1 * S_f_1)

    # ––– boundary faces (OPTIMIZED LOOP) –––––––––––––––––––––––––––––––––––
    for i in range(n_boundary):
        f = boundary_faces[i]

        # Pre-fetch components
        U_f_vec = U_f[f]
        S_f_vec = vector_S_f[f]
        U_f_0 = U_f_vec[0]
        U_f_1 = U_f_vec[1]
        S_f_0 = S_f_vec[0]
        S_f_1 = S_f_vec[1]

        # Manual dot product
        mdot[f] = rho * (U_f_0 * S_f_0 + U_f_1 * S_f_1)

    return mdot


@njit(cache=True, fastmath=True, nogil=True)
def rhie_chow_velocity(mesh, U_star, U_star_bar, U_old_bar, U_old_faces, grad_p_bar, grad_p, p, alpha_uv, bold_D_bar):
    """
    Wrapper function that maintains the same interface while using optimized memory access.
    """
    # Compute internal faces with optimized memory access
    U_faces = rhie_chow_velocity_internal_faces(mesh, U_star, grad_p_bar, grad_p, bold_D_bar)

    # Apply boundary conditions with optimized memory access
    U_faces = rhie_chow_velocity_boundary_faces(mesh, U_faces)

    return U_faces
