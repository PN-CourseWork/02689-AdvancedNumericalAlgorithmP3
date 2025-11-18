import numpy as np



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

    for i in range(n):
        a = A_diag[i]
        a_relaxed = a * inv_alpha
        relaxed_diagonal[i] = a_relaxed
        relaxed_rhs[i] = rhs[i] + scale * a * phi[i]  

    return relaxed_diagonal, relaxed_rhs




def interpolate_to_face(mesh, quantity):
    """
    Optimized interpolation to faces with better memory access patterns.
    Handles both scalar and vector quantities efficiently.
    """
    n_faces = mesh.face_areas.shape[0]
    n_internal = mesh.internal_faces.shape[0]
    n_boundary = mesh.boundary_faces.shape[0]
    
    if quantity.ndim == 1:
        # Scalar field
        interpolated_quantity = np.zeros((n_faces, 1), dtype=np.float64)
        
        # Process internal faces
        for i in range(n_internal):
            f = mesh.internal_faces[i]
            P = mesh.owner_cells[f]
            N = mesh.neighbor_cells[f]
            gf = mesh.face_interp_factors[f]
            interpolated_quantity[f, 0] = gf * quantity[N] + (1.0 - gf) * quantity[P]

        # Process boundary faces
        for i in range(n_boundary):
            f = mesh.boundary_faces[i]
            P = mesh.owner_cells[f]
            interpolated_quantity[f, 0] = quantity[P]
            
    else:
        # Vector field - optimized for common 2D case
        n_components = quantity.shape[1]
        interpolated_quantity = np.zeros((n_faces, n_components), dtype=np.float64)
        
        if n_components == 2:
            # Optimized 2D vector case with manual unrolling
            for i in range(n_internal):
                f = mesh.internal_faces[i]
                P = mesh.owner_cells[f]
                N = mesh.neighbor_cells[f]
                gf = mesh.face_interp_factors[f]
                
                interpolated_quantity[f, 0] = gf * quantity[N, 0] + (1.0 - gf) * quantity[P, 0]
                interpolated_quantity[f, 1] = gf * quantity[N, 1] + (1.0 - gf) * quantity[P, 1]

            for i in range(n_boundary):
                f = mesh.boundary_faces[i]
                P = mesh.owner_cells[f]
                interpolated_quantity[f, 0] = quantity[P, 0]
                interpolated_quantity[f, 1] = quantity[P, 1]
        else:
            # General vector case
            for i in range(n_internal):
                f = mesh.internal_faces[i]
                P = mesh.owner_cells[f]
                N = mesh.neighbor_cells[f]
                gf = mesh.face_interp_factors[f]
                
                for c in range(n_components):
                    interpolated_quantity[f, c] = gf * quantity[N, c] + (1.0 - gf) * quantity[P, c]

            for i in range(n_boundary):
                f = mesh.boundary_faces[i]
                P = mesh.owner_cells[f]
                for c in range(n_components):
                    interpolated_quantity[f, c] = quantity[P, c]

    return interpolated_quantity



def bold_Dv_calculation(mesh, A_u_diag, A_v_diag):
    n_cells = mesh.cell_volumes.shape[0]
    bold_Dv = np.zeros((n_cells, 2), dtype=np.float64)

    for i in range(n_cells):
        bold_Dv[i, 0] = mesh.cell_volumes[i] / (A_u_diag[i] + 1e-14)  # D_u
        bold_Dv[i, 1] = mesh.cell_volumes[i] / (A_v_diag[i] + 1e-14)  # D_v

    return bold_Dv