"""Vectorized matrix assembly for structured Cartesian grids.

This module provides highly optimized matrix assembly routines that exploit
the regular structure of Cartesian grids to eliminate loops and use NumPy
array operations.
"""
import numpy as np


def assemble_diffusion_convection_matrix_structured(
    mesh,
    mdot,
    grad_phi,
    rho,
    mu,
    component_idx,
    phi,
    scheme="Upwind",
    limiter=None
):
    """Vectorized assembly for structured grids (much faster than generic version).

    Parameters
    ----------
    mesh : MeshData2D
        Structured mesh with nx, ny attributes
    mdot : ndarray
        Mass flux at faces
    grad_phi : ndarray
        Gradient field (unused for upwind, kept for compatibility)
    rho : float
        Density
    mu : float
        Dynamic viscosity
    component_idx : int
        Component index (0=u, 1=v)
    phi : ndarray
        Field values at cell centers
    scheme : str
        Convection scheme ("Upwind" supported)
    limiter : str
        Limiter (not used for upwind)

    Returns
    -------
    row, col, data, b : tuple
        Sparse matrix triplet format and RHS vector
    """
    n_cells = mesh.cell_volumes.shape[0]
    n_internal = mesh.internal_faces.shape[0]
    n_boundary = mesh.boundary_faces.shape[0]

    # Pre-allocate arrays
    max_nnz = 8 * n_internal + 3 * n_boundary
    row = np.zeros(max_nnz, dtype=np.int64)
    col = np.zeros(max_nnz, dtype=np.int64)
    data = np.zeros(max_nnz, dtype=np.float64)
    b = np.zeros(n_cells, dtype=np.float64)
    idx = 0

    # ========== VECTORIZED INTERNAL FACES ==========
    internal_faces = mesh.internal_faces
    P = mesh.owner_cells[internal_faces]
    N = mesh.neighbor_cells[internal_faces]

    # Vectorized convection fluxes (upwind scheme)
    mdot_int = mdot[internal_faces]
    convFlux_P_f = np.maximum(mdot_int, 0.0)
    convFlux_N_f = -np.maximum(-mdot_int, 0.0)

    # Vectorized diffusion fluxes
    vector_S_f = mesh.vector_S_f[internal_faces]
    vector_d_CE = mesh.vector_d_CE[internal_faces]
    E_mag = np.linalg.norm(vector_S_f, axis=1)
    d_mag = np.linalg.norm(vector_d_CE, axis=1)
    geoDiff = E_mag / d_mag
    diffFlux_P_f = mu * geoDiff
    diffFlux_N_f = -mu * geoDiff

    # Combined fluxes
    Flux_P_f = convFlux_P_f + diffFlux_P_f
    Flux_N_f = convFlux_N_f + diffFlux_N_f

    # Assemble internal face contributions (vectorized)
    n_int = len(internal_faces)

    # P-P diagonal
    row[idx:idx+n_int] = P
    col[idx:idx+n_int] = P
    data[idx:idx+n_int] = Flux_P_f
    idx += n_int

    # P-N off-diagonal
    row[idx:idx+n_int] = P
    col[idx:idx+n_int] = N
    data[idx:idx+n_int] = Flux_N_f
    idx += n_int

    # N-N diagonal
    row[idx:idx+n_int] = N
    col[idx:idx+n_int] = N
    data[idx:idx+n_int] = -Flux_N_f
    idx += n_int

    # N-P off-diagonal
    row[idx:idx+n_int] = N
    col[idx:idx+n_int] = P
    data[idx:idx+n_int] = -Flux_P_f
    idx += n_int

    # ========== VECTORIZED BOUNDARY FACES ==========
    boundary_faces = mesh.boundary_faces
    P_b = mesh.owner_cells[boundary_faces]
    bc_val = mesh.boundary_values[boundary_faces, component_idx]

    # Vectorized diffusion at boundary
    E_f = mesh.vector_S_f[boundary_faces]
    E_mag_b = np.linalg.norm(E_f, axis=1)
    d_PB = mesh.d_Cb[boundary_faces]
    diffFlux_P_b = mu * E_mag_b / d_PB
    diffFlux_N_b = -diffFlux_P_b * bc_val

    # Vectorized convection at boundary
    mdot_b = mdot[boundary_faces]
    convFlux_P_b = mdot_b
    convFlux_N_b = -mdot_b * bc_val

    # Combined boundary fluxes
    Flux_total_b = diffFlux_P_b + convFlux_P_b

    # Assemble boundary contributions
    n_bnd = len(boundary_faces)
    row[idx:idx+n_bnd] = P_b
    col[idx:idx+n_bnd] = P_b
    data[idx:idx+n_bnd] = Flux_total_b
    idx += n_bnd

    # RHS contributions from boundaries (vectorized)
    np.add.at(b, P_b, -(diffFlux_N_b + convFlux_N_b))

    return row[:idx], col[:idx], data[:idx], b
