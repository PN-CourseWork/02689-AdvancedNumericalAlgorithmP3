"""SIMPLE iteration functions for lid-driven cavity flow.

This module provides the step() and initialization functions for SIMPLE algorithm,
separated from the iteration loop which is handled by the base solver.
"""

import numpy as np
from scipy.sparse import coo_matrix

from fv.assembly.convection_diffusion_matrix import assemble_diffusion_convection_matrix
from fv.discretization.gradient.leastSquares import compute_cell_gradients
from fv.linear_solvers.scipy_solver import scipy_solver
from fv.assembly.rhie_chow import mdot_calculation, rhie_chow_velocity
from fv.assembly.pressure_correction_eq_assembly import assemble_pressure_correction_matrix
from fv.assembly.divergence import compute_divergence_from_face_fluxes
from fv.core.corrections import velocity_correction
from fv.core.helpers import (
    bold_Dv_calculation,
    interpolate_to_face,
    relax_momentum_equation,
)


def initialize_simple_state(mesh, config):
    """Initialize SIMPLE algorithm state.

    Parameters
    ----------
    mesh : MeshData2D
        Mesh data structure
    config : FVinfo
        Configuration

    Returns
    -------
    state : dict
        Dictionary containing all SIMPLE state variables
    """
    n_cells = mesh.cell_volumes.shape[0]
    n_faces = mesh.internal_faces.shape[0] + mesh.boundary_faces.shape[0]

    # Compute fluid properties from config
    rho = 1.0
    mu = rho * config.lid_velocity * config.Lx / config.Re

    state = {
        # Fluid properties
        'rho': rho,
        'mu': mu,
        # Velocity fields
        'u': np.ascontiguousarray(np.zeros(n_cells)),
        'v': np.ascontiguousarray(np.zeros(n_cells)),
        'u_prev_iter': np.ascontiguousarray(np.zeros(n_cells)),
        'v_prev_iter': np.ascontiguousarray(np.zeros(n_cells)),
        # Pressure field
        'p': np.ascontiguousarray(np.zeros(n_cells)),
        # Mass fluxes
        'mdot': np.ascontiguousarray(np.zeros(n_faces)),
        # Linear solver settings
        'linear_solver_settings': {
            'momentum': {'type': 'bcgs', 'preconditioner': 'hypre', 'tolerance': 1e-6, 'max_iterations': 1000},
            'pressure': {'type': 'bcgs', 'preconditioner': 'hypre', 'tolerance': 1e-6, 'max_iterations': 1000}
        }
    }

    return state


def simple_step(mesh, config, state):
    """Perform one SIMPLE iteration.

    Parameters
    ----------
    mesh : MeshData2D
        Mesh data structure
    config : FVinfo
        Configuration
    state : dict
        Current state from previous iteration

    Returns
    -------
    state : dict
        Updated state after one iteration
    """
    n_cells = mesh.cell_volumes.shape[0]

    # Extract state variables
    u = state['u']
    v = state['v']
    u_prev_iter = state['u_prev_iter']
    v_prev_iter = state['v_prev_iter']
    p = state['p']
    mdot = state['mdot']
    rho = state['rho']
    mu = state['mu']
    linear_solver_settings = state['linear_solver_settings']

    #=============================================================================
    # PRECOMPUTE QUANTITIES
    #=============================================================================
    grad_p = compute_cell_gradients(mesh, p, weighted=True, weight_exponent=0.5, use_limiter=False)
    grad_p_bar = interpolate_to_face(mesh, grad_p)

    # Stack u and v for interpolation functions
    U_prev_iter = np.column_stack([u_prev_iter, v_prev_iter])

    grad_u = compute_cell_gradients(mesh, u, weighted=True, weight_exponent=0.5, use_limiter=True)
    grad_v = compute_cell_gradients(mesh, v, weighted=True, weight_exponent=0.5, use_limiter=True)

    #=============================================================================
    # ASSEMBLE and solve U-MOMENTUM EQUATIONS
    #=============================================================================
    row, col, data, b_u = assemble_diffusion_convection_matrix(
        mesh, mdot, grad_u, U_prev_iter, rho, mu, 0, phi=u,
        scheme=config.convection_scheme, limiter=config.limiter
    )
    A_u = coo_matrix((data, (row, col)), shape=(n_cells, n_cells)).tocsr()
    A_u_diag = A_u.diagonal()
    rhs_u = b_u - grad_p[:, 0] * mesh.cell_volumes

    # Relax
    relaxed_A_u_diag, rhs_u = relax_momentum_equation(rhs_u, A_u_diag, u_prev_iter, config.alpha_uv)
    A_u.setdiag(relaxed_A_u_diag)

    # Solve
    u_star, _, _= scipy_solver(A_u, rhs_u, **linear_solver_settings['momentum'])
    A_u.setdiag(A_u_diag)  # Restore original diagonal

    #=============================================================================
    # ASSEMBLE and solve V-MOMENTUM EQUATIONS
    #=============================================================================
    row, col, data, b_v = assemble_diffusion_convection_matrix(
        mesh, mdot, grad_v, U_prev_iter, rho, mu, 1, phi=v,
        scheme=config.convection_scheme, limiter=config.limiter
    )
    A_v = coo_matrix((data, (row, col)), shape=(n_cells, n_cells)).tocsr()
    A_v_diag = A_v.diagonal()
    rhs_v = b_v - grad_p[:, 1] * mesh.cell_volumes

    # Relax
    relaxed_A_v_diag, rhs_v = relax_momentum_equation(rhs_v, A_v_diag, v_prev_iter, config.alpha_uv)
    A_v.setdiag(relaxed_A_v_diag)

    # Solve
    v_star, _, _= scipy_solver(A_v, rhs_v, **linear_solver_settings['momentum'])
    A_v.setdiag(A_v_diag)  # Restore original diagonal

    #=============================================================================
    # RHIE-CHOW, MASS FLUX, and PRESSURE CORRECTION
    #=============================================================================
    bold_D = bold_Dv_calculation(mesh, A_u_diag, A_v_diag)
    bold_D_bar = interpolate_to_face(mesh, bold_D)

    # Stack u_star and v_star for Rhie-Chow
    U_star = np.column_stack([u_star, v_star])
    U_star_rc = rhie_chow_velocity(mesh, U_star, grad_p_bar, grad_p, bold_D_bar)

    mdot_star = mdot_calculation(mesh, rho, U_star_rc)

    row, col, data = assemble_pressure_correction_matrix(mesh, rho)
    A_p = coo_matrix((data, (row, col)), shape=(n_cells, n_cells)).tocsr()
    rhs_p = -compute_divergence_from_face_fluxes(mesh, mdot_star)

    p_prime, _, _ = scipy_solver(A_p, rhs_p, remove_nullspace=True, **linear_solver_settings['pressure'])

    #=============================================================================
    # CORRECTIONS (SIMPLE)
    #=============================================================================
    grad_p_prime = compute_cell_gradients(mesh, p_prime, weighted=True, weight_exponent=0.5, use_limiter=False)
    U_prime = velocity_correction(mesh, grad_p_prime, bold_D)
    u_prime = U_prime[:, 0]
    v_prime = U_prime[:, 1]

    u_corrected = u_star + u_prime
    v_corrected = v_star + v_prime

    U_prime_face = interpolate_to_face(mesh, U_prime)

    mdot_prime = mdot_calculation(mesh, rho, U_prime_face)
    mdot_corrected = mdot_star + mdot_prime

    p_corrected = p + config.alpha_p * p_prime

    # Update state
    state['u'] = u_corrected
    state['v'] = v_corrected
    state['p'] = p_corrected
    state['u_prev_iter'] = u_corrected.copy()
    state['v_prev_iter'] = v_corrected.copy()
    state['mdot'] = mdot_corrected

    return state
