import numpy as np
from numba import njit

@njit(inline="always")
def MUSCL(r):
    return max(0.0, min(2.0, 2.0 * r, 0.5 * (1 + r))) if r > 0 else 0.0

@njit(inline="always")
def OSPRE(r):
    return (3 * r * (r + 1)) / (2 * (r * r + r + 1 + 1e-12)) if r > 0 else 0.0

@njit(inline="always")
def H_Cui(r):
    return (3 * (r + abs(r))) / (2 * (r + 2 + 1e-12)) if r > 0 else 0.0

@njit(inline="always", cache=True, fastmath=True)
def compute_convective_stencil(
    f, mesh, rho, mdot, u_field, grad_phi, component_idx,
    phi, scheme="Upwind", limiter=None
):
    P = mesh.owner_cells[f]
    N = mesh.neighbor_cells[f]

    g_f = mesh.face_interp_factors[f]
    d_CE = np.ascontiguousarray(mesh.vector_d_CE[f])

    # Moukalled 15.72 (negative sign for neighbor handled in matrix assembly)
    Flux_P_f = max(mdot[f], 0)
    Flux_N_f = -max(-mdot[f],0)


    # stuff for TVD and other HO schemes
    phi_P = phi[P]
    phi_N = phi[N]
    F_low = mdot[f] * (phi_P if mdot[f]  >= 0 else phi_N)

    gradC = grad_phi[P]
    gradN = grad_phi[N]
    grad_f = g_f * gradN + (1 - g_f) * gradC

    # For orthogonal Cartesian grids, skewness = 0, so no correction needed
    grad_f_mark = grad_f

    d_Cf = d_CE * g_f


    if scheme == "TVD":  
        # Compute the limiter
        if limiter is None:
            psi = 1.0 # numba type safeguard
            
        # Determine upwind and downwind cells based on mass flux direction
        if mdot[f] >= 0:
            # Flow from P to N
            phi_up = phi_P
            phi_down = phi_N
            # For P as upwind, W is the upwind neighbor of P
            phi_W = 2 * phi_P - phi_N  # Linear extrapolation from P to W
            r = (phi_N - phi_P) / (phi_P - phi_W + 1e-12)
        else:
            # Flow from N to P
            phi_up = phi_N
            phi_down = phi_P
            # For N as upwind, W is the upwind neighbor of N
            phi_W = 2 * phi_N - phi_P  # Linear extrapolation from N to W
            r = (phi_P - phi_N) / (phi_N - phi_W + 1e-12)

        if limiter == "MUSCL":
            psi = MUSCL(r)
        elif limiter == "OSPRE":
            psi = OSPRE(r)
        elif limiter == "H_Cui":
            psi = H_Cui(r)

        # Apply the limiter to get high-order face value
        phi_HO = phi_up + 0.5 * psi * (phi_down - phi_up)
        F_high = mdot[f] * phi_HO
        convDC = (F_high - F_low)
    elif scheme == "Upwind": 
        convDC = 0.0 
    elif scheme != "Upwind":
        # set coefficients
        if scheme == "Central difference":
            a = 0.0
            b = 1.0
        elif scheme == "SOU":
            a = 2.0
            b = -1.0
        elif scheme == "QUICK":
            a = 0.5
            b = 0.5
        # Compute the high order term
        phi_HO = phi_P +  np.dot(gradC * a + grad_f_mark * b, d_Cf)
        F_high = mdot[f] * phi_HO
        convDC = (F_high - F_low)
    


    return Flux_P_f, Flux_N_f, convDC

@njit(inline="always", cache=True, fastmath=True)
def compute_boundary_convective_flux(f, mdot, bc_value):
    """
    First-order upwind boundary convection flux for velocity component.

    All velocity boundaries use Dirichlet BC with known face values.
    The flux is the mass flux times the boundary value.
    """
    # Dirichlet BC: face value is known (bc_value)
    # Flux = mass_flux * boundary_value (goes to source term)
    return mdot[f], -mdot[f] * bc_value