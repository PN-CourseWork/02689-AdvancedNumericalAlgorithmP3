from numba import njit


@njit()
def MUSCL(r):
    return max(0.0, min(2.0, 2.0 * r, 0.5 * (1 + r))) if r > 0 else 0.0


@njit()
def compute_convective_stencil(f, mesh, mdot, phi, scheme):
    P = mesh.owner_cells[f]
    N = mesh.neighbor_cells[f]

    # Moukalled 15.72 (negative sign for neighbor handled in matrix assembly)
    Flux_P_f = max(mdot[f], 0)
    Flux_N_f = -max(-mdot[f], 0)

    if scheme == "Upwind":
        convDC = 0.0
    elif scheme == "TVD":
        # Variables needed for TVD
        phi_P = phi[P]
        phi_N = phi[N]
        F_low = mdot[f] * (phi_P if mdot[f] >= 0 else phi_N)

        # Compute the limiter
        psi = 1.0  # numba type safeguard

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

            psi = MUSCL(r)

        # Apply the limiter to get high-order face value
        phi_HO = phi_up + 0.5 * psi * (phi_down - phi_up)
        F_high = mdot[f] * phi_HO
        convDC = F_high - F_low

    return Flux_P_f, Flux_N_f, convDC
