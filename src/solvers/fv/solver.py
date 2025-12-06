"""Finite volume solver for lid-driven cavity.

This module implements a collocated finite volume solver using
SIMPLE algorithm for pressure-velocity coupling.
"""

import numpy as np
from scipy.sparse import csr_matrix

from ..base import LidDrivenCavitySolver
from ..datastructures import FVParameters, FVSolverFields

from solvers.fv.assembly.convection_diffusion_matrix import (
    assemble_diffusion_convection_matrix,
)
from solvers.fv.discretization.gradient.structured_gradient import (
    compute_cell_gradients_structured,
)
from solvers.fv.linear_solvers.scipy_solver import scipy_solver
from solvers.fv.assembly.rhie_chow import mdot_calculation, rhie_chow_velocity
from solvers.fv.assembly.pressure_correction_eq_assembly import (
    assemble_pressure_correction_matrix,
)
from solvers.fv.assembly.divergence import compute_divergence_from_face_fluxes
from solvers.fv.core.corrections import velocity_correction
from solvers.fv.core.helpers import (
    bold_Dv_calculation,
    interpolate_to_face,
    interpolate_velocity_to_face,
    relax_momentum_equation,
)


class FVSolver(LidDrivenCavitySolver):
    """Finite volume solver for lid-driven cavity problem.

    This solver uses a collocated grid arrangement with Rhie-Chow interpolation
    for pressure-velocity coupling using the SIMPLE algorithm.

    Parameters
    ----------
    params : FVParameters
        Parameters with physics (Re, lid velocity, domain size) and
        FV-specific settings (nx, ny, convection scheme, etc.).
    """

    Parameters = FVParameters
    rho = 1.0  # Constant fluid density

    def __init__(self, **kwargs):
        """Initialize FV solver."""
        super().__init__(**kwargs)

        # Create mesh
        from shared.meshing.simple_structured import create_structured_mesh_2d

        self.mesh = create_structured_mesh_2d(
            nx=self.params.nx,
            ny=self.params.ny,
            Lx=self.params.Lx,
            Ly=self.params.Ly,
            lid_velocity=self.params.lid_velocity,
        )

        # Get dimensions from mesh
        n_cells = self.mesh.cell_volumes.shape[0]
        n_faces = self.mesh.internal_faces.shape[0] + self.mesh.boundary_faces.shape[0]

        # Compute fluid properties
        self.mu = self.rho * self.params.lid_velocity * self.params.Lx / self.params.Re

        # Allocate internal solver arrays
        self.arrays = FVSolverFields.allocate(n_cells, n_faces)

        # Initialize output fields (base class handles this)
        self._init_fields(
            x=self.mesh.cell_centers[:, 0], y=self.mesh.cell_centers[:, 1]
        )

        # Cache commonly used values
        self.n_cells = n_cells
        self.shape_full = (self.params.ny, self.params.nx)
        self.dx_min = self.params.Lx / self.params.nx
        self.dy_min = self.params.Ly / self.params.ny

    def _solve_momentum_equation(
        self, component_idx, phi, grad_phi, phi_prev_iter, grad_p_component, M
    ):
        """Solve a single momentum equation (u or v).

        Parameters
        ----------
        component_idx : int
            Component index (0 for u, 1 for v)
        phi : ndarray
            Current velocity component (u or v)
        grad_phi : ndarray
            Gradient of velocity component
        phi_prev_iter : ndarray
            Previous iteration velocity component
        grad_p_component : ndarray
            Pressure gradient component (x or y)
        M : LinearOperator or None
            Reusable preconditioner (updated in-place)

        Returns
        -------
        phi_star : ndarray
            Predicted velocity component
        A_diag : ndarray
            Diagonal of momentum matrix (needed for pressure correction)
        M : LinearOperator
            Updated preconditioner for reuse
        """
        # Assemble momentum equation
        row, col, data, b = assemble_diffusion_convection_matrix(
            self.mesh,
            self.arrays.mdot,
            grad_phi,
            self.rho,
            self.mu,
            component_idx,
            phi=phi,
            scheme=self.params.convection_scheme,
            limiter=self.params.limiter,
        )
        A = csr_matrix((data, (row, col)), shape=(self.n_cells, self.n_cells))
        A_diag = A.diagonal()
        rhs = b - grad_p_component * self.mesh.cell_volumes

        # Apply under-relaxation
        relaxed_A_diag, rhs = relax_momentum_equation(
            rhs, A_diag, phi_prev_iter, self.params.alpha_uv
        )
        A.setdiag(relaxed_A_diag)

        # Solve with scipy
        phi_star, M = scipy_solver(
            A,
            rhs,
            M=M,
            tolerance=self.params.linear_solver_tol,
        )

        return phi_star, A_diag, M

    def step(self):
        """Perform one SIMPLE iteration.

        Returns
        -------
        u, v, p : np.ndarray
            Updated velocity and pressure fields
        """
        a = self.arrays  # Shorthand for readability

        # Swap buffers at start (zero-copy)
        a.u, a.u_prev = a.u_prev, a.u
        a.v, a.v_prev = a.v_prev, a.v

        # Compute pressure gradient (no limiter for pressure) - reuse buffers
        compute_cell_gradients_structured(
            self.mesh, a.p, use_limiter=False, out=a.grad_p
        )
        interpolate_to_face(self.mesh, a.grad_p, out=a.grad_p_bar)

        # Compute velocity gradients (with limiter) - reuse buffers
        compute_cell_gradients_structured(
            self.mesh, a.u_prev, use_limiter=True, out=a.grad_u
        )
        compute_cell_gradients_structured(
            self.mesh, a.v_prev, use_limiter=True, out=a.grad_v
        )

        # Solve momentum equations (reuse preconditioners)
        u_star, A_u_diag, a.M_u = self._solve_momentum_equation(
            0, a.u_prev, a.grad_u, a.u_prev, a.grad_p[:, 0], a.M_u
        )
        v_star, A_v_diag, a.M_v = self._solve_momentum_equation(
            1, a.v_prev, a.grad_v, a.v_prev, a.grad_p[:, 1], a.M_v
        )

        # Pressure correction - reuse buffers
        bold_Dv_calculation(self.mesh, A_u_diag, A_v_diag, out=a.bold_D)
        interpolate_to_face(self.mesh, a.bold_D, out=a.bold_D_bar)

        rhie_chow_velocity(
            self.mesh,
            u_star,
            v_star,
            a.grad_p_bar,
            a.grad_p,
            a.bold_D_bar,
            out=a.U_star_rc,
        )

        mdot_calculation(self.mesh, self.rho, a.U_star_rc, out=a.mdot_star)

        # Assemble pressure correction matrix
        row, col, data = assemble_pressure_correction_matrix(self.mesh, self.rho)
        rhs_p = -compute_divergence_from_face_fluxes(self.mesh, a.mdot_star)

        # Solve pressure correction with scipy (handles nullspace internally)
        A_p = csr_matrix((data, (row, col)), shape=(self.n_cells, self.n_cells))
        p_prime, a.M_p = scipy_solver(
            A_p,
            rhs_p,
            M=a.M_p,
            tolerance=self.params.linear_solver_tol,
            remove_nullspace=True,
        )

        # Velocity and pressure corrections - reuse buffers
        compute_cell_gradients_structured(
            self.mesh, p_prime, use_limiter=False, out=a.grad_p_prime
        )
        velocity_correction(
            self.mesh, a.grad_p_prime, a.bold_D, u_prime=a.u_prime, v_prime=a.v_prime
        )

        # Update velocity and pressure (in-place operations into fresh buffers)
        np.add(u_star, a.u_prime, out=a.u)
        np.add(v_star, a.v_prime, out=a.v)
        a.p += self.params.alpha_p * p_prime

        # Update mass flux - reuse buffers
        interpolate_velocity_to_face(
            self.mesh, a.u_prime, a.v_prime, out=a.U_prime_face
        )
        mdot_calculation(self.mesh, self.rho, a.U_prime_face, out=a.mdot_prime)
        np.add(a.mdot_star, a.mdot_prime, out=a.mdot)

        # No copy needed! u and v now have new values, u_prev and v_prev have old values
        # Next iteration they will swap again

        return a.u, a.v, a.p

    def _compute_algebraic_residuals(self):
        """Return algebraic residuals for SIMPLE algorithm.

        For FV-SIMPLE:
        - Momentum residuals: velocity correction magnitudes ||u'||, ||v'||
        - Continuity residual: mass imbalance ||div(mdot)||
        """
        # Mass imbalance (continuity): divergence of corrected mass flux
        mass_imbalance = compute_divergence_from_face_fluxes(
            self.mesh, self.arrays.mdot
        )

        return {
            "u_residual": np.linalg.norm(self.arrays.u_prime),
            "v_residual": np.linalg.norm(self.arrays.v_prime),
            "continuity_residual": np.linalg.norm(mass_imbalance),
        }
