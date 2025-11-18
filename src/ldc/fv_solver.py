"""Finite volume solver for lid-driven cavity.

This module implements a collocated finite volume solver using
SIMPLE algorithm for pressure-velocity coupling.
"""

import numpy as np
from scipy.sparse import csr_matrix
from dataclasses import replace

from .base_solver import LidDrivenCavitySolver
from .datastructures import FVinfo, FVFields, TimeSeries, InternalFields

from fv.assembly.convection_diffusion_matrix import assemble_diffusion_convection_matrix
from fv.discretization.gradient.structured_gradient import compute_cell_gradients_structured
from fv.linear_solvers.scipy_solver import scipy_solver
from fv.assembly.rhie_chow import mdot_calculation, rhie_chow_velocity
from fv.assembly.pressure_correction_eq_assembly import assemble_pressure_correction_matrix
from fv.assembly.divergence import compute_divergence_from_face_fluxes
from fv.core.corrections import velocity_correction
from fv.core.helpers import bold_Dv_calculation, interpolate_to_face, relax_momentum_equation


class FVSolver(LidDrivenCavitySolver):
    """Finite volume solver for lid-driven cavity problem.

    This solver uses a collocated grid arrangement with Rhie-Chow interpolation
    for pressure-velocity coupling using the SIMPLE algorithm.

    Parameters
    ----------
    config : FVConfig
        Configuration with physics (Re, lid velocity, domain size) and
        FV-specific parameters (nx, ny, convection scheme, etc.).
    """

    # Make config class accessible via solver
    Config = FVinfo

    def __init__(self, **kwargs):
        """Initialize FV solver.

        Parameters
        ----------
        **kwargs
            Configuration parameters passed to FVConfig.
            Can also pass config=FVConfig(...) directly.
        """
        super().__init__(**kwargs)

        # Linear solver settings (same for both momentum and pressure)
        self.linear_solver_settings = {'type': 'bcgs', 'preconditioner': 'hypre', 'tolerance': 1e-6, 'max_iterations': 1000}

        # Cache commonly used values
        self.n_cells = self.mesh.cell_volumes.shape[0]

    def _solve_momentum_equation(self, component_idx, phi, grad_phi, phi_prev_iter, grad_p_component):
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

        Returns
        -------
        phi_star : ndarray
            Predicted velocity component
        A_diag : ndarray
            Diagonal of momentum matrix (needed for pressure correction)
        """
        # Assemble momentum equation
        row, col, data, b = assemble_diffusion_convection_matrix(
            self.mesh, self.fields.mdot, grad_phi, self.rho, self.mu,
            component_idx, phi=phi,
            scheme=self.config.convection_scheme, limiter=self.config.limiter
        )
        A = csr_matrix((data, (row, col)), shape=(self.n_cells, self.n_cells))
        A_diag = A.diagonal()
        rhs = b - grad_p_component * self.mesh.cell_volumes

        # Apply under-relaxation
        relaxed_A_diag, rhs = relax_momentum_equation(rhs, A_diag, phi_prev_iter, self.config.alpha_uv)
        A.setdiag(relaxed_A_diag)

        # Solve
        phi_star, *_ = scipy_solver(A, rhs, **self.linear_solver_settings)

        return phi_star, A_diag

    def step(self):
        """Perform one SIMPLE iteration.

        Returns
        -------
        u, v, p : np.ndarray
            Updated velocity and pressure fields
        """
        # Compute pressure gradient (no limiter for pressure)
        grad_p = compute_cell_gradients_structured(self.mesh, self.fields.p, use_limiter=False)
        grad_p_bar = interpolate_to_face(self.mesh, grad_p)

        # Compute velocity gradients (with limiter)
        grad_u = compute_cell_gradients_structured(self.mesh, self.fields.u, use_limiter=True)
        grad_v = compute_cell_gradients_structured(self.mesh, self.fields.v, use_limiter=True)

        # Solve momentum equations
        u_star, A_u_diag = self._solve_momentum_equation(0, self.fields.u, grad_u, self.fields.u_prev_iter, grad_p[:, 0])
        v_star, A_v_diag = self._solve_momentum_equation(1, self.fields.v, grad_v, self.fields.v_prev_iter, grad_p[:, 1])

        # Pressure correction
        bold_D = bold_Dv_calculation(self.mesh, A_u_diag, A_v_diag)
        bold_D_bar = interpolate_to_face(self.mesh, bold_D)

        U_star = np.column_stack([u_star, v_star])
        U_star_rc = rhie_chow_velocity(self.mesh, U_star, grad_p_bar, grad_p, bold_D_bar)

        mdot_star = mdot_calculation(self.mesh, self.rho, U_star_rc)

        row, col, data = assemble_pressure_correction_matrix(self.mesh, self.rho)
        A_p = csr_matrix((data, (row, col)), shape=(self.n_cells, self.n_cells))
        rhs_p = -compute_divergence_from_face_fluxes(self.mesh, mdot_star)

        p_prime, *_ = scipy_solver(A_p, rhs_p, remove_nullspace=True, **self.linear_solver_settings)

        # Velocity and pressure corrections
        grad_p_prime = compute_cell_gradients_structured(self.mesh, p_prime, use_limiter=False)
        U_prime = velocity_correction(self.mesh, grad_p_prime, bold_D)

        u_corrected = u_star + U_prime[:, 0]
        v_corrected = v_star + U_prime[:, 1]
        p_corrected = self.fields.p + self.config.alpha_p * p_prime

        # Update mass flux
        U_prime_face = interpolate_to_face(self.mesh, U_prime)
        mdot_prime = mdot_calculation(self.mesh, self.rho, U_prime_face)
        mdot_corrected = mdot_star + mdot_prime

        # Update solver state
        self.fields.u = u_corrected
        self.fields.v = v_corrected
        self.fields.p = p_corrected
        self.fields.u_prev_iter = u_corrected.copy()
        self.fields.v_prev_iter = v_corrected.copy()
        self.fields.mdot = mdot_corrected

        return self.fields.u, self.fields.v, self.fields.p

    def _create_output_dataclasses(self, residual_history, final_iter_count, is_converged):
        """Create FV-specific output dataclasses."""
        # Extract residuals using list comprehensions
        u_residuals = [r['u'] for r in residual_history]
        v_residuals = [r['v'] for r in residual_history]
        combined_residual = [max(r['u'], r['v']) for r in residual_history]

        fields = FVFields(
            u=self.fields.u,
            v=self.fields.v,
            p=self.fields.p,
            x=self.mesh.cell_centers[:, 0],
            y=self.mesh.cell_centers[:, 1],
            grid_points=self.mesh.cell_centers,
            mdot=self.fields.mdot,
        )

        time_series = TimeSeries(
            residual=combined_residual,
            u_residual=u_residuals,
            v_residual=v_residuals,
            continuity_residual=None,  # Can add this later if needed
        )

        # Update config with convergence info instead of duplicating all fields
        metadata = replace(
            self.config,
            iterations=final_iter_count,
            converged=is_converged,
            final_residual=combined_residual[-1] if combined_residual else float('inf'),
        )

        return fields, time_series, metadata
