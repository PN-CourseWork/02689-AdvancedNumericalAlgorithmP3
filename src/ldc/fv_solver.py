"""Finite volume solver for lid-driven cavity.

This module implements a collocated finite volume solver using
SIMPLE algorithm for pressure-velocity coupling.
"""

import numpy as np
from scipy.sparse import csr_matrix
from dataclasses import replace

from .base_solver import LidDrivenCavitySolver
from .datastructures import FVinfo, FVResultFields, TimeSeries, FVSolverFields

from fv.assembly.convection_diffusion_matrix import assemble_diffusion_convection_matrix
from fv.discretization.gradient.structured_gradient import compute_cell_gradients_structured
from fv.linear_solvers.scipy_solver import scipy_solver
from fv.assembly.rhie_chow import mdot_calculation, rhie_chow_velocity
from fv.assembly.pressure_correction_eq_assembly import assemble_pressure_correction_matrix
from fv.assembly.divergence import compute_divergence_from_face_fluxes
from fv.core.corrections import velocity_correction
from fv.core.helpers import bold_Dv_calculation, interpolate_to_face, interpolate_velocity_to_face, relax_momentum_equation


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

    # Constant fluid density
    rho = 1.0

    def __init__(self, **kwargs):
        """Initialize FV solver.

        Parameters
        ----------
        **kwargs
            Configuration parameters passed to FVConfig.
            Can also pass config=FVConfig(...) directly.
        """
        super().__init__(**kwargs)

        # Initialize fields
        n_cells = self.mesh.cell_volumes.shape[0]
        n_faces = self.mesh.internal_faces.shape[0] + self.mesh.boundary_faces.shape[0]

        # Compute dynamic viscosity from Reynolds number
        self.mu = self.rho * self.config.lid_velocity * self.config.Lx / self.config.Re

        # Initialize all solver arrays in a single dataclass
        self.arrays = FVSolverFields(
            # Current solution
            u=np.zeros(n_cells),
            v=np.zeros(n_cells),
            p=np.zeros(n_cells),
            mdot=np.zeros(n_faces),
            # Previous iteration
            u_prev=np.zeros(n_cells),
            v_prev=np.zeros(n_cells),
            # Gradient buffers
            grad_p=np.zeros((n_cells, 2)),
            grad_u=np.zeros((n_cells, 2)),
            grad_v=np.zeros((n_cells, 2)),
            grad_p_prime=np.zeros((n_cells, 2)),
            # Face interpolation buffers
            grad_p_bar=np.zeros((n_faces, 2)),
            bold_D=np.zeros((n_cells, 2)),
            bold_D_bar=np.zeros((n_faces, 2)),
            # Velocity and flux work buffers
            U_star_rc=np.zeros((n_faces, 2)),
            U_prime_face=np.zeros((n_faces, 2)),
            u_prime=np.zeros(n_cells),
            v_prime=np.zeros(n_cells),
            mdot_star=np.zeros(n_faces),
            mdot_prime=np.zeros(n_faces),
        )

        # Linear solver settings (same for both momentum and pressure)
        self.linear_solver_settings = {'type': 'bcgs', 'preconditioner': 'hypre', 'tolerance': 1e-6, 'max_iterations': 1000}

        # Cache commonly used values
        self.n_cells = n_cells

    def _initialize_fields(self):
        """Initialize fields - no-op since initialization happens in __init__."""
        pass

    # Properties for backward compatibility with base solver
    @property
    def u(self):
        return self.arrays.u

    @u.setter
    def u(self, value):
        self.arrays.u = value

    @property
    def v(self):
        return self.arrays.v

    @v.setter
    def v(self, value):
        self.arrays.v = value

    @property
    def p(self):
        return self.arrays.p

    @p.setter
    def p(self, value):
        self.arrays.p = value

    @property
    def mdot(self):
        return self.arrays.mdot

    @mdot.setter
    def mdot(self, value):
        self.arrays.mdot = value

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
            self.mesh, self.mdot, grad_phi, self.rho, self.mu,
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
        a = self.arrays  # Shorthand for readability

        # Swap buffers at start (zero-copy)
        a.u, a.u_prev = a.u_prev, a.u
        a.v, a.v_prev = a.v_prev, a.v

        # Compute pressure gradient (no limiter for pressure) - reuse buffers
        compute_cell_gradients_structured(self.mesh, a.p, use_limiter=False, out=a.grad_p)
        interpolate_to_face(self.mesh, a.grad_p, out=a.grad_p_bar)

        # Compute velocity gradients (with limiter) - reuse buffers
        compute_cell_gradients_structured(self.mesh, a.u_prev, use_limiter=True, out=a.grad_u)
        compute_cell_gradients_structured(self.mesh, a.v_prev, use_limiter=True, out=a.grad_v)

        # Solve momentum equations
        u_star, A_u_diag = self._solve_momentum_equation(0, a.u_prev, a.grad_u, a.u_prev, a.grad_p[:, 0])
        v_star, A_v_diag = self._solve_momentum_equation(1, a.v_prev, a.grad_v, a.v_prev, a.grad_p[:, 1])

        # Pressure correction - reuse buffers
        bold_Dv_calculation(self.mesh, A_u_diag, A_v_diag, out=a.bold_D)
        interpolate_to_face(self.mesh, a.bold_D, out=a.bold_D_bar)

        rhie_chow_velocity(self.mesh, u_star, v_star, a.grad_p_bar, a.grad_p, a.bold_D_bar, out=a.U_star_rc)

        mdot_calculation(self.mesh, self.rho, a.U_star_rc, out=a.mdot_star)

        row, col, data = assemble_pressure_correction_matrix(self.mesh, self.rho)
        A_p = csr_matrix((data, (row, col)), shape=(self.n_cells, self.n_cells))
        rhs_p = -compute_divergence_from_face_fluxes(self.mesh, a.mdot_star)

        p_prime, *_ = scipy_solver(A_p, rhs_p, remove_nullspace=True, **self.linear_solver_settings)

        # Velocity and pressure corrections - reuse buffers
        compute_cell_gradients_structured(self.mesh, p_prime, use_limiter=False, out=a.grad_p_prime)
        velocity_correction(self.mesh, a.grad_p_prime, a.bold_D, u_prime=a.u_prime, v_prime=a.v_prime)

        # Update velocity and pressure (in-place operations into fresh buffers)
        np.add(u_star, a.u_prime, out=a.u)
        np.add(v_star, a.v_prime, out=a.v)
        a.p += self.config.alpha_p * p_prime

        # Update mass flux - reuse buffers
        interpolate_velocity_to_face(self.mesh, a.u_prime, a.v_prime, out=a.U_prime_face)
        mdot_calculation(self.mesh, self.rho, a.U_prime_face, out=a.mdot_prime)
        np.add(a.mdot_star, a.mdot_prime, out=a.mdot)

        # No copy needed! u and v now have new values, u_prev and v_prev have old values
        # Next iteration they will swap again

        return a.u, a.v, a.p

    def _create_output_dataclasses(self, residual_history, final_iter_count, is_converged):
        """Create FV-specific output dataclasses."""
        # Extract residuals using list comprehensions
        u_residuals = [r['u'] for r in residual_history]
        v_residuals = [r['v'] for r in residual_history]
        combined_residual = [max(r['u'], r['v']) for r in residual_history]

        fields = FVResultFields(
            u=self.u,
            v=self.v,
            p=self.p,
            x=self.mesh.cell_centers[:, 0],
            y=self.mesh.cell_centers[:, 1],
            grid_points=self.mesh.cell_centers,
            mdot=self.mdot,
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
