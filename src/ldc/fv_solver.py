"""Finite volume solver for lid-driven cavity.

This module implements a collocated finite volume solver using
SIMPLE algorithm for pressure-velocity coupling.
"""

import numpy as np
from scipy.sparse import csr_matrix
from dataclasses import replace, dataclass

from .base_solver import LidDrivenCavitySolver
from .datastructures import FVinfo, FVFields, TimeSeries


@dataclass
class SolverState:
    """Current solution state."""
    u: np.ndarray
    v: np.ndarray
    p: np.ndarray
    mdot: np.ndarray


@dataclass
class PreviousIteration:
    """Previous iteration values for under-relaxation."""
    u: np.ndarray
    v: np.ndarray


@dataclass
class WorkBuffers:
    """Pre-allocated work buffers reused each iteration."""
    # Gradient buffers
    grad_p: np.ndarray
    grad_u: np.ndarray
    grad_v: np.ndarray
    grad_p_prime: np.ndarray

    # Face interpolation buffers
    grad_p_bar: np.ndarray
    bold_D: np.ndarray
    bold_D_bar: np.ndarray

    # Velocity and flux buffers
    U_star_rc: np.ndarray
    U_prime_face: np.ndarray
    u_prime: np.ndarray
    v_prime: np.ndarray
    mdot_star: np.ndarray
    mdot_prime: np.ndarray

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

        # Initialize solution state
        self.state = SolverState(
            u=np.zeros(n_cells),
            v=np.zeros(n_cells),
            p=np.zeros(n_cells),
            mdot=np.zeros(n_faces),
        )

        # Initialize previous iteration buffers
        self.prev = PreviousIteration(
            u=np.zeros(n_cells),
            v=np.zeros(n_cells),
        )

        # Pre-allocate work buffers
        self.buf = WorkBuffers(
            # Gradient buffers
            grad_p=np.zeros((n_cells, 2)),
            grad_u=np.zeros((n_cells, 2)),
            grad_v=np.zeros((n_cells, 2)),
            grad_p_prime=np.zeros((n_cells, 2)),
            # Face interpolation buffers
            grad_p_bar=np.zeros((n_faces, 2)),
            bold_D=np.zeros((n_cells, 2)),
            bold_D_bar=np.zeros((n_faces, 2)),
            # Velocity and flux buffers
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
        return self.state.u

    @u.setter
    def u(self, value):
        self.state.u = value

    @property
    def v(self):
        return self.state.v

    @v.setter
    def v(self, value):
        self.state.v = value

    @property
    def p(self):
        return self.state.p

    @p.setter
    def p(self, value):
        self.state.p = value

    @property
    def mdot(self):
        return self.state.mdot

    @mdot.setter
    def mdot(self, value):
        self.state.mdot = value

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
        # Swap buffers at start (zero-copy) - prev now has old values, state gets fresh buffer
        self.state.u, self.prev.u = self.prev.u, self.state.u
        self.state.v, self.prev.v = self.prev.v, self.state.v

        # Compute pressure gradient (no limiter for pressure) - reuse buffers
        compute_cell_gradients_structured(self.mesh, self.state.p, use_limiter=False, out=self.buf.grad_p)
        interpolate_to_face(self.mesh, self.buf.grad_p, out=self.buf.grad_p_bar)

        # Compute velocity gradients (with limiter) - reuse buffers
        compute_cell_gradients_structured(self.mesh, self.prev.u, use_limiter=True, out=self.buf.grad_u)
        compute_cell_gradients_structured(self.mesh, self.prev.v, use_limiter=True, out=self.buf.grad_v)

        # Solve momentum equations
        u_star, A_u_diag = self._solve_momentum_equation(0, self.prev.u, self.buf.grad_u, self.prev.u, self.buf.grad_p[:, 0])
        v_star, A_v_diag = self._solve_momentum_equation(1, self.prev.v, self.buf.grad_v, self.prev.v, self.buf.grad_p[:, 1])

        # Pressure correction - reuse buffers
        bold_Dv_calculation(self.mesh, A_u_diag, A_v_diag, out=self.buf.bold_D)
        interpolate_to_face(self.mesh, self.buf.bold_D, out=self.buf.bold_D_bar)

        rhie_chow_velocity(self.mesh, u_star, v_star, self.buf.grad_p_bar, self.buf.grad_p, self.buf.bold_D_bar, out=self.buf.U_star_rc)

        mdot_calculation(self.mesh, self.rho, self.buf.U_star_rc, out=self.buf.mdot_star)

        row, col, data = assemble_pressure_correction_matrix(self.mesh, self.rho)
        A_p = csr_matrix((data, (row, col)), shape=(self.n_cells, self.n_cells))
        rhs_p = -compute_divergence_from_face_fluxes(self.mesh, self.buf.mdot_star)

        p_prime, *_ = scipy_solver(A_p, rhs_p, remove_nullspace=True, **self.linear_solver_settings)

        # Velocity and pressure corrections - reuse buffers
        compute_cell_gradients_structured(self.mesh, p_prime, use_limiter=False, out=self.buf.grad_p_prime)
        velocity_correction(self.mesh, self.buf.grad_p_prime, self.buf.bold_D, u_prime=self.buf.u_prime, v_prime=self.buf.v_prime)

        # Update velocity and pressure (in-place operations into fresh buffers)
        np.add(u_star, self.buf.u_prime, out=self.state.u)
        np.add(v_star, self.buf.v_prime, out=self.state.v)
        self.state.p += self.config.alpha_p * p_prime

        # Update mass flux - reuse buffers
        interpolate_velocity_to_face(self.mesh, self.buf.u_prime, self.buf.v_prime, out=self.buf.U_prime_face)
        mdot_calculation(self.mesh, self.rho, self.buf.U_prime_face, out=self.buf.mdot_prime)
        np.add(self.buf.mdot_star, self.buf.mdot_prime, out=self.state.mdot)

        # No copy needed! state now has new values, prev has old values
        # Next iteration they will swap again

        return self.state.u, self.state.v, self.state.p

    def _create_output_dataclasses(self, residual_history, final_iter_count, is_converged):
        """Create FV-specific output dataclasses."""
        # Extract residuals using list comprehensions
        u_residuals = [r['u'] for r in residual_history]
        v_residuals = [r['v'] for r in residual_history]
        combined_residual = [max(r['u'], r['v']) for r in residual_history]

        fields = FVFields(
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
