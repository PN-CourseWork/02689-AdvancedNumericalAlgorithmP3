"""Single Grid (SG) Spectral solver for lid-driven cavity.

This is the base spectral solver using pseudospectral method without multigrid:
- Velocities on full (Nx+1)×(Ny+1) Legendre-Gauss-Lobatto grid
- Pressure on reduced (Nx-1)×(Ny-1) inner grid
- Artificial compressibility for pressure-velocity coupling
- 4-stage RK4 explicit time stepping with adaptive CFL

Corner singularity treatment options:
- "smoothing": Simple cosine smoothing of lid velocity near corners
- "subtraction": Analytical singular solution subtraction (Botella & Peyret 1998)
"""

import logging

import numpy as np

from ..base import LidDrivenCavitySolver
from ..datastructures import SpectralParameters, SpectralSolverFields
from solvers.spectral.basis.spectral import (
    LegendreLobattoBasis,
    ChebyshevLobattoBasis,
)
from solvers.spectral.operators.corner import create_corner_treatment

log = logging.getLogger(__name__)


class SGSolver(LidDrivenCavitySolver):
    """Single Grid Pseudospectral solver for lid-driven cavity problem.

    Uses explicit time-stepping with artificial compressibility to solve
    the incompressible Navier-Stokes equations on a Legendre-Gauss-Lobatto grid.

    This is the base spectral solver without multigrid acceleration.
    For multigrid variants, see FSGSolver, VMGSolver, FMGSolver.

    Parameters
    ----------
    params : SpectralParameters
        Parameters with physics (Re, lid velocity, domain size) and
        spectral-specific settings (Nx, Ny, CFL, beta_squared, etc.).
    """

    Parameters = SpectralParameters

    def __init__(self, **kwargs):
        """Initialize single grid spectral solver."""
        super().__init__(**kwargs)

        # Create spectral basis based on params
        if self.params.basis_type.lower() == "chebyshev":
            self.basis_x = ChebyshevLobattoBasis(domain=(0.0, self.params.Lx))
            self.basis_y = ChebyshevLobattoBasis(domain=(0.0, self.params.Ly))
            log.info("Using Chebyshev-Gauss-Lobatto basis")
        elif self.params.basis_type.lower() == "legendre":
            self.basis_x = LegendreLobattoBasis(domain=(0.0, self.params.Lx))
            self.basis_y = LegendreLobattoBasis(domain=(0.0, self.params.Ly))
            log.info("Using Legendre-Gauss-Lobatto basis")
        else:
            raise ValueError(
                f"Unknown basis_type: {self.params.basis_type}. Use 'legendre' or 'chebyshev'"
            )

        # Setup grids and differentiation matrices
        self._setup_grids()
        self._build_diff_matrices()

        # Cache grid shapes
        self.shape_full = (self.params.nx + 1, self.params.ny + 1)
        self.shape_inner = (self.params.nx - 1, self.params.ny - 1)

        # Allocate internal solver arrays
        n_nodes_full = self.shape_full[0] * self.shape_full[1]
        n_nodes_inner = self.shape_inner[0] * self.shape_inner[1]
        self.arrays = SpectralSolverFields.allocate(n_nodes_full, n_nodes_inner)

        # Initialize output fields (base class handles this)
        self._init_fields(x=self.x_full.ravel(), y=self.y_full.ravel())

        # Create persistent 2D views (modifications affect underlying 1D arrays)
        self.u_2d = self.arrays.u.reshape(self.shape_full)
        self.v_2d = self.arrays.v.reshape(self.shape_full)
        self.u_stage_2d = self.arrays.u_stage.reshape(self.shape_full)
        self.v_stage_2d = self.arrays.v_stage.reshape(self.shape_full)
        self.p_2d = self.arrays.p.reshape(self.shape_inner)
        self.dp_dx_inner_2d = self.arrays.dp_dx_inner.reshape(self.shape_inner)
        self.dp_dy_inner_2d = self.arrays.dp_dy_inner.reshape(self.shape_inner)

        # Create corner treatment handler
        self.corner_treatment = create_corner_treatment(
            method=self.params.corner_treatment,
            smoothing_width=self.params.corner_smoothing,
        )
        log.info(f"Using corner treatment: {self.params.corner_treatment}")

        # Cache singular velocity and derivatives if using subtraction method
        self._uses_modified_convection = (
            self.corner_treatment.uses_modified_convection()
        )
        if self._uses_modified_convection:
            log.info("Subtraction method: caching singular velocity and derivatives")
            # Get singular velocity u_s, v_s at all grid points
            u_s_2d, v_s_2d = self.corner_treatment.get_singular_velocity(
                self.x_full, self.y_full, self.params.Lx, self.params.Ly
            )
            self._u_s = u_s_2d.ravel()
            self._v_s = v_s_2d.ravel()

            # Get analytical derivatives (NOT spectral!)
            dus_dx, dus_dy, dvs_dx, dvs_dy = (
                self.corner_treatment.get_singular_velocity_derivatives(
                    self.x_full, self.y_full, self.params.Lx, self.params.Ly
                )
            )
            self._dus_dx = dus_dx.ravel()
            self._dus_dy = dus_dy.ravel()
            self._dvs_dx = dvs_dx.ravel()
            self._dvs_dy = dvs_dy.ravel()

        # Initialize lid velocity with corner treatment
        self._initialize_lid_velocity()

    def _setup_grids(self):
        """Setup full and reduced grids using Legendre-Gauss-Lobatto nodes."""
        # Full grid: (Nx+1) × (Ny+1)
        x_nodes = self.basis_x.nodes(self.params.nx + 1)
        y_nodes = self.basis_y.nodes(self.params.ny + 1)
        self.x_full, self.y_full = np.meshgrid(x_nodes, y_nodes, indexing="ij")

        # Reduced grid for pressure: (Nx-1) × (Ny-1) - interior points only
        self.x_inner = x_nodes[1:-1]
        self.y_inner = y_nodes[1:-1]
        self.x_reduced, self.y_reduced = np.meshgrid(
            self.x_inner, self.y_inner, indexing="ij"
        )

        # Grid spacing (minimum) for CFL calculation
        self.dx_min = np.min(np.diff(x_nodes))
        self.dy_min = np.min(np.diff(y_nodes))

    def _apply_lid_boundary(self, u_2d, v_2d):
        """Apply lid boundary condition using corner treatment.

        Parameters
        ----------
        u_2d, v_2d : np.ndarray
            2D velocity arrays on full grid (Nx+1, Ny+1), modified in place
        """
        # Get lid velocity from corner treatment
        x_lid = self.x_full[:, -1]  # x coordinates on top boundary
        y_lid = self.y_full[:, -1]  # y coordinates on top boundary

        u_lid, v_lid = self.corner_treatment.get_lid_velocity(
            x_lid,
            y_lid,
            lid_velocity=self.params.lid_velocity,
            Lx=self.params.Lx,
            Ly=self.params.Ly,
        )

        u_2d[:, -1] = u_lid
        v_2d[:, -1] = v_lid

    def _extrapolate_to_full_grid(self, inner_2d):
        """Extrapolate field from inner grid (Nx-1, Ny-1) to full grid (Nx+1, Ny+1).

        Uses linear extrapolation to boundaries and averaging for corners.

        Parameters
        ----------
        inner_2d : np.ndarray
            Field on inner grid, shape (Nx-1, Ny-1)

        Returns
        -------
        full_2d : np.ndarray
            Field on full grid, shape (Nx+1, Ny+1)
        """
        full_2d = np.zeros(self.shape_full)

        # Copy interior values
        full_2d[1:-1, 1:-1] = inner_2d

        # Extrapolate to boundaries (linear extrapolation)
        # West/East boundaries
        full_2d[0, 1:-1] = 2 * full_2d[1, 1:-1] - full_2d[2, 1:-1]
        full_2d[-1, 1:-1] = 2 * full_2d[-2, 1:-1] - full_2d[-3, 1:-1]

        # South/North boundaries
        full_2d[1:-1, 0] = 2 * full_2d[1:-1, 1] - full_2d[1:-1, 2]
        full_2d[1:-1, -1] = 2 * full_2d[1:-1, -2] - full_2d[1:-1, -3]

        # Corners (average of neighbors)
        full_2d[0, 0] = 0.5 * (full_2d[0, 1] + full_2d[1, 0])
        full_2d[0, -1] = 0.5 * (full_2d[0, -2] + full_2d[1, -1])
        full_2d[-1, 0] = 0.5 * (full_2d[-1, 1] + full_2d[-2, 0])
        full_2d[-1, -1] = 0.5 * (full_2d[-1, -2] + full_2d[-2, -1])

        return full_2d

    def _build_diff_matrices(self):
        """Build spectral differentiation matrices using tensor products."""
        Nx, Ny = self.params.nx, self.params.ny

        # 1D differentiation matrices on full grid
        x_nodes_full = self.basis_x.nodes(Nx + 1)
        y_nodes_full = self.basis_y.nodes(Ny + 1)
        Dx_1d = self.basis_x.diff_matrix(x_nodes_full)  # (Nx+1) × (Nx+1)
        Dy_1d = self.basis_y.diff_matrix(y_nodes_full)  # (Ny+1) × (Ny+1)

        # 2D differentiation via Kronecker products
        # For meshgrid with indexing='ij': first index is x, second is y
        Ix = np.eye(Nx + 1)
        Iy = np.eye(Ny + 1)
        self.Dx = np.kron(Dx_1d, Iy)  # d/dx on full grid
        self.Dy = np.kron(Ix, Dy_1d)  # d/dy on full grid

        # Laplacian: ∇² = ∂²/∂x² + ∂²/∂y²
        Dxx_1d = Dx_1d @ Dx_1d
        Dyy_1d = Dy_1d @ Dy_1d
        self.Dxx = np.kron(Dxx_1d, Iy)
        self.Dyy = np.kron(Ix, Dyy_1d)
        self.Laplacian = self.Dxx + self.Dyy

        # 1D differentiation matrices on reduced grid (for pressure)
        Dx_inner_1d = self.basis_x.diff_matrix(self.x_inner)  # (Nx-1) × (Nx-1)
        Dy_inner_1d = self.basis_y.diff_matrix(self.y_inner)  # (Ny-1) × (Ny-1)

        # 2D differentiation on reduced grid
        Ix_inner = np.eye(Nx - 1)
        Iy_inner = np.eye(Ny - 1)
        self.Dx_inner = np.kron(Dx_inner_1d, Iy_inner)
        self.Dy_inner = np.kron(Ix_inner, Dy_inner_1d)

    def _initialize_lid_velocity(self):
        """Initialize lid velocity using corner treatment."""
        # Apply lid boundary condition using corner treatment handler
        self._apply_lid_boundary(self.u_2d, self.v_2d)

    def _interpolate_pressure_gradient(self):
        """Compute pressure gradient on inner grid and interpolate to full grid.

        PN-PN-2 method:
        1. Pressure p exists on (Nx-1) × (Ny-1) inner grid
        2. Compute ∂p/∂x and ∂p/∂y on inner grid using inner diff matrices
        3. Extrapolate gradients to boundaries on full grid
        """
        # Compute pressure gradient on inner grid (this is where pressure actually lives!)
        self.arrays.dp_dx_inner[:] = self.Dx_inner @ self.arrays.p
        self.arrays.dp_dy_inner[:] = self.Dy_inner @ self.arrays.p

        # Extrapolate to full grid (using 2D views)
        dp_dx_2d = self._extrapolate_to_full_grid(self.dp_dx_inner_2d)
        dp_dy_2d = self._extrapolate_to_full_grid(self.dp_dy_inner_2d)

        # Store flattened on full grid
        self.arrays.dp_dx[:] = dp_dx_2d.ravel()
        self.arrays.dp_dy[:] = dp_dy_2d.ravel()

    def _compute_residuals(self, u, v, p):
        """Compute RHS residuals for pseudo time-stepping.

        PN-PN-2 method:
        - u, v on full (Nx+1) × (Ny+1) grid (these are u_c, v_c for subtraction method)
        - p on inner (Nx-1) × (Ny-1) grid
        - R_u, R_v on full grid
        - R_p on inner grid

        For subtraction method (Zhang & Xi 2010):
        Modified convection: u_c·∇u_c + u_s·∇u_c + u_c·∇u_s + u_s·∇u_s

        Parameters
        ----------
        u, v : np.ndarray
            Current velocity fields on full grid (u_c, v_c for subtraction)
        p : np.ndarray
            Current pressure field on INNER grid

        Updates
        -------
        self.arrays.R_u, self.arrays.R_v (full grid), self.arrays.R_p (inner grid)
        """
        # Compute velocity derivatives on full grid (spectral differentiation of u_c)
        self.arrays.du_dx[:] = self.Dx @ u
        self.arrays.du_dy[:] = self.Dy @ u
        self.arrays.dv_dx[:] = self.Dx @ v
        self.arrays.dv_dy[:] = self.Dy @ v

        # Compute Laplacians on full grid
        self.arrays.lap_u[:] = self.Laplacian @ u
        self.arrays.lap_v[:] = self.Laplacian @ v

        # Compute pressure gradient from inner grid p and interpolate to full grid
        self._interpolate_pressure_gradient()

        # Compute convection terms
        if self._uses_modified_convection:
            # Subtraction method: u_c·∇u_c + u_s·∇u_c + u_c·∇u_s + u_s·∇u_s
            # Term 1: u_c·∇u_c (standard convection of computational velocity)
            conv_u = u * self.arrays.du_dx + v * self.arrays.du_dy
            conv_v = u * self.arrays.dv_dx + v * self.arrays.dv_dy

            # Term 2: u_s·∇u_c (singular velocity advecting computational gradient)
            conv_u += self._u_s * self.arrays.du_dx + self._v_s * self.arrays.du_dy
            conv_v += self._u_s * self.arrays.dv_dx + self._v_s * self.arrays.dv_dy

            # Term 3: u_c·∇u_s (computational velocity advecting singular gradient)
            conv_u += u * self._dus_dx + v * self._dus_dy
            conv_v += u * self._dvs_dx + v * self._dvs_dy

            # Term 4: u_s·∇u_s (singular velocity advecting singular gradient)
            conv_u += self._u_s * self._dus_dx + self._v_s * self._dus_dy
            conv_v += self._u_s * self._dvs_dx + self._v_s * self._dvs_dy
        else:
            # Standard convection: (u·∇)u
            conv_u = u * self.arrays.du_dx + v * self.arrays.du_dy
            conv_v = u * self.arrays.dv_dx + v * self.arrays.dv_dy

        nu = 1.0 / self.params.Re

        self.arrays.R_u[:] = -conv_u - self.arrays.dp_dx + nu * self.arrays.lap_u
        self.arrays.R_v[:] = -conv_v - self.arrays.dp_dy + nu * self.arrays.lap_v

        # Continuity residual on INNER grid: R_p = -β²(∂u/∂x + ∂v/∂y)
        # Compute divergence on full grid, then restrict to inner grid
        divergence_full = self.arrays.du_dx + self.arrays.dv_dy
        divergence_2d = divergence_full.reshape(self.shape_full)
        divergence_inner = divergence_2d[1:-1, 1:-1].ravel()

        # Pressure residual on inner grid
        self.arrays.R_p[:] = -self.params.beta_squared * divergence_inner

    def _enforce_boundary_conditions(self, u, v):
        """Enforce boundary conditions on all walls using corner treatment.

        For smoothing method: No-slip on walls, smoothed lid velocity on top.
        For subtraction method: u_c = -u_s on walls, u_c = V_lid - u_s on top.

        Parameters
        ----------
        u, v : np.ndarray
            Velocity fields (1D flat arrays) to modify in place
        """
        # Create 2D views (cheap - just metadata)
        u_2d = u.reshape(self.shape_full)
        v_2d = v.reshape(self.shape_full)

        # Get wall velocities from corner treatment (0 for smoothing, -u_s for subtraction)
        # West boundary
        u_wall, v_wall = self.corner_treatment.get_wall_velocity(
            self.x_full[0, :], self.y_full[0, :], self.params.Lx, self.params.Ly
        )
        u_2d[0, :] = u_wall
        v_2d[0, :] = v_wall

        # East boundary
        u_wall, v_wall = self.corner_treatment.get_wall_velocity(
            self.x_full[-1, :], self.y_full[-1, :], self.params.Lx, self.params.Ly
        )
        u_2d[-1, :] = u_wall
        v_2d[-1, :] = v_wall

        # South boundary
        u_wall, v_wall = self.corner_treatment.get_wall_velocity(
            self.x_full[:, 0], self.y_full[:, 0], self.params.Lx, self.params.Ly
        )
        u_2d[:, 0] = u_wall
        v_2d[:, 0] = v_wall

        # North boundary (moving lid)
        self._apply_lid_boundary(u_2d, v_2d)

    def _compute_adaptive_timestep(self):
        """Compute adaptive pseudo-timestep based on CFL condition.

        Returns
        -------
        float
            Adaptive timestep ∆τ
        """
        # Maximum velocities (avoid division by zero)
        u_max = max(np.max(np.abs(self.arrays.u)), self.params.lid_velocity)
        v_max = max(np.max(np.abs(self.arrays.v)), 1e-10)

        # Wave speeds: λ_x and λ_y from equation (9)
        nu = 1.0 / self.params.Re
        lambda_x = (
            u_max + np.sqrt(u_max**2 + self.params.beta_squared)
        ) / self.dx_min + nu / self.dx_min**2
        lambda_y = (
            v_max + np.sqrt(v_max**2 + self.params.beta_squared)
        ) / self.dy_min + nu / self.dy_min**2

        return self.params.CFL / (lambda_x + lambda_y)

    def step(self):
        """Perform one RK4 pseudo time-step.

        PN-PN-2: Updates u, v on full grid and p on inner grid.

        Returns
        -------
        u, v, p : np.ndarray
            Updated velocities (full grid) and pressure (inner grid)
        """
        a = self.arrays  # Shorthand

        # Swap buffers at start (for residual calculation in solve())
        a.u, a.u_prev = a.u_prev, a.u
        a.v, a.v_prev = a.v_prev, a.v

        # Compute adaptive timestep
        dt = self._compute_adaptive_timestep()

        # 4-stage RK4: φ^(i) = φ^n + α_i·∆τ·R(φ^(i-1))
        rk4_coeffs = [0.25, 1.0 / 3.0, 0.5, 1.0]
        u_in, v_in, p_in = a.u, a.v, a.p

        for i, alpha in enumerate(rk4_coeffs):
            self._compute_residuals(u_in, v_in, p_in)

            # Last stage: write to final arrays; otherwise use staging arrays
            if i < 3:
                a.u_stage[:] = a.u + alpha * dt * a.R_u
                a.v_stage[:] = a.v + alpha * dt * a.R_v
                a.p_stage[:] = a.p + alpha * dt * a.R_p
                self._enforce_boundary_conditions(a.u_stage, a.v_stage)
                u_in, v_in, p_in = a.u_stage, a.v_stage, a.p_stage
            else:
                a.u[:] = a.u + alpha * dt * a.R_u
                a.v[:] = a.v + alpha * dt * a.R_v
                a.p[:] = a.p + alpha * dt * a.R_p
                self._enforce_boundary_conditions(a.u, a.v)

        return a.u, a.v, a.p

    def _finalize_fields(self):
        """Copy final solution to output fields.

        Override base class because PN-PN-2 pressure lives on inner grid
        and needs interpolation to full grid for output.
        """
        self.fields.u[:] = self.arrays.u
        self.fields.v[:] = self.arrays.v
        # Interpolate pressure from inner to full grid
        p_full_2d = self._extrapolate_to_full_grid(self.p_2d)
        self.fields.p[:] = p_full_2d.ravel()

    def _compute_algebraic_residuals(self):
        """Return algebraic residuals from pseudo time-stepping.

        For spectral solver, the algebraic residuals are the RHS of the
        time-stepping equations (R_u, R_v, R_p) computed during step().
        """
        return {
            "u_residual": np.linalg.norm(self.arrays.R_u),
            "v_residual": np.linalg.norm(self.arrays.R_v),
            "continuity_residual": np.linalg.norm(self.arrays.R_p),
        }

    def _compute_vorticity_for_export(
        self, U_2d: np.ndarray, V_2d: np.ndarray, x: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """Compute vorticity using spectral differentiation.

        Override base class to use spectral differentiation matrices
        for higher accuracy.

        Parameters
        ----------
        U_2d, V_2d : np.ndarray
            2D velocity arrays (ny, nx) - note: different from internal (nx+1, ny+1)
        x, y : np.ndarray
            1D coordinate arrays

        Returns
        -------
        np.ndarray
            Vorticity field (ny, nx)
        """
        # Use internal spectral differentiation on the full grid arrays
        # The fields are already finalized in self.arrays
        dv_dx = self.Dx @ self.arrays.v
        du_dy = self.Dy @ self.arrays.u
        vorticity = dv_dx - du_dy

        # Reshape to match the expected output (ny, nx) from VTK grid ordering
        # Internal shape is (Nx+1, Ny+1), but VTK uses (Ny+1, Nx+1) ordering
        vort_2d = vorticity.reshape(self.shape_full)  # (Nx+1, Ny+1)
        return vort_2d.T  # Transpose to (Ny+1, Nx+1) for VTK

    def _compute_vorticity(self) -> np.ndarray:
        """Compute vorticity using spectral differentiation.

        Override base class finite difference implementation.
        """
        dv_dx = self.Dx @ self.arrays.v
        du_dy = self.Dy @ self.arrays.u
        return dv_dx - du_dy

    def _compute_gradient(
        self, field: np.ndarray, bc_walls: float = 0.0, bc_lid: float = None
    ) -> tuple:
        """Compute gradient using spectral differentiation.

        Override base class finite difference implementation.
        BC parameters are ignored since spectral methods handle BCs through
        the differentiation matrices and boundary point values.
        """
        df_dx = self.Dx @ field
        df_dy = self.Dy @ field
        return df_dx, df_dy

    def _compute_quadrature_weights(self) -> np.ndarray:
        """Compute 2D quadrature weights for integration on Gauss-Lobatto grid.

        Uses trapezoidal rule weights based on non-uniform node spacing.
        Returns weights as 1D array matching self.arrays.u ordering.
        """
        # Get 1D nodes
        x_nodes = self.basis_x.nodes(self.params.nx + 1)
        y_nodes = self.basis_y.nodes(self.params.ny + 1)

        # Compute 1D trapezoidal weights
        def trapezoidal_weights(nodes):
            n = len(nodes)
            w = np.zeros(n)
            for i in range(1, n - 1):
                w[i] = 0.5 * (nodes[i + 1] - nodes[i - 1])
            w[0] = 0.5 * (nodes[1] - nodes[0])
            w[-1] = 0.5 * (nodes[-1] - nodes[-2])
            return w

        wx = trapezoidal_weights(x_nodes)
        wy = trapezoidal_weights(y_nodes)

        # 2D weights via outer product, then flatten to match array ordering
        # shape_full = (nx+1, ny+1) with indexing='ij', so W[i,j] = wx[i] * wy[j]
        W_2d = np.outer(wx, wy)
        return W_2d.ravel()

    def _compute_energy(self) -> float:
        """Compute kinetic energy using spectral quadrature: E = 0.5 * ∫(u² + v²) dA."""
        W = self._compute_quadrature_weights()
        return 0.5 * float(np.sum(W * (self.arrays.u**2 + self.arrays.v**2)))

    def _compute_enstrophy(self) -> float:
        """Compute enstrophy using spectral quadrature: Z = 0.5 * ∫ω² dA."""
        omega = self._compute_vorticity()
        W = self._compute_quadrature_weights()
        return 0.5 * float(np.sum(W * omega**2))

    def _compute_palinstrophy(self) -> float:
        """Compute palinstrophy using spectral quadrature: P = 0.5 * ∫||∇ω||² dA."""
        omega = self._compute_vorticity()
        domega_dx, domega_dy = self._compute_gradient(omega)
        W = self._compute_quadrature_weights()
        return 0.5 * float(np.sum(W * (domega_dx**2 + domega_dy**2)))
