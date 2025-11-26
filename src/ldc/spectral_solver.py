"""Spectral solver for lid-driven cavity using pseudospectral method.

This solver implements the PN-PN-2 method with:
- Velocities on full (Nx+1)×(Ny+1) Legendre-Gauss-Lobatto grid
- Pressure on reduced (Nx-1)×(Ny-1) inner grid
- Artificial compressibility for pressure-velocity coupling
- 4-stage RK4 explicit time stepping with adaptive CFL
"""

import numpy as np

from .base_solver import LidDrivenCavitySolver
from .datastructures import SpectralInfo, SpectralResultFields, SpectralSolverFields
from spectral.spectral import LegendreLobattoBasis, ChebyshevLobattoBasis


class SpectralSolver(LidDrivenCavitySolver):
    """Pseudospectral solver for lid-driven cavity problem.

    Uses explicit time-stepping with artificial compressibility to solve
    the incompressible Navier-Stokes equations on a Legendre-Gauss-Lobatto grid.

    Parameters
    ----------
    config : SpectralInfo
        Configuration with physics (Re, lid velocity, domain size) and
        spectral-specific parameters (Nx, Ny, CFL, beta_squared, etc.).
    """

    Config = SpectralInfo
    ResultFields = SpectralResultFields

    def __init__(self, **kwargs):
        """Initialize spectral solver.

        Parameters
        ----------
        **kwargs
            Configuration parameters passed to SpectralInfo.
            Supports basis_type='legendre' or basis_type='chebyshev'.
        """
        super().__init__(**kwargs)

        # Create spectral basis based on config
        if self.config.basis_type.lower() == "chebyshev":
            self.basis_x = ChebyshevLobattoBasis(domain=(0.0, self.config.Lx))
            self.basis_y = ChebyshevLobattoBasis(domain=(0.0, self.config.Ly))
            print(f"Using Chebyshev-Gauss-Lobatto basis (Zhang et al. 2010)")
        elif self.config.basis_type.lower() == "legendre":
            self.basis_x = LegendreLobattoBasis(domain=(0.0, self.config.Lx))
            self.basis_y = LegendreLobattoBasis(domain=(0.0, self.config.Ly))
            print(f"Using Legendre-Gauss-Lobatto basis")
        else:
            raise ValueError(f"Unknown basis_type: {self.config.basis_type}. Use 'legendre' or 'chebyshev'")

        # Setup grids
        self._setup_grids()

        # Build differentiation matrices
        self._build_diff_matrices()

        # Cache grid shapes
        self.shape_full = (self.config.Nx + 1, self.config.Ny + 1)
        self.shape_inner = (self.config.Nx - 1, self.config.Ny - 1)

        # Allocate solver arrays
        n_nodes_full = self.shape_full[0] * self.shape_full[1]
        n_nodes_inner = self.shape_inner[0] * self.shape_inner[1]
        self.arrays = SpectralSolverFields.allocate(n_nodes_full, n_nodes_inner)

        # Create persistent 2D views (avoid repeated reshaping)
        # These are VIEWS, not copies - modifications affect the underlying 1D arrays
        self.u_2d = self.arrays.u.reshape(self.shape_full)
        self.v_2d = self.arrays.v.reshape(self.shape_full)
        self.u_stage_2d = self.arrays.u_stage.reshape(self.shape_full)
        self.v_stage_2d = self.arrays.v_stage.reshape(self.shape_full)

        self.p_2d = self.arrays.p.reshape(self.shape_inner)
        self.dp_dx_inner_2d = self.arrays.dp_dx_inner.reshape(self.shape_inner)
        self.dp_dy_inner_2d = self.arrays.dp_dy_inner.reshape(self.shape_inner)

        # Initialize lid velocity with corner smoothing
        self._initialize_lid_velocity()

    def _setup_grids(self):
        """Setup full and reduced grids using Legendre-Gauss-Lobatto nodes."""
        # Full grid: (Nx+1) × (Ny+1)
        x_nodes = self.basis_x.nodes(self.config.Nx + 1)
        y_nodes = self.basis_y.nodes(self.config.Ny + 1)
        self.x_full, self.y_full = np.meshgrid(x_nodes, y_nodes, indexing='xy')

        # Reduced grid for pressure: (Nx-1) × (Ny-1) - interior points only
        self.x_inner = x_nodes[1:-1]
        self.y_inner = y_nodes[1:-1]
        self.x_reduced, self.y_reduced = np.meshgrid(self.x_inner, self.y_inner, indexing='xy')

        # Grid spacing (minimum) for CFL calculation
        self.dx_min = np.min(np.diff(x_nodes))
        self.dy_min = np.min(np.diff(y_nodes))

    def _apply_corner_smoothing(self, u_2d):
        """Apply corner smoothing to lid velocity on top boundary.

        Parameters
        ----------
        u_2d : np.ndarray
            2D velocity array on full grid (ny, nx) with indexing='xy', modified in place
        """
        if self.config.corner_smoothing > 0:
            Nx = self.config.Nx
            smooth_width = int(self.config.corner_smoothing * Nx)
            if smooth_width > 0:
                # Vectorized corner smoothing
                # With indexing='xy', shape is (ny, nx)
                # Top boundary is at u_2d[-1, :] (last row)
                # Smooth along x-direction at left and right corners
                i_values = np.arange(smooth_width)
                factors = 0.5 * (1 - np.cos(np.pi * i_values / smooth_width))

                # Left and right corners on top boundary
                u_2d[-1, i_values] = factors * self.config.lid_velocity
                u_2d[-1, Nx - i_values] = factors * self.config.lid_velocity

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
        Nx, Ny = self.config.Nx, self.config.Ny

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
        """Initialize lid velocity with corner smoothing to avoid spurious modes."""
        # Set top boundary (y = Ly) to lid velocity
        self.u_2d[:, -1] = self.config.lid_velocity

        # Apply corner smoothing using smooth transition
        self._apply_corner_smoothing(self.u_2d)

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
        - u, v on full (Nx+1) × (Ny+1) grid
        - p on inner (Nx-1) × (Ny-1) grid
        - R_u, R_v on full grid
        - R_p on inner grid

        Parameters
        ----------
        u, v : np.ndarray
            Current velocity fields on full grid
        p : np.ndarray
            Current pressure field on INNER grid

        Updates
        -------
        self.arrays.R_u, self.arrays.R_v (full grid), self.arrays.R_p (inner grid)
        """
        # Compute velocity derivatives on full grid
        self.arrays.du_dx[:] = self.Dx @ u
        self.arrays.du_dy[:] = self.Dy @ u
        self.arrays.dv_dx[:] = self.Dx @ v
        self.arrays.dv_dy[:] = self.Dy @ v

        # Compute Laplacians on full grid
        self.arrays.lap_u[:] = self.Laplacian @ u
        self.arrays.lap_v[:] = self.Laplacian @ v

        # Compute pressure gradient from inner grid p and interpolate to full grid
        self._interpolate_pressure_gradient()

        # Momentum residuals on FULL grid: R = -(u·∇)u - ∇p + (1/Re)∇²u
        conv_u = u * self.arrays.du_dx + v * self.arrays.du_dy
        conv_v = u * self.arrays.dv_dx + v * self.arrays.dv_dy

        nu = 1.0 / self.config.Re

        self.arrays.R_u[:] = -conv_u - self.arrays.dp_dx + nu * self.arrays.lap_u
        self.arrays.R_v[:] = -conv_v - self.arrays.dp_dy + nu * self.arrays.lap_v

        # Continuity residual on INNER grid: R_p = -β²(∂u/∂x + ∂v/∂y)
        # Compute divergence on full grid, then restrict to inner grid
        divergence_full = self.arrays.du_dx + self.arrays.dv_dy
        divergence_2d = divergence_full.reshape(self.shape_full)
        divergence_inner = divergence_2d[1:-1, 1:-1].ravel()

        # Pressure residual on inner grid
        self.arrays.R_p[:] = -self.config.beta_squared * divergence_inner

    def _enforce_boundary_conditions(self, u, v):
        """Enforce no-slip boundary conditions on all walls.

        Parameters
        ----------
        u, v : np.ndarray
            Velocity fields (1D flat arrays) to modify in place
        """
        # Create 2D views (cheap - just metadata)
        # With indexing='xy', shape is (ny, nx):
        #   First dimension = y (rows): 0=South, -1=North
        #   Second dimension = x (cols): 0=West, -1=East
        u_2d = u.reshape(self.shape_full)
        v_2d = v.reshape(self.shape_full)

        # No-slip on all boundaries
        u_2d[0, :] = 0.0   # South (bottom, y=0)
        u_2d[:, 0] = 0.0   # West (left, x=0)
        u_2d[:, -1] = 0.0  # East (right, x=max)
        v_2d[0, :] = 0.0   # South (bottom, y=0)
        v_2d[:, 0] = 0.0   # West (left, x=0)
        v_2d[:, -1] = 0.0  # East (right, x=max)
        v_2d[-1, :] = 0.0  # North (top, y=max)

        # Moving lid on north boundary (top, y=max)
        u_2d[-1, :] = self.config.lid_velocity
        self._apply_corner_smoothing(u_2d)

    def _compute_adaptive_timestep(self):
        """Compute adaptive pseudo-timestep based on CFL condition.

        Returns
        -------
        float
            Adaptive timestep ∆τ
        """
        # Maximum velocities (avoid division by zero)
        u_max = max(np.max(np.abs(self.arrays.u)), self.config.lid_velocity)
        v_max = max(np.max(np.abs(self.arrays.v)), 1e-10)

        # Wave speeds: λ_x and λ_y from equation (9)
        nu = 1.0 / self.config.Re
        lambda_x = (u_max + np.sqrt(u_max**2 + self.config.beta_squared)) / self.dx_min + nu / self.dx_min**2
        lambda_y = (v_max + np.sqrt(v_max**2 + self.config.beta_squared)) / self.dy_min + nu / self.dy_min**2

        return self.config.CFL / (lambda_x + lambda_y)

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
        rk4_coeffs = [0.25, 1.0/3.0, 0.5, 1.0]
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

    def _create_result_fields(self):
        """Create spectral-specific result fields with grid data.

        For PN-PN-2: Pressure lives on inner grid, so we interpolate to full grid
        for output/visualization purposes.
        """
        # Interpolate pressure from inner to full grid for output
        p_full_2d = self._extrapolate_to_full_grid(self.p_2d)

        return SpectralResultFields(
            u=self.arrays.u,
            v=self.arrays.v,
            p=p_full_2d.ravel(),
            x=self.x_full.ravel(),
            y=self.y_full.ravel(),
            grid_points=np.column_stack([self.x_full.ravel(), self.y_full.ravel()]),
            u_prev=self.arrays.u_prev,
            v_prev=self.arrays.v_prev,
        )

    def _compute_derivatives(self):
        """Compute velocity and pressure derivatives using spectral differentiation.

        Returns
        -------
        dict
            Dictionary with velocity/pressure derivatives and Laplacians.
        """
        # Compute velocity derivatives using spectral differentiation matrices
        du_dx = (self.Dx @ self.arrays.u)
        du_dy = (self.Dy @ self.arrays.u)
        dv_dx = (self.Dx @ self.arrays.v)
        dv_dy = (self.Dy @ self.arrays.v)

        # Compute velocity Laplacians
        lap_u = (self.Laplacian @ self.arrays.u)
        lap_v = (self.Laplacian @ self.arrays.v)

        # Compute pressure gradient and interpolate to full grid
        # Pressure lives on inner grid for PN-PN-2
        p_inner_2d = self.arrays.p.reshape(self.shape_inner)
        dp_dx_inner = (self.Dx_inner @ self.arrays.p)
        dp_dy_inner = (self.Dy_inner @ self.arrays.p)

        # Interpolate pressure gradient from inner to full grid
        dp_dx_inner_2d = dp_dx_inner.reshape(self.shape_inner)
        dp_dy_inner_2d = dp_dy_inner.reshape(self.shape_inner)

        dp_dx_full_2d = self._extrapolate_to_full_grid(dp_dx_inner_2d)
        dp_dy_full_2d = self._extrapolate_to_full_grid(dp_dy_inner_2d)

        return {
            'du_dx': du_dx,
            'du_dy': du_dy,
            'dv_dx': dv_dx,
            'dv_dy': dv_dy,
            'dp_dx': dp_dx_full_2d.ravel(),
            'dp_dy': dp_dy_full_2d.ravel(),
            'lap_u': lap_u,
            'lap_v': lap_v
        }
