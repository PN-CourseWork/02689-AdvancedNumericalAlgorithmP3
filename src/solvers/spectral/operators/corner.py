"""Corner singularity treatment for lid-driven cavity flow.

Two methods for handling corner singularities at the lid-wall junctions:

1. Smoothing method: Simple cosine smoothing of lid velocity near corners.
   - Easy to implement, works well
   - Approximation to the standard cavity problem

2. Subtraction method: Analytical singular solution subtraction.
   - Following Zhang & Xi (2010), Botella & Peyret (1998), Hancock et al. (1981)
   - Decomposes u = u_c + u_s where u_s is the Moffatt singular solution
   - Modified NS equations: u_c*nabla(u_c) + u_s*nabla(u_c) + u_c*nabla(u_s) + u_s*nabla(u_s)
                          = -nabla(p_c) + Re^(-1) * laplacian(u_c)
   - Requires analytical computation of u_s and its derivatives
"""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


# =============================================================================
# Abstract Base Class
# =============================================================================


class CornerTreatment(ABC):
    """Abstract base class for corner singularity treatment."""

    @abstractmethod
    def get_lid_velocity(
        self,
        x: np.ndarray,
        y: np.ndarray,
        lid_velocity: float,
        Lx: float,
        Ly: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (u, v) boundary condition on the lid (top boundary)."""
        pass

    @abstractmethod
    def get_wall_velocity(
        self,
        x: np.ndarray,
        y: np.ndarray,
        Lx: float,
        Ly: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (u, v) boundary condition on stationary walls."""
        pass

    def get_singular_velocity(
        self,
        x: np.ndarray,
        y: np.ndarray,
        Lx: float,
        Ly: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return singular velocity field (u_s, v_s). Zero for smoothing method."""
        shape = np.asarray(x).shape
        return np.zeros(shape), np.zeros(shape)

    def get_singular_velocity_derivatives(
        self,
        x: np.ndarray,
        y: np.ndarray,
        Lx: float,
        Ly: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return analytical derivatives (du_s/dx, du_s/dy, dv_s/dx, dv_s/dy).

        These must be computed analytically, NOT spectrally, because
        the singular solution has unbounded derivatives near corners.
        """
        shape = np.asarray(x).shape
        zeros = np.zeros(shape)
        return zeros, zeros, zeros, zeros

    def uses_modified_convection(self) -> bool:
        """Return True if this method requires modified convection terms."""
        return False


# =============================================================================
# Method 1: Smoothing (Simple, works easily)
# =============================================================================


class SmoothingTreatment(CornerTreatment):
    """Corner treatment via cosine smoothing of lid velocity.

    Simple approach that smoothly transitions the lid velocity from 0 at
    corners to full velocity away from corners. Avoids the discontinuity
    but does not remove the mathematical singularity.

    Parameters
    ----------
    smoothing_width : float
        Fraction of domain width to smooth at each corner (default: 0.15)
    """

    def __init__(self, smoothing_width: float = 0.15):
        self.smoothing_width = smoothing_width

    def get_lid_velocity(
        self,
        x: np.ndarray,
        y: np.ndarray,
        lid_velocity: float,
        Lx: float,
        Ly: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Lid velocity with cosine smoothing at corners."""
        x_flat = np.asarray(x).ravel()

        # Start with full lid velocity
        u_lid = np.full_like(x_flat, lid_velocity, dtype=float)
        v_lid = np.zeros_like(x_flat, dtype=float)

        if self.smoothing_width > 0:
            smooth_dist = self.smoothing_width * Lx

            # Smooth near left corner (x = 0)
            mask_left = x_flat < smooth_dist
            if np.any(mask_left):
                factor = 0.5 * (1 - np.cos(np.pi * x_flat[mask_left] / smooth_dist))
                u_lid[mask_left] = factor * lid_velocity

            # Smooth near right corner (x = Lx)
            mask_right = x_flat > (Lx - smooth_dist)
            if np.any(mask_right):
                factor = 0.5 * (
                    1 - np.cos(np.pi * (Lx - x_flat[mask_right]) / smooth_dist)
                )
                u_lid[mask_right] = factor * lid_velocity

        return u_lid.reshape(x.shape), v_lid.reshape(x.shape)

    def get_wall_velocity(
        self,
        x: np.ndarray,
        y: np.ndarray,
        Lx: float,
        Ly: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Stationary walls have zero velocity."""
        shape = np.asarray(x).shape
        return np.zeros(shape), np.zeros(shape)


# =============================================================================
# Method 2: Subtraction (Zhang & Xi / Botella & Peyret method)
# =============================================================================


class SubtractionTreatment(CornerTreatment):
    """Corner treatment via analytical singular solution subtraction.

    Following Zhang & Xi (2010), Botella & Peyret (1998):
    - Decompose: u = u_c + u_s, p = p_c + p_s
    - u_s is the Moffatt singular solution near corners
    - Solve modified NS for smooth part u_c

    The singular solution comes from the Stokes streamfunction:
    psi = r^lambda * f(theta) where lambda ~ 1.5446 for 90 deg corner

    Velocities: u_s ~ r^(lambda-1) (bounded), but grad(u_s) ~ r^(lambda-2) (singular)

    Modified convection: u_c*grad(u_c) + u_s*grad(u_c) + u_c*grad(u_s) + u_s*grad(u_s)

    Boundary conditions:
    - Lid: u_c = V_lid - u_s, v_c = -v_s
    - Walls: u_c = -u_s, v_c = -v_s
    """

    # Moffatt eigenvalue for 90 deg corner: sin(lambda*pi/2) = lambda (first root > 1)
    LAMBDA = 1.5445840107634553

    def __init__(self):
        self.lam = self.LAMBDA
        # Precompute constants for angular function
        self._sin_lam_pi2 = np.sin(self.lam * np.pi / 2)
        self._sin_lam2_pi2 = np.sin((self.lam - 2) * np.pi / 2)
        # Normalization: f'(0) = 1
        self._C = (
            self.lam / self._sin_lam_pi2 - (self.lam - 2) / self._sin_lam2_pi2
        )

    def _f_and_derivatives(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute angular function f(theta), f'(theta), and f''(theta).

        The Moffatt angular function satisfies:
        - f(0) = 0, f'(0) = 1 (unit lid velocity)
        - f(-pi/2) = 0, f'(-pi/2) = 0 (no-slip wall)

        f(theta) = [sin(lambda*theta)/sin(lambda*pi/2)
                   - sin((lambda-2)*theta)/sin((lambda-2)*pi/2)] / C

        Returns
        -------
        f, df, ddf : np.ndarray
            Angular function and its first two derivatives
        """
        lam = self.lam

        # f(theta)
        term1 = np.sin(lam * theta) / self._sin_lam_pi2
        term2 = np.sin((lam - 2) * theta) / self._sin_lam2_pi2
        f = (term1 - term2) / self._C

        # f'(theta)
        dterm1 = lam * np.cos(lam * theta) / self._sin_lam_pi2
        dterm2 = (lam - 2) * np.cos((lam - 2) * theta) / self._sin_lam2_pi2
        df = (dterm1 - dterm2) / self._C

        # f''(theta)
        ddterm1 = -lam**2 * np.sin(lam * theta) / self._sin_lam_pi2
        ddterm2 = -(lam - 2)**2 * np.sin((lam - 2) * theta) / self._sin_lam2_pi2
        ddf = (ddterm1 - ddterm2) / self._C

        return f, df, ddf

    def _compute_singular_solution(
        self,
        x: np.ndarray,
        y: np.ndarray,
        corner_x: float,
        corner_y: float,
        Lx: float,
        Ly: float,
        compute_derivatives: bool = False,
    ) -> dict:
        """Compute Moffatt singular solution and optionally its derivatives.

        For the streamfunction psi = r^lambda * f(theta):
        - Polar velocities: u_r = (1/r) * d_psi/d_theta, u_theta = -d_psi/d_r
        - Cartesian velocities: u = u_r*cos - u_theta*sin, v = u_r*sin + u_theta*cos

        Derivatives use chain rule:
        - d/dx = cos * d/dr - (sin/r) * d/d_theta
        - d/dy = sin * d/dr + (cos/r) * d/d_theta
        """
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        original_shape = x_arr.shape

        # Flatten for computation
        x_flat = x_arr.ravel()
        y_flat = y_arr.ravel()

        # Local coordinates relative to corner
        dx = x_flat - corner_x
        dy = y_flat - corner_y

        r = np.sqrt(dx**2 + dy**2)
        r_safe = np.maximum(r, 1e-14)

        # Angle: for left corner theta=0 along +x, for right corner flip
        is_left = corner_x < Lx / 2
        if is_left:
            theta = np.arctan2(dy, dx)
        else:
            theta = np.arctan2(dy, -dx)

        cos_t = np.where(r > 1e-14, dx / r_safe, 1.0)
        sin_t = np.where(r > 1e-14, dy / r_safe, 0.0)

        if not is_left:
            cos_t = -cos_t  # Flip for right corner geometry

        lam = self.lam

        # Get angular function and derivatives
        f, df, ddf = self._f_and_derivatives(theta)

        # Powers of r
        r_lam_1 = r_safe ** (lam - 1)  # For velocities
        r_lam_2 = r_safe ** (lam - 2)  # For velocity derivatives

        # Polar velocities: u_r = r^(lam-1) * f', u_theta = -lam * r^(lam-1) * f
        u_r = r_lam_1 * df
        u_theta = -lam * r_lam_1 * f

        # Convert to Cartesian (before any corner flip)
        u_s_local = u_r * cos_t - u_theta * sin_t
        v_s_local = u_r * sin_t + u_theta * cos_t

        # For right corner, flip u back
        if not is_left:
            u_s = -u_s_local
            v_s = v_s_local
        else:
            u_s = u_s_local
            v_s = v_s_local

        # Zero at exact corner
        at_corner = r < 1e-12
        u_s = np.where(at_corner, 0.0, u_s)
        v_s = np.where(at_corner, 0.0, v_s)

        result = {
            "u_s": u_s.reshape(original_shape),
            "v_s": v_s.reshape(original_shape),
        }

        if compute_derivatives:
            # Compute Cartesian derivatives using chain rule
            # d/dx = cos * d/dr - (sin/r) * d/d_theta
            # d/dy = sin * d/dr + (cos/r) * d/d_theta

            # Derivatives of polar velocities w.r.t. r and theta
            # u_r = r^(lam-1) * f'(theta)
            # du_r/dr = (lam-1) * r^(lam-2) * f'
            # du_r/d_theta = r^(lam-1) * f''
            du_r_dr = (lam - 1) * r_lam_2 * df
            du_r_dtheta = r_lam_1 * ddf

            # u_theta = -lam * r^(lam-1) * f(theta)
            # du_theta/dr = -lam * (lam-1) * r^(lam-2) * f
            # du_theta/d_theta = -lam * r^(lam-1) * f'
            du_theta_dr = -lam * (lam - 1) * r_lam_2 * f
            du_theta_dtheta = -lam * r_lam_1 * df

            # Cartesian u = u_r*cos - u_theta*sin
            # du/dr = du_r/dr * cos - du_theta/dr * sin
            # du/d_theta = du_r/d_theta * cos + u_r * (-sin) - du_theta/d_theta * sin - u_theta * cos
            #            = (du_r/d_theta - u_theta) * cos - (du_theta/d_theta + u_r) * sin
            du_dr = du_r_dr * cos_t - du_theta_dr * sin_t
            du_dtheta = (du_r_dtheta - u_theta) * cos_t - (du_theta_dtheta + u_r) * sin_t

            # Cartesian v = u_r*sin + u_theta*cos
            # dv/dr = du_r/dr * sin + du_theta/dr * cos
            # dv/d_theta = du_r/d_theta * sin + u_r * cos + du_theta/d_theta * cos - u_theta * sin
            #            = (du_r/d_theta + u_theta) * sin + (du_theta/d_theta + u_r) * cos
            # Wait, let me recalculate:
            # dv/d_theta = (du_r/d_theta)*sin + u_r*cos + (du_theta/d_theta)*cos + u_theta*(-sin)
            #            = (du_r/d_theta - u_theta)*sin + (du_theta/d_theta + u_r)*cos
            # Hmm, that's different. Let me be more careful.
            # v = u_r * sin_t + u_theta * cos_t
            # dv/d_theta = (d/d_theta)[u_r * sin_t + u_theta * cos_t]
            #            = du_r/d_theta * sin_t + u_r * cos_t + du_theta/d_theta * cos_t - u_theta * sin_t
            dv_dr = du_r_dr * sin_t + du_theta_dr * cos_t
            dv_dtheta = du_r_dtheta * sin_t + u_r * cos_t + du_theta_dtheta * cos_t - u_theta * sin_t

            # Now apply chain rule to get Cartesian derivatives
            # For left corner: dx/dr = cos_t, dy/dr = sin_t
            #                  dx/d_theta = -r*sin_t, dy/d_theta = r*cos_t
            # So: d/dx = cos_t * d/dr - (sin_t/r) * d/d_theta
            #     d/dy = sin_t * d/dr + (cos_t/r) * d/d_theta

            # Protect against division by zero near corner
            inv_r = np.where(r > 1e-12, 1.0 / r_safe, 0.0)

            dus_dx_local = cos_t * du_dr - sin_t * inv_r * du_dtheta
            dus_dy_local = sin_t * du_dr + cos_t * inv_r * du_dtheta
            dvs_dx_local = cos_t * dv_dr - sin_t * inv_r * dv_dtheta
            dvs_dy_local = sin_t * dv_dr + cos_t * inv_r * dv_dtheta

            # For right corner, need to handle sign flips
            # The local cos_t was flipped, so derivatives need adjustment
            if not is_left:
                # u was flipped: u_s = -u_s_local
                # So du_s/dx = -du_s_local/dx, but x is also flipped
                # Actually: d(-u)/d(-x) = du/dx, so sign cancels for dus_dx
                # For dus_dy: d(-u)/dy = -du/dy
                dus_dx = dus_dx_local  # Signs cancel
                dus_dy = -dus_dy_local
                dvs_dx = -dvs_dx_local  # dv/d(-x) = -dv/dx
                dvs_dy = dvs_dy_local
            else:
                dus_dx = dus_dx_local
                dus_dy = dus_dy_local
                dvs_dx = dvs_dx_local
                dvs_dy = dvs_dy_local

            # Zero at corner
            dus_dx = np.where(at_corner, 0.0, dus_dx)
            dus_dy = np.where(at_corner, 0.0, dus_dy)
            dvs_dx = np.where(at_corner, 0.0, dvs_dx)
            dvs_dy = np.where(at_corner, 0.0, dvs_dy)

            result["dus_dx"] = dus_dx.reshape(original_shape)
            result["dus_dy"] = dus_dy.reshape(original_shape)
            result["dvs_dx"] = dvs_dx.reshape(original_shape)
            result["dvs_dy"] = dvs_dy.reshape(original_shape)

        return result

    def get_singular_velocity(
        self,
        x: np.ndarray,
        y: np.ndarray,
        Lx: float,
        Ly: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute total singular velocity from both corners: u_s = u_s^A + u_s^B."""
        # Left corner (0, Ly)
        left = self._compute_singular_solution(x, y, 0.0, Ly, Lx, Ly)

        # Right corner (Lx, Ly)
        right = self._compute_singular_solution(x, y, Lx, Ly, Lx, Ly)

        u_s = left["u_s"] + right["u_s"]
        v_s = left["v_s"] + right["v_s"]

        return u_s, v_s

    def get_singular_velocity_derivatives(
        self,
        x: np.ndarray,
        y: np.ndarray,
        Lx: float,
        Ly: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute analytical derivatives of singular velocity."""
        # Left corner
        left = self._compute_singular_solution(
            x, y, 0.0, Ly, Lx, Ly, compute_derivatives=True
        )

        # Right corner
        right = self._compute_singular_solution(
            x, y, Lx, Ly, Lx, Ly, compute_derivatives=True
        )

        dus_dx = left["dus_dx"] + right["dus_dx"]
        dus_dy = left["dus_dy"] + right["dus_dy"]
        dvs_dx = left["dvs_dx"] + right["dvs_dx"]
        dvs_dy = left["dvs_dy"] + right["dvs_dy"]

        return dus_dx, dus_dy, dvs_dx, dvs_dy

    def get_lid_velocity(
        self,
        x: np.ndarray,
        y: np.ndarray,
        lid_velocity: float,
        Lx: float,
        Ly: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Lid BC: u_c = V_lid - u_s, v_c = -v_s (so total u = V_lid, v = 0)."""
        u_s, v_s = self.get_singular_velocity(x, y, Lx, Ly)
        u_lid = lid_velocity - u_s
        v_lid = -v_s
        return u_lid, v_lid

    def get_wall_velocity(
        self,
        x: np.ndarray,
        y: np.ndarray,
        Lx: float,
        Ly: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Wall BC: u_c = -u_s, v_c = -v_s (so total u = 0, v = 0)."""
        u_s, v_s = self.get_singular_velocity(x, y, Lx, Ly)
        return -u_s, -v_s

    def uses_modified_convection(self) -> bool:
        """Subtraction method requires modified convection terms."""
        return True


# =============================================================================
# Factory Function
# =============================================================================


def create_corner_treatment(
    method: str = "smoothing",
    smoothing_width: float = 0.15,
    **kwargs,
) -> CornerTreatment:
    """Create corner treatment handler from configuration.

    Parameters
    ----------
    method : str
        Treatment method: "smoothing" or "subtraction"
    smoothing_width : float
        Width parameter for smoothing method (fraction of domain)

    Returns
    -------
    CornerTreatment
        Configured corner treatment handler
    """
    method_lower = method.lower()

    if method_lower == "smoothing":
        return SmoothingTreatment(smoothing_width=smoothing_width)
    elif method_lower == "subtraction":
        return SubtractionTreatment()
    else:
        raise ValueError(
            f"Unknown corner treatment method: {method}. "
            f"Use 'smoothing' or 'subtraction'."
        )
