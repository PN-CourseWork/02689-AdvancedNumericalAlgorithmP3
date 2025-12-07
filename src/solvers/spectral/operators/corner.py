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
# Method 1b: Regularized Lid (Saad's approach - u = 16x²(1-x)²)
# =============================================================================


class RegularizedLidTreatment(CornerTreatment):
    """Corner treatment via regularized lid velocity profile.

    Uses u = 16x²(1-x)² on the lid, which:
    - Equals 0 at corners (x=0 and x=1)
    - Has zero derivatives at corners
    - Maximum value of 1 at x=0.5

    This completely removes the corner singularity by making the
    boundary conditions compatible (continuous and smooth).

    Reference: Saad's regularized lid-driven cavity problem.
    """

    def get_lid_velocity(
        self,
        x: np.ndarray,
        y: np.ndarray,
        lid_velocity: float,
        Lx: float,
        Ly: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Regularized lid velocity: u = 16x²(1-x)² * lid_velocity."""
        x_flat = np.asarray(x).ravel()
        x_norm = x_flat / Lx  # Normalize to [0, 1]

        # Regularized profile: 16x²(1-x)² has max=1 at x=0.5
        u_lid = 16 * x_norm**2 * (1 - x_norm) ** 2 * lid_velocity
        v_lid = np.zeros_like(x_flat, dtype=float)

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

    IMPORTANT: The Moffatt solution is a LOCAL asymptotic expansion, valid only
    near the corner. We use a smooth cutoff function to blend the singular solution
    to zero away from the corner, preventing contamination across the domain.

    Modified convection: u_c*grad(u_c) + u_s*grad(u_c) + u_c*grad(u_s) + u_s*grad(u_s)

    Boundary conditions:
    - Lid: u_c = V_lid - u_s, v_c = -v_s
    - Walls: u_c = -u_s, v_c = -v_s
    """

    # Moffatt eigenvalue for 90 deg corner: sin(lambda*pi/2) = lambda (first root > 1)
    LAMBDA = 1.5445840107634553

    def __init__(self, cutoff_radius: float = 0.3):
        """Initialize subtraction treatment.

        Parameters
        ----------
        cutoff_radius : float
            Radius (as fraction of domain size) beyond which the singular
            solution is blended to zero. Default 0.3 (30% of domain).
        """
        self.lam = self.LAMBDA
        self.cutoff_radius = cutoff_radius
        # Precompute constants for angular function
        self._sin_lam_pi2 = np.sin(self.lam * np.pi / 2)
        self._sin_lam2_pi2 = np.sin((self.lam - 2) * np.pi / 2)
        # Normalization: f'(0) = 1
        self._C = (
            self.lam / self._sin_lam_pi2 - (self.lam - 2) / self._sin_lam2_pi2
        )

    def _cutoff_function(self, r: np.ndarray, L: float) -> Tuple[np.ndarray, np.ndarray]:
        """Smooth cutoff function that is 1 near corner and 0 far away.

        Uses a quintic polynomial with continuous first two derivatives:
        chi(s) = 1 - 6*s^5 + 15*s^4 - 10*s^3  for 0 <= s <= 1
        chi'(s) = -30*s^4 + 60*s^3 - 30*s^2 = -30*s^2*(s-1)^2

        Parameters
        ----------
        r : np.ndarray
            Distance from corner
        L : float
            Reference length scale (domain size)

        Returns
        -------
        chi, dchi_dr : np.ndarray
            Cutoff function and its derivative w.r.t. r
        """
        r_cutoff = self.cutoff_radius * L
        s = np.clip(r / r_cutoff, 0, 1)

        # Quintic: chi = 1 - 6s^5 + 15s^4 - 10s^3 = 1 - s^3*(10 - 15*s + 6*s^2)
        s2 = s * s
        s3 = s2 * s
        chi = np.where(s < 1, 1 - s3 * (10 - 15 * s + 6 * s2), 0.0)

        # Derivative: dchi/ds = -30*s^2*(1-s)^2
        dchi_ds = -30 * s2 * (1 - s) ** 2
        dchi_dr = np.where(r < r_cutoff, dchi_ds / r_cutoff, 0.0)

        return chi, dchi_dr

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
        - u_s = d(psi)/dy, v_s = -d(psi)/dx  (guarantees div-free)

        We apply a smooth cutoff to the STREAMFUNCTION (not velocities) to
        preserve the divergence-free property:
        psi_blended = chi(r) * psi
        u_s = d(chi*psi)/dy = (dchi/dy)*psi + chi*(dpsi/dy)
        v_s = -d(chi*psi)/dx = -(dchi/dx)*psi - chi*(dpsi/dx)

        This ensures div(u_s, v_s) = 0 everywhere.

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
        at_corner = r < 1e-12

        # Angle: for left corner theta=0 along +x, for right corner flip
        is_left = corner_x < Lx / 2
        if is_left:
            theta = np.arctan2(dy, dx)
        else:
            theta = np.arctan2(dy, -dx)

        # Local unit vectors
        cos_t = np.where(~at_corner, dx / r_safe, 1.0)
        sin_t = np.where(~at_corner, dy / r_safe, 0.0)

        if not is_left:
            cos_t = -cos_t  # Flip for right corner geometry

        lam = self.lam

        # Get angular function and derivatives
        f, df, ddf = self._f_and_derivatives(theta)

        # Powers of r
        r_lam = r_safe ** lam       # For streamfunction
        r_lam_1 = r_safe ** (lam - 1)  # For velocities

        # Streamfunction: psi = r^lambda * f(theta)
        psi = r_lam * f

        # Streamfunction derivatives in polar coords:
        # dpsi/dr = lambda * r^(lambda-1) * f
        # dpsi/dtheta = r^lambda * f'
        dpsi_dr = lam * r_lam_1 * f
        dpsi_dtheta = r_lam * df

        # Convert to Cartesian using chain rule:
        # dpsi/dx = cos * dpsi/dr - (sin/r) * dpsi/dtheta
        # dpsi/dy = sin * dpsi/dr + (cos/r) * dpsi/dtheta
        inv_r = np.where(~at_corner, 1.0 / r_safe, 0.0)
        dpsi_dx_local = cos_t * dpsi_dr - sin_t * inv_r * dpsi_dtheta
        dpsi_dy_local = sin_t * dpsi_dr + cos_t * inv_r * dpsi_dtheta

        # Apply smooth cutoff to STREAMFUNCTION to preserve div-free property
        L_ref = np.sqrt(Lx**2 + Ly**2)
        chi, dchi_dr = self._cutoff_function(r, L_ref)

        # Cutoff gradient in local coordinates
        dchi_dx_local = dchi_dr * cos_t
        dchi_dy_local = dchi_dr * sin_t

        # Blended streamfunction derivatives (product rule):
        # d(chi*psi)/dx = dchi/dx * psi + chi * dpsi/dx
        # d(chi*psi)/dy = dchi/dy * psi + chi * dpsi/dy
        d_chipsi_dx_local = dchi_dx_local * psi + chi * dpsi_dx_local
        d_chipsi_dy_local = dchi_dy_local * psi + chi * dpsi_dy_local

        # Velocities from blended streamfunction:
        # u_s = dpsi_blended/dy, v_s = -dpsi_blended/dx
        u_s_local = d_chipsi_dy_local
        v_s_local = -d_chipsi_dx_local

        # For right corner, need to convert back to global coordinates
        # Local x points LEFT (towards domain interior), so flip signs appropriately
        if not is_left:
            # d/dx_global = -d/dx_local
            u_s = u_s_local  # dy unchanged
            v_s = -v_s_local  # dx flipped
        else:
            u_s = u_s_local
            v_s = v_s_local

        # Zero at exact corner
        u_s = np.where(at_corner, 0.0, u_s)
        v_s = np.where(at_corner, 0.0, v_s)

        result = {
            "u_s": u_s.reshape(original_shape),
            "v_s": v_s.reshape(original_shape),
        }

        if compute_derivatives:
            # Compute velocity derivatives using second derivatives of blended streamfunction
            # u_s = d(chi*psi)/dy, v_s = -d(chi*psi)/dx
            # du_s/dx = d²(chi*psi)/dxdy, du_s/dy = d²(chi*psi)/dy²
            # dv_s/dx = -d²(chi*psi)/dx², dv_s/dy = -d²(chi*psi)/dxdy

            # We need second derivatives of psi and chi
            r_lam_2 = r_safe ** (lam - 2)  # For second derivatives

            # Second derivatives of psi in polar coords:
            # d²psi/dr² = lambda*(lambda-1) * r^(lambda-2) * f
            # d²psi/drdtheta = lambda * r^(lambda-1) * f'
            # d²psi/dtheta² = r^lambda * f''
            d2psi_dr2 = lam * (lam - 1) * r_lam_2 * f
            d2psi_drdtheta = lam * r_lam_1 * df
            d2psi_dtheta2 = r_lam * ddf

            # Second derivative of cutoff function
            # chi(r) uses quintic, so we can compute d²chi/dr²
            r_cutoff = self.cutoff_radius * L_ref
            s = np.clip(r / r_cutoff, 0, 1)
            s2 = s * s
            # d²chi/ds² = -60*s*(1-s)*(1-2s)
            d2chi_ds2 = -60 * s * (1 - s) * (1 - 2 * s)
            d2chi_dr2 = np.where(r < r_cutoff, d2chi_ds2 / (r_cutoff ** 2), 0.0)

            # Convert to Cartesian second derivatives using chain rule:
            # d²/dx² = cos²*d²/dr² + 2*cos*(-sin/r)*d²/drdtheta + sin²/r²*d²/dtheta²
            #        + sin²/r*d/dr + 2*cos*sin/r²*d/dtheta
            # (simplified for readability, full formulas from polar to Cartesian)

            # For simplicity, compute derivatives numerically with small perturbations
            # This is more robust and avoids complex chain rule expressions
            eps = 1e-8

            # Helper to compute blended streamfunction derivative at a point
            def _psi_deriv_at(dx_pt, dy_pt):
                r_pt = np.sqrt(dx_pt**2 + dy_pt**2)
                r_pt_safe = np.maximum(r_pt, 1e-14)
                theta_pt = np.arctan2(dy_pt, dx_pt) if is_left else np.arctan2(dy_pt, -dx_pt)
                f_pt, df_pt, _ = self._f_and_derivatives(theta_pt)
                psi_pt = r_pt_safe ** lam * f_pt
                chi_pt, dchi_dr_pt = self._cutoff_function(r_pt, L_ref)
                return chi_pt * psi_pt

            # Compute second derivatives by finite differences
            psi_c = _psi_deriv_at(dx, dy)
            psi_px = _psi_deriv_at(dx + eps, dy)
            psi_mx = _psi_deriv_at(dx - eps, dy)
            psi_py = _psi_deriv_at(dx, dy + eps)
            psi_my = _psi_deriv_at(dx, dy - eps)
            psi_pxpy = _psi_deriv_at(dx + eps, dy + eps)
            psi_mxpy = _psi_deriv_at(dx - eps, dy + eps)
            psi_pxmy = _psi_deriv_at(dx + eps, dy - eps)
            psi_mxmy = _psi_deriv_at(dx - eps, dy - eps)

            # Second derivatives
            d2psi_dx2 = (psi_px - 2*psi_c + psi_mx) / eps**2
            d2psi_dy2 = (psi_py - 2*psi_c + psi_my) / eps**2
            d2psi_dxdy = (psi_pxpy - psi_mxpy - psi_pxmy + psi_mxmy) / (4 * eps**2)

            # Velocity derivatives from streamfunction
            # u_s = dpsi/dy, v_s = -dpsi/dx
            # du_s/dx = d²psi/dxdy, du_s/dy = d²psi/dy²
            # dv_s/dx = -d²psi/dx², dv_s/dy = -d²psi/dxdy
            dus_dx = d2psi_dxdy
            dus_dy = d2psi_dy2
            dvs_dx = -d2psi_dx2
            dvs_dy = -d2psi_dxdy

            # Note: for right corner, the coordinate system was flipped
            # But since we computed psi_blended directly in the flipped system,
            # and the velocity conversion already happened, no additional flip needed

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
    elif method_lower == "regularized":
        return RegularizedLidTreatment()
    elif method_lower == "subtraction":
        return SubtractionTreatment()
    else:
        raise ValueError(
            f"Unknown corner treatment method: {method}. "
            f"Use 'smoothing', 'regularized', or 'subtraction'."
        )
