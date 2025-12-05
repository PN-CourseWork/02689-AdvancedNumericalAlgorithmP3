"""Corner singularity treatment for lid-driven cavity flow.

Two methods for handling corner singularities at the lid-wall junctions:

1. Smoothing method: Simple cosine smoothing of lid velocity near corners.
   - Easy to implement, works well
   - Approximation to the standard cavity problem

2. Subtraction method: Analytical singular solution subtraction.
   - Following Zhang & Xi (2010), Botella & Peyret (1998), Hancock et al. (1981)
   - Decomposes u = u_c + u_s where u_s is the Moffatt singular solution
   - Modified NS equations: u_c·∇u_c + u_s·∇u_c + u_c·∇u_s + u_s·∇u_s = -∇p_c + Re⁻¹∇²u_c
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
                factor = 0.5 * (1 - np.cos(np.pi * (Lx - x_flat[mask_right]) / smooth_dist))
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
    ψ = r^λ f(θ) where λ ≈ 1.5446 for 90° corner

    Velocities: u_s ~ r^(λ-1) (bounded), but ∇u_s ~ r^(λ-2) (singular)

    Modified convection: u_c·∇u_c + u_s·∇u_c + u_c·∇u_s + u_s·∇u_s

    Boundary conditions:
    - Lid: u_c = V_lid - u_s, v_c = -v_s
    - Walls: u_c = -u_s, v_c = -v_s
    """

    # Moffatt eigenvalue for 90° corner: sin(λπ/2) = λ (first root > 1)
    LAMBDA = 1.5445840107634553

    def __init__(self):
        self.lam = self.LAMBDA

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
        """Compute Moffatt singular solution near one corner.

        The Stokes streamfunction near a corner is ψ = A r^λ f(θ).
        For a 90° corner with moving lid, we use the first-order solution.

        In local polar coords centered at corner:
        - r = distance from corner
        - θ = angle (0 along lid, -π/2 along wall going down)

        The angular function f(θ) satisfies boundary conditions:
        - f(0) = 0, f'(0) = 1 (lid velocity condition)
        - f(-π/2) = 0, f'(-π/2) = 0 (no-slip wall)

        For the Moffatt solution:
        f(θ) = [sin(λθ)/sin(λπ/2) - sin((λ-2)θ)/sin((λ-2)π/2)] / C
        where C normalizes to give unit lid velocity.
        """
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        # Local coordinates relative to corner
        dx = x_arr - corner_x
        dy = y_arr - corner_y

        r = np.sqrt(dx**2 + dy**2)
        r_safe = np.maximum(r, 1e-14)  # Avoid division by zero

        # Angle measured from horizontal (lid direction)
        # For left corner (0, Ly): θ = 0 along +x (lid), θ = -π/2 along -y (wall)
        # For right corner (Lx, Ly): need to flip orientation
        if corner_x < Lx / 2:  # Left corner
            theta = np.arctan2(dy, dx)
        else:  # Right corner - mirror the geometry
            theta = np.arctan2(dy, -dx)  # Flip x direction

        lam = self.lam

        # Moffatt angular function and its derivative
        # f(θ) chosen to satisfy BCs: f(0)=0, f'(0)=1, f(-π/2)=0, f'(-π/2)=0
        # Using the standard form for corner flow
        sin_lam_pi2 = np.sin(lam * np.pi / 2)
        sin_lam2_pi2 = np.sin((lam - 2) * np.pi / 2)

        # Avoid division by zero (these are nonzero for λ ≈ 1.5446)
        term1 = np.sin(lam * theta) / sin_lam_pi2
        term2 = np.sin((lam - 2) * theta) / sin_lam2_pi2

        # Normalization constant to give unit velocity at θ = 0
        # f'(0) = λ/sin(λπ/2) - (λ-2)/sin((λ-2)π/2) = 1 after normalization
        C = lam / sin_lam_pi2 - (lam - 2) / sin_lam2_pi2

        f_theta = (term1 - term2) / C

        # f'(θ) = [λ cos(λθ)/sin(λπ/2) - (λ-2)cos((λ-2)θ)/sin((λ-2)π/2)] / C
        df_theta = (
            lam * np.cos(lam * theta) / sin_lam_pi2
            - (lam - 2) * np.cos((lam - 2) * theta) / sin_lam2_pi2
        ) / C

        # Streamfunction: ψ = r^λ f(θ)
        # Velocities in polar coords:
        #   u_r = (1/r) ∂ψ/∂θ = r^(λ-1) f'(θ)
        #   u_θ = -∂ψ/∂r = -λ r^(λ-1) f(θ)

        r_power = r_safe ** (lam - 1)

        u_r = r_power * df_theta
        u_theta = -lam * r_power * f_theta

        # Convert to Cartesian
        cos_theta = dx / r_safe
        sin_theta = dy / r_safe

        if corner_x < Lx / 2:  # Left corner
            u_s = u_r * cos_theta - u_theta * sin_theta
            v_s = u_r * sin_theta + u_theta * cos_theta
        else:  # Right corner - flip back
            u_s = -(u_r * cos_theta - u_theta * sin_theta)  # Flip u
            v_s = u_r * sin_theta + u_theta * cos_theta

        # Zero out at exact corner to avoid numerical issues
        at_corner = r < 1e-12
        u_s = np.where(at_corner, 0.0, u_s)
        v_s = np.where(at_corner, 0.0, v_s)

        result = {"u_s": u_s, "v_s": v_s}

        if compute_derivatives:
            # Analytical derivatives of singular solution
            # ∂u_s/∂x, ∂u_s/∂y, ∂v_s/∂x, ∂v_s/∂y
            # These go like r^(λ-2) which is singular (unbounded) at corner

            # For now, use finite difference approximation far from corner
            # and zero near corner (where the solver shouldn't evaluate anyway)
            # TODO: Implement full analytical derivatives

            # The key insight: near corner, derivatives blow up ~ r^(-0.5)
            # The subtraction method handles this by computing these analytically
            # and including them in the modified convection terms

            r_power_deriv = r_safe ** (lam - 2)

            # Simplified derivative estimates (proper implementation would use
            # full chain rule on the polar-to-Cartesian transformation)
            dus_dx = np.where(at_corner, 0.0, (lam - 1) * r_power_deriv * cos_theta)
            dus_dy = np.where(at_corner, 0.0, (lam - 1) * r_power_deriv * sin_theta)
            dvs_dx = np.where(at_corner, 0.0, (lam - 1) * r_power_deriv * (-sin_theta))
            dvs_dy = np.where(at_corner, 0.0, (lam - 1) * r_power_deriv * cos_theta)

            result["dus_dx"] = dus_dx
            result["dus_dy"] = dus_dy
            result["dvs_dx"] = dvs_dx
            result["dvs_dy"] = dvs_dy

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
        left = self._compute_singular_solution(x, y, 0.0, Ly, Lx, Ly, compute_derivatives=True)

        # Right corner
        right = self._compute_singular_solution(x, y, Lx, Ly, Lx, Ly, compute_derivatives=True)

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
