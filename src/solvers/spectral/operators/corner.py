"""Corner singularity treatment for lid-driven cavity flow.

Two methods for handling corner singularities at the lid-wall junctions:

1. Smoothing method: Simple cosine smoothing of lid velocity near corners.
   - Easy to implement, works well
   - Approximation to the standard cavity problem

2. Saad/Polynomial method: Polynomial regularization u = 16x²(1-x)².
   - C∞ smooth profile from Shenfun/Saad
   - Used in spectral methods to avoid Gibbs oscillations
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

    def uses_modified_convection(self) -> bool:
        """Return True if this method requires modified convection terms.

        Only the subtraction method (removed) required this.
        """
        return False


# =============================================================================
# Method 1: Smoothing (Cosine smoothing near corners)
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
# Method 2: Saad/Polynomial Regularization (smooth C∞ profile)
# =============================================================================


class SaadTreatment(CornerTreatment):
    """Corner treatment using polynomial regularization (Saad approach).

    Uses the profile:
        u = (1-ξ)²(1+ξ)² where ξ ∈ [-1, 1]

    On physical domain [0, Lx], this becomes:
        u = 16 * (x/Lx)² * (1 - x/Lx)² * lid_velocity

    This gives:
    - u = 0 at corners (x=0 and x=Lx)
    - u = lid_velocity at center (x=Lx/2)
    - C∞ smooth everywhere (all derivatives exist)

    Reference: Shenfun documentation, used in Saad-style benchmarks
    """

    def get_lid_velocity(
        self,
        x: np.ndarray,
        y: np.ndarray,
        lid_velocity: float,
        Lx: float,
        Ly: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Lid velocity with polynomial: 16x²(1-x)²."""
        x_flat = np.asarray(x).ravel()

        # Normalized coordinate
        xi = x_flat / Lx

        # Polynomial profile: 16 * xi^2 * (1 - xi)^2
        # Equivalent to (1-ξ)²(1+ξ)² on [-1,1] domain
        profile = 16.0 * xi**2 * (1.0 - xi)**2

        u_lid = profile * lid_velocity
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


# Alias for backward compatibility
PolynomialTreatment = SaadTreatment


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
        Treatment method:
        - "smoothing": Cosine smoothing near corners (controlled by smoothing_width)
        - "saad" or "polynomial": u = 16x²(1-x)² regularization (C∞ smooth)
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
    elif method_lower in ("polynomial", "saad"):
        return SaadTreatment()
    else:
        raise ValueError(
            f"Unknown corner treatment method: {method}. "
            f"Use 'smoothing', 'polynomial', or 'saad'."
        )
