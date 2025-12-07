"""Transfer operators for spectral multigrid methods.

Implements prolongation (coarse -> fine) and restriction (fine -> coarse)
operators for multigrid algorithms.

Based on Zhang & Xi (2010): "An explicit Chebyshev pseudospectral multigrid
method for incompressible Navier-Stokes equations"

Two methods are supported:
- FFT/DCT-based: Uses Discrete Cosine Transform (paper method, Eq. 10-11)
- Polynomial-based: Uses Chebyshev polynomial fitting/evaluation

For FSG (Full Single Grid), only prolongation is needed.
For VMG/FMG, both prolongation and restriction are required.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import numpy as np
from scipy.fft import dct


class ProlongationMethod(Enum):
    """Available prolongation methods."""

    FFT = "fft"
    POLYNOMIAL = "polynomial"


class RestrictionMethod(Enum):
    """Available restriction methods."""

    FFT = "fft"
    INJECTION = "injection"


# =============================================================================
# Abstract Base Classes
# =============================================================================


class Prolongation(ABC):
    """Abstract base class for prolongation operators (coarse -> fine)."""

    @abstractmethod
    def prolongate_1d(self, u_coarse: np.ndarray, n_fine: int) -> np.ndarray:
        """Interpolate 1D array from coarse to fine grid.

        Parameters
        ----------
        u_coarse : np.ndarray
            Values on coarse grid (n_coarse points)
        n_fine : int
            Number of points on fine grid

        Returns
        -------
        np.ndarray
            Interpolated values on fine grid (n_fine points)
        """
        pass

    def prolongate_2d(
        self,
        u_coarse_2d: np.ndarray,
        shape_fine: Tuple[int, int],
    ) -> np.ndarray:
        """Prolongate 2D field from coarse to fine grid.

        Uses row-column algorithm: interpolate in x-direction, then y-direction.

        Parameters
        ----------
        u_coarse_2d : np.ndarray
            Field on coarse grid, shape (nx_c, ny_c)
        shape_fine : tuple
            Target shape (nx_f, ny_f)

        Returns
        -------
        np.ndarray
            Field on fine grid
        """
        nx_c, ny_c = u_coarse_2d.shape
        nx_f, ny_f = shape_fine

        if (nx_c, ny_c) == (nx_f, ny_f):
            return u_coarse_2d.copy()

        # Interpolate in x-direction (column by column)
        temp = np.zeros((nx_f, ny_c))
        for j in range(ny_c):
            temp[:, j] = self.prolongate_1d(u_coarse_2d[:, j], nx_f)

        # Interpolate in y-direction (row by row)
        u_fine_2d = np.zeros((nx_f, ny_f))
        for i in range(nx_f):
            u_fine_2d[i, :] = self.prolongate_1d(temp[i, :], ny_f)

        return u_fine_2d


class Restriction(ABC):
    """Abstract base class for restriction operators (fine -> coarse)."""

    @abstractmethod
    def restrict_1d(self, u_fine: np.ndarray, n_coarse: int) -> np.ndarray:
        """Restrict 1D array from fine to coarse grid.

        Parameters
        ----------
        u_fine : np.ndarray
            Values on fine grid (n_fine points)
        n_coarse : int
            Number of points on coarse grid

        Returns
        -------
        np.ndarray
            Restricted values on coarse grid (n_coarse points)
        """
        pass

    def restrict_2d(
        self,
        u_fine_2d: np.ndarray,
        shape_coarse: Tuple[int, int],
    ) -> np.ndarray:
        """Restrict 2D field from fine to coarse grid.

        Uses row-column algorithm: restrict in x-direction, then y-direction.

        Parameters
        ----------
        u_fine_2d : np.ndarray
            Field on fine grid, shape (nx_f, ny_f)
        shape_coarse : tuple
            Target shape (nx_c, ny_c)

        Returns
        -------
        np.ndarray
            Field on coarse grid
        """
        nx_f, ny_f = u_fine_2d.shape
        nx_c, ny_c = shape_coarse

        if (nx_f, ny_f) == (nx_c, ny_c):
            return u_fine_2d.copy()

        # Restrict in x-direction (column by column)
        temp = np.zeros((nx_c, ny_f))
        for j in range(ny_f):
            temp[:, j] = self.restrict_1d(u_fine_2d[:, j], nx_c)

        # Restrict in y-direction (row by row)
        u_coarse_2d = np.zeros((nx_c, ny_c))
        for i in range(nx_c):
            u_coarse_2d[i, :] = self.restrict_1d(temp[i, :], ny_c)

        return u_coarse_2d


# =============================================================================
# FFT/DCT-based Operators (Zhang & Xi 2010 paper method)
# =============================================================================


class FFTProlongation(Prolongation):
    """FFT/DCT-based prolongation operator.

    From Zhang & Xi (2010), Eq. 10-11:
    1. Compute discrete Chebyshev coefficients via DCT-I
    2. Evaluate polynomial at fine grid points

    This is spectrally accurate and efficient via FFT.

    The key relationship is:
        c_k = DCT-I(f) / N

    where N is the polynomial degree (n_points - 1), and the Chebyshev series
    uses the primed notation where c_0 and c_N are halved during evaluation.
    """

    def prolongate_1d(self, u_coarse: np.ndarray, n_fine: int) -> np.ndarray:
        """Prolongate using DCT-based spectral interpolation.

        Uses Chebyshev interpolation: compute coefficients on coarse grid,
        then evaluate the polynomial at fine grid points.

        Parameters
        ----------
        u_coarse : np.ndarray
            Values on coarse Chebyshev-Lobatto grid (n_coarse points)
        n_fine : int
            Number of points on fine grid

        Returns
        -------
        np.ndarray
            Interpolated values on fine grid
        """
        n_coarse = len(u_coarse)

        if n_coarse == n_fine:
            return u_coarse.copy()

        if n_coarse > n_fine:
            raise ValueError(
                f"Prolongation requires n_coarse ({n_coarse}) <= n_fine ({n_fine})"
            )

        # Step 1: Compute Chebyshev coefficients via DCT-I
        # The relationship is: c_k = DCT-I(f) / N (no pre-weighting needed)
        N_c = n_coarse - 1
        coeffs = dct(u_coarse, type=1) / N_c

        # Step 2: Evaluate Chebyshev polynomial at fine grid points
        # The Chebyshev series uses primed notation:
        #   f(x) = c_0/2 + sum_{k=1}^{N-1} c_k * T_k(x) + c_N/2 * T_N(x)
        # At CGL nodes x_i = cos(pi*i/M): T_k(x_i) = cos(k*pi*i/M)
        N_f = n_fine - 1
        u_fine = np.zeros(n_fine)

        for i in range(n_fine):
            theta = np.pi * i / N_f
            # First coefficient halved
            u_fine[i] = coeffs[0] / 2
            # Interior coefficients
            for k in range(1, N_c):
                u_fine[i] += coeffs[k] * np.cos(k * theta)
            # Last coefficient halved
            u_fine[i] += coeffs[N_c] / 2 * np.cos(N_c * theta)

        return u_fine


class FFTRestriction(Restriction):
    """FFT/DCT-based restriction operator.

    From Zhang & Xi (2010), Section 3.3:
    1. Compute discrete Chebyshev coefficients via DCT-I
    2. **Truncate high-frequency coefficients** (set to zero)
    3. Evaluate on coarse grid using only low-frequency coefficients

    This is used for restricting residuals in V-cycle multigrid.
    The truncation prevents aliasing and maintains stability.

    The key relationship is:
        c_k = DCT-I(f) / N

    where N is the polynomial degree (n_points - 1), and the Chebyshev series
    uses the primed notation where c_0 and c_N are halved during evaluation.
    """

    def restrict_1d(self, u_fine: np.ndarray, n_coarse: int) -> np.ndarray:
        """Restrict using DCT-based spectral truncation.

        From the paper: "Afterwards we set the coefficients belonging to
        the high frequencies to zero and compute the residuals on the
        coarse grid collocation points."

        Parameters
        ----------
        u_fine : np.ndarray
            Values on fine Chebyshev-Lobatto grid (n_fine points)
        n_coarse : int
            Number of points on coarse grid

        Returns
        -------
        np.ndarray
            Restricted values on coarse grid
        """
        n_fine = len(u_fine)

        if n_fine == n_coarse:
            return u_fine.copy()

        if n_fine < n_coarse:
            raise ValueError(
                f"Restriction requires n_fine ({n_fine}) >= n_coarse ({n_coarse})"
            )

        # Step 1: Compute Chebyshev coefficients via DCT-I
        # The relationship is: c_k = DCT-I(f) / N (no pre-weighting needed)
        N_f = n_fine - 1
        coeffs = dct(u_fine, type=1) / N_f

        # Step 2: Evaluate polynomial at coarse grid points using TRUNCATED coefficients
        # Per Zhang & Xi (2010): "set the coefficients belonging to the high
        # frequencies to zero" - only use first N_c + 1 coefficients
        # The Chebyshev series uses primed notation:
        #   f(x) = c_0/2 + sum_{k=1}^{N_c-1} c_k * T_k(x) + c_{N_c}/2 * T_{N_c}(x)
        N_c = n_coarse - 1
        u_coarse = np.zeros(n_coarse)

        for i in range(n_coarse):
            theta = np.pi * i / N_c
            # First coefficient halved
            u_coarse[i] = coeffs[0] / 2
            # Interior coefficients (only up to N_c - 1, TRUNCATING high frequencies)
            for k in range(1, N_c):
                u_coarse[i] += coeffs[k] * np.cos(k * theta)
            # Last TRUNCATED coefficient halved
            u_coarse[i] += coeffs[N_c] / 2 * np.cos(N_c * theta)

        return u_coarse


# =============================================================================
# Polynomial-based Operators (original implementation)
# =============================================================================


class PolynomialProlongation(Prolongation):
    """Polynomial-based prolongation using Chebyshev fitting.

    Uses numpy's Chebyshev polynomial routines:
    1. Fit Chebyshev polynomial to coarse grid data
    2. Evaluate at fine grid points

    This is mathematically equivalent to FFT method but uses
    direct polynomial operations.
    """

    def prolongate_1d(self, u_coarse: np.ndarray, n_fine: int) -> np.ndarray:
        """Prolongate using Chebyshev polynomial fitting.

        Parameters
        ----------
        u_coarse : np.ndarray
            Values on coarse Chebyshev-Lobatto grid (n_coarse points)
        n_fine : int
            Number of points on fine grid

        Returns
        -------
        np.ndarray
            Interpolated values on fine grid
        """
        from numpy.polynomial.chebyshev import chebfit, chebval

        n_coarse = len(u_coarse)

        if n_coarse == n_fine:
            return u_coarse.copy()

        # Chebyshev-Lobatto nodes on [-1, 1]
        x_coarse = np.cos(np.pi * np.arange(n_coarse) / (n_coarse - 1))
        x_fine = np.cos(np.pi * np.arange(n_fine) / (n_fine - 1))

        # Fit Chebyshev polynomial to coarse data
        coeffs = chebfit(x_coarse, u_coarse, deg=n_coarse - 1)

        # Evaluate at fine grid points
        u_fine = chebval(x_fine, coeffs)

        return u_fine


class InjectionRestriction(Restriction):
    """Direct injection restriction operator.

    For Chebyshev-Lobatto grids with N_coarse = N_fine / 2,
    the coarse grid points are a subset of fine grid points.
    This allows simple direct injection.

    This is used for restricting variables (not residuals) in FAS scheme.
    """

    def restrict_1d(self, u_fine: np.ndarray, n_coarse: int) -> np.ndarray:
        """Restrict using direct injection.

        Parameters
        ----------
        u_fine : np.ndarray
            Values on fine Chebyshev-Lobatto grid (n_fine points)
        n_coarse : int
            Number of points on coarse grid

        Returns
        -------
        np.ndarray
            Restricted values on coarse grid
        """
        n_fine = len(u_fine)

        if n_fine == n_coarse:
            return u_fine.copy()

        # For Lobatto grids with full coarsening (N_c = N_f / 2),
        # coarse points are every other fine point
        if n_fine == 2 * n_coarse - 1:
            # Standard case: N_f = 2*N_c - 1
            return u_fine[::2].copy()
        elif n_fine % 2 == 1 and n_coarse == (n_fine + 1) // 2:
            # Alternative indexing
            return u_fine[::2].copy()
        else:
            # Fallback: use indices based on cosine mapping
            # Find closest fine grid points to coarse grid points
            x_fine = np.cos(np.pi * np.arange(n_fine) / (n_fine - 1))
            x_coarse = np.cos(np.pi * np.arange(n_coarse) / (n_coarse - 1))

            u_coarse = np.zeros(n_coarse)
            for i, xc in enumerate(x_coarse):
                idx = np.argmin(np.abs(x_fine - xc))
                u_coarse[i] = u_fine[idx]

            return u_coarse


# =============================================================================
# Transfer Operator Container
# =============================================================================


@dataclass
class TransferOperators:
    """Container for prolongation and restriction operators.

    This class holds the configured operators and provides convenience
    methods for transferring solutions between multigrid levels.
    """

    prolongation: Prolongation
    restriction: Restriction

    def prolongate_field(
        self,
        field_coarse: np.ndarray,
        shape_coarse: Tuple[int, int],
        shape_fine: Tuple[int, int],
    ) -> np.ndarray:
        """Prolongate a flattened field from coarse to fine grid.

        Parameters
        ----------
        field_coarse : np.ndarray
            Flattened field on coarse grid
        shape_coarse : tuple
            Shape of coarse grid (nx, ny)
        shape_fine : tuple
            Shape of fine grid (nx, ny)

        Returns
        -------
        np.ndarray
            Flattened field on fine grid
        """
        field_2d = field_coarse.reshape(shape_coarse)
        result_2d = self.prolongation.prolongate_2d(field_2d, shape_fine)
        return result_2d.ravel()

    def restrict_field(
        self,
        field_fine: np.ndarray,
        shape_fine: Tuple[int, int],
        shape_coarse: Tuple[int, int],
    ) -> np.ndarray:
        """Restrict a flattened field from fine to coarse grid.

        Parameters
        ----------
        field_fine : np.ndarray
            Flattened field on fine grid
        shape_fine : tuple
            Shape of fine grid (nx, ny)
        shape_coarse : tuple
            Shape of coarse grid (nx, ny)

        Returns
        -------
        np.ndarray
            Flattened field on coarse grid
        """
        field_2d = field_fine.reshape(shape_fine)
        result_2d = self.restriction.restrict_2d(field_2d, shape_coarse)
        return result_2d.ravel()


# =============================================================================
# Factory Function
# =============================================================================


def create_transfer_operators(
    prolongation_method: str = "fft",
    restriction_method: str = "fft",
) -> TransferOperators:
    """Create transfer operators from configuration.

    Parameters
    ----------
    prolongation_method : str
        Method for prolongation: "fft" or "polynomial"
    restriction_method : str
        Method for restriction: "fft" or "injection"

    Returns
    -------
    TransferOperators
        Configured transfer operators
    """
    # Create prolongation operator
    if prolongation_method == "fft":
        prolongation = FFTProlongation()
    elif prolongation_method == "polynomial":
        prolongation = PolynomialProlongation()
    else:
        raise ValueError(f"Unknown prolongation method: {prolongation_method}")

    # Create restriction operator
    if restriction_method == "fft":
        restriction = FFTRestriction()
    elif restriction_method == "injection":
        restriction = InjectionRestriction()
    else:
        raise ValueError(f"Unknown restriction method: {restriction_method}")

    return TransferOperators(prolongation=prolongation, restriction=restriction)
