"""Spectral basis utilities for Navier-Stokes solver."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .polynomial import (
    legendre_gauss_lobatto_nodes,
    vandermonde,
    vandermonde_normalized,
    vandermonde_x,
)


def chebyshev_gauss_lobatto_nodes(num_points: int) -> np.ndarray:
    """
    Return Chebyshev-Gauss-Lobatto nodes on [-1, 1].

    Parameters
    ----------
    num_points : int
        Number of nodes (N+1)

    Returns
    -------
    np.ndarray
        Chebyshev-Gauss-Lobatto nodes: x_j = -cos(πj/N) for j=0,...,N

    Notes
    -----
    These are the extrema of the Chebyshev polynomial T_N(x), including
    the endpoints ±1. Following Zhang et al. (2010), Eq. on page 2.
    """
    N = num_points - 1
    j = np.arange(num_points)
    return -np.cos(np.pi * j / N)


def chebyshev_diff_matrix(nodes: np.ndarray) -> np.ndarray:
    """
    Return Chebyshev spectral differentiation matrix.

    Constructs differentiation matrix D for Chebyshev-Gauss-Lobatto nodes
    following the formulas in Zhang et al. (2010), page 2, Eqs. (3-4).

    Parameters
    ----------
    nodes : np.ndarray
        Chebyshev-Gauss-Lobatto nodes in [-1, 1]

    Returns
    -------
    np.ndarray
        Differentiation matrix of shape (N+1, N+1)

    References
    ----------
    Zhang et al. (2010), "An explicit Chebyshev pseudospectral multigrid method"
    Trefethen (2000), "Spectral Methods in MATLAB"
    """
    N = len(nodes) - 1

    if N == 0:
        return np.zeros((1, 1))

    # Compute weights c_i (Zhang et al. 2010, page 2)
    # c_i = 2 if i = 0 or N, otherwise c_i = 1
    c = np.ones(N + 1)
    c[0] = 2.0
    c[N] = 2.0

    # Build matrix
    D = np.zeros((N + 1, N + 1))

    # Off-diagonal entries: D_ij = (c_i/c_j) * (-1)^(i+j) / (x_i - x_j)
    for i in range(N + 1):
        for j in range(N + 1):
            if i != j:
                D[i, j] = (c[i] / c[j]) * (-1.0)**(i + j) / (nodes[i] - nodes[j])

    # Diagonal entries: Use negative row sum to ensure d/dx[constant] = 0
    # This is equivalent to the interior formula -x_i/(2(1-x_i^2)) for interior points
    # and gives the correct values at boundaries (avoiding the Zhang et al. sign error)
    for i in range(N + 1):
        D[i, i] = -np.sum(D[i, :])

    return D


def legendre_diff_matrix(nodes: np.ndarray) -> np.ndarray:
    r"""
    Return Legendre spectral differentiation matrix at arbitrary nodes.

    Constructs the spectral differentiation matrix :math:`D` such that :math:`D\mathbf{u}`
    approximates :math:`\frac{du}{dx}` at the collocation nodes. The matrix is computed
    using Vandermonde matrices without requiring explicit quadrature.

    Parameters
    ----------
    nodes : np.ndarray
        Collocation nodes

    Returns
    -------
    np.ndarray
        Differentiation matrix of shape (N, N)

    Notes
    -----
    The differentiation matrix is constructed as

    .. math::

        D = V_x V^{-1}

    where :math:`V` is the Vandermonde matrix and :math:`V_x` contains
    derivatives of the basis polynomials. This approach works for arbitrary
    node distributions.

    References
    ----------
    Engsig-Karup, "Lecture 2: Polynomial Methods"
    """
    V = vandermonde(nodes, 0.0, 0.0)
    Vx = vandermonde_x(nodes, 0.0, 0.0)
    identity = np.eye(nodes.size)
    return Vx @ np.linalg.solve(V, identity)


def legendre_mass_matrix(nodes: np.ndarray) -> np.ndarray:
    """
    Return Legendre spectral mass matrix using normalized basis.

    Parameters
    ----------
    nodes : np.ndarray
        Collocation nodes

    Returns
    -------
    np.ndarray
        Mass matrix of shape (N, N)
    """
    V_norm = vandermonde_normalized(nodes, 0.0, 0.0)
    return np.linalg.inv(V_norm @ V_norm.T)


def fourier_diff_matrix_cotangent(N: int) -> np.ndarray:
    """
    Construct Fourier differentiation matrix using cotangent identity.

    Computes the spectral differentiation matrix for periodic functions
    on an equispaced grid using the cotangent formula. The matrix entries
    are constructed directly without FFT operations.

    Parameters
    ----------
    N : int
        Number of grid points

    Returns
    -------
    np.ndarray
        Fourier differentiation matrix of shape (N, N)

    Notes
    -----
    The diagonal entries are set to ensure that differentiating a constant
    function yields zero, which is enforced by requiring each row sum to
    be zero. This construction is exact for the Fourier collocation method
    on periodic domains.

    References
    ----------
    Engsig-Karup, "Lecture 1: Fourier Methods"
    Kopriva (2009), "Implementing Spectral Methods for PDEs"
    """
    indices = np.arange(N)
    diff = indices[:, None] - indices[None, :]
    D = np.zeros((N, N), dtype=float)

    mask = diff != 0
    angles = np.pi * diff[mask] / N
    parity = (-1) ** (indices[:, None] + indices[None, :])

    cot_vals = np.cos(angles) / np.sin(angles)
    D[mask] = 0.5 * parity[mask] * cot_vals

    D[np.diag_indices_from(D)] = -np.sum(D, axis=1)
    return D


def fourier_diff_matrix_complex(N: int) -> np.ndarray:
    """
    Construct complex-valued Fourier differentiation matrix via DFT matrices.

    Parameters
    ----------
    N : int
        Number of grid points

    Returns
    -------
    np.ndarray
        Complex Fourier differentiation matrix of shape (N, N)

    Notes
    -----
    The matrix is assembled using the relation

    .. math::

        D = F^{-1} \\mathrm{diag}(ik) F,

    where :math:`F` is the discrete Fourier transform matrix with equispaced
    nodes on :math:`[0, 2\\pi)`. This corresponds to representing derivatives
    in Fourier space using complex exponentials.
    """
    if N <= 0:
        raise ValueError("Number of grid points N must be positive.")

    indices = np.arange(N, dtype=float)
    # Discrete Fourier transform matrix (consistent with numpy.fft.fft)
    phase = -2j * np.pi * np.outer(indices, indices) / N
    F = np.exp(phase)

    dx = 2 * np.pi / N
    wavenumbers = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    ik = 1j * wavenumbers

    diag_ik_F = ik[:, None] * F
    D = (np.conjugate(F) / N) @ diag_ik_F
    return D.astype(np.complex128)


def fourier_diff_matrix_on_interval(
    N: int,
    a: float = -2.0,
    b: float = 2.0,
    representation: str = "real",
) -> np.ndarray:
    """
    Fourier differentiation matrix rescaled to periodic interval :math:`[a, b]`.

    Parameters
    ----------
    N : int
        Number of grid points
    a : float, optional
        Left endpoint (default: -2.0)
    b : float, optional
        Right endpoint (default: 2.0)
    representation : {"real", "complex"}, optional
        Choose between the real-valued cotangent form or the complex-valued
        DFT form. Default is "real".

    Returns
    -------
    np.ndarray
        Rescaled Fourier differentiation matrix of shape (N, N)
    """
    scale = 2 * np.pi / (b - a)
    rep = representation.lower()
    if rep == "real":
        base = fourier_diff_matrix_cotangent(N)
    elif rep == "complex":
        base = fourier_diff_matrix_complex(N)
    else:
        raise ValueError(
            "Invalid representation. Expected 'real' or 'complex', "
            f"got '{representation}'."
        )
    return scale * base


def barycentric_weights(nodes: np.ndarray) -> np.ndarray:
    """Compute barycentric interpolation weights for given nodes.

    Parameters
    ----------
    nodes : np.ndarray
        Interpolation nodes

    Returns
    -------
    np.ndarray
        Barycentric weights
    """
    n = len(nodes)
    w = np.ones(n)
    for j in range(n):
        for k in range(n):
            if k != j:
                w[j] /= (nodes[j] - nodes[k])
    return w


def barycentric_interpolate(nodes: np.ndarray, values: np.ndarray,
                            x_new: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
    """Barycentric Lagrange interpolation.

    Evaluates the polynomial interpolant at new points using the
    numerically stable barycentric formula.

    Parameters
    ----------
    nodes : np.ndarray
        Original interpolation nodes
    values : np.ndarray
        Function values at nodes
    x_new : np.ndarray
        Points where to evaluate the interpolant
    weights : np.ndarray, optional
        Precomputed barycentric weights

    Returns
    -------
    np.ndarray
        Interpolated values at x_new
    """
    if weights is None:
        weights = barycentric_weights(nodes)

    x_new = np.atleast_1d(x_new)
    result = np.zeros_like(x_new, dtype=float)

    for i, x in enumerate(x_new):
        # Check if x coincides with a node
        diff = x - nodes
        if np.any(np.abs(diff) < 1e-14):
            idx = np.argmin(np.abs(diff))
            result[i] = values[idx]
        else:
            # Barycentric formula
            terms = weights / diff
            result[i] = np.sum(terms * values) / np.sum(terms)

    return result


class SpectralBasis(ABC):
    """Abstract interface for nodal spectral bases."""

    def __init__(self, domain: tuple[float, float] | None = None):
        self.domain = domain
        self._cached_weights = {}  # Cache barycentric weights

    @abstractmethod
    def nodes(self, num_points: int) -> np.ndarray:
        """
        Return nodal points for the basis.

        Parameters
        ----------
        num_points : int
            Number of collocation points

        Returns
        -------
        np.ndarray
            Nodal points in the configured domain
        """

    @abstractmethod
    def diff_matrix(self, nodes: np.ndarray) -> np.ndarray:
        """
        Return differentiation matrix evaluated at `nodes`.

        Parameters
        ----------
        nodes : np.ndarray
            Collocation nodes

        Returns
        -------
        np.ndarray
            Differentiation matrix of shape (N, N)
        """

    def mass_matrix(self, nodes: np.ndarray) -> np.ndarray:
        """
        Return mass (quadrature) matrix for `nodes`.

        Subclasses can override when a closed-form expression is available.
        """
        raise NotImplementedError("Basis does not define a mass matrix.")

    def interpolate(self, nodes: np.ndarray, values: np.ndarray,
                    x_new: np.ndarray) -> np.ndarray:
        """Interpolate values from nodes to new points using barycentric formula.

        Parameters
        ----------
        nodes : np.ndarray
            Original collocation nodes (in physical domain)
        values : np.ndarray
            Function values at nodes
        x_new : np.ndarray
            New points where to evaluate (in physical domain)

        Returns
        -------
        np.ndarray
            Interpolated values at x_new
        """
        # Map to reference domain for numerical stability
        a, b = self.domain
        nodes_ref = 2.0 * (nodes - a) / (b - a) - 1.0
        x_new_ref = 2.0 * (x_new - a) / (b - a) - 1.0

        # Get or compute barycentric weights
        n = len(nodes)
        if n not in self._cached_weights:
            self._cached_weights[n] = barycentric_weights(nodes_ref)

        return barycentric_interpolate(nodes_ref, values, x_new_ref,
                                       self._cached_weights[n])


class LegendreLobattoBasis(SpectralBasis):
    """Legendre-Gauss-Lobatto nodal polynomial basis."""

    def __init__(self, domain: tuple[float, float] = (-1.0, 1.0)):
        super().__init__(domain=domain)

    def nodes(self, num_points: int) -> np.ndarray:
        """
        Return nodes mapped to the configured domain.

        Parameters
        ----------
        num_points : int
            Number of Legendre-Gauss-Lobatto nodes

        Returns
        -------
        np.ndarray
            LGL nodes mapped to the physical domain
        """
        xi = legendre_gauss_lobatto_nodes(num_points)
        if self.domain == (-1.0, 1.0):
            return xi
        a, b = self.domain
        return 0.5 * (b - a) * (xi + 1.0) + a

    def diff_matrix(self, nodes: np.ndarray) -> np.ndarray:
        """
        Return derivative matrix scaled to the physical domain.

        Parameters
        ----------
        nodes : np.ndarray
            Physical domain nodes

        Returns
        -------
        np.ndarray
            Scaled differentiation matrix of shape (N, N)
        """
        xi = legendre_gauss_lobatto_nodes(nodes.size)
        D_xi = legendre_diff_matrix(xi)
        a, b = self.domain
        scale = 2.0 / (b - a)
        return scale * D_xi

    def mass_matrix(self, nodes: np.ndarray) -> np.ndarray:
        """
        Return mass matrix associated with Legendre basis.

        Parameters
        ----------
        nodes : np.ndarray
            Physical domain nodes

        Returns
        -------
        np.ndarray
            Scaled mass matrix of shape (N, N)
        """
        xi = legendre_gauss_lobatto_nodes(nodes.size)
        M = legendre_mass_matrix(xi)
        a, b = self.domain
        return 0.5 * (b - a) * M


class ChebyshevLobattoBasis(SpectralBasis):
    """Chebyshev-Gauss-Lobatto nodal polynomial basis.

    Implements the Chebyshev pseudospectral method as described in
    Zhang et al. (2010), "An explicit Chebyshev pseudospectral multigrid
    method for incompressible Navier-Stokes equations".
    """

    def __init__(self, domain: tuple[float, float] = (-1.0, 1.0)):
        super().__init__(domain=domain)

    def nodes(self, num_points: int) -> np.ndarray:
        """
        Return Chebyshev-Gauss-Lobatto nodes mapped to the configured domain.

        Parameters
        ----------
        num_points : int
            Number of Chebyshev-Gauss-Lobatto nodes

        Returns
        -------
        np.ndarray
            CGL nodes mapped to the physical domain
        """
        xi = chebyshev_gauss_lobatto_nodes(num_points)
        if self.domain == (-1.0, 1.0):
            return xi
        a, b = self.domain
        return 0.5 * (b - a) * (xi + 1.0) + a

    def diff_matrix(self, nodes: np.ndarray) -> np.ndarray:
        """
        Return Chebyshev derivative matrix scaled to the physical domain.

        Parameters
        ----------
        nodes : np.ndarray
            Physical domain nodes

        Returns
        -------
        np.ndarray
            Scaled differentiation matrix of shape (N, N)
        """
        xi = chebyshev_gauss_lobatto_nodes(nodes.size)
        D_xi = chebyshev_diff_matrix(xi)
        a, b = self.domain
        scale = 2.0 / (b - a)
        return scale * D_xi


class FourierEquispacedBasis(SpectralBasis):
    """Equispaced Fourier basis on a periodic interval."""

    def __init__(
        self,
        domain: tuple[float, float] = (0.0, 2.0 * np.pi),
        representation: str = "real",
    ):
        super().__init__(domain=domain)
        self.representation = representation

    def nodes(self, num_points: int) -> np.ndarray:
        """
        Return equispaced nodes on the periodic domain.

        Parameters
        ----------
        num_points : int
            Number of equispaced nodes

        Returns
        -------
        np.ndarray
            Equispaced nodes on the periodic interval
        """
        a, b = self.domain
        return np.linspace(a, b, num_points, endpoint=False)

    def diff_matrix(self, nodes: np.ndarray) -> np.ndarray:
        """
        Return Fourier differentiation matrix.

        Parameters
        ----------
        nodes : np.ndarray
            Fourier collocation nodes

        Returns
        -------
        np.ndarray
            Fourier differentiation matrix of shape (N, N)
        """
        a, b = self.domain
        return fourier_diff_matrix_on_interval(
            nodes.size, a=a, b=b, representation=self.representation
        )

    def mass_matrix(self, nodes: np.ndarray) -> np.ndarray:
        """
        Return diagonal mass matrix for trapezoidal quadrature.

        Parameters
        ----------
        nodes : np.ndarray
            Fourier collocation nodes

        Returns
        -------
        np.ndarray
            Diagonal mass matrix of shape (N, N)
        """
        a, b = self.domain
        return np.eye(nodes.size) * ((b - a) / nodes.size)
