"""Spectral basis utilities for Navier-Stokes solver."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .polynomial import (
    legendre_gauss_lobatto_nodes,
    vandermonde,
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


class SpectralBasis(ABC):
    """Abstract interface for nodal spectral bases."""

    def __init__(self, domain: tuple[float, float] | None = None):
        self.domain = domain

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
