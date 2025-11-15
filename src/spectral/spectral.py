"""Spectral basis utilities for Navier-Stokes solver."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .polynomial import (
    legendre_gauss_lobatto_nodes,
    vandermonde,
    vandermonde_x,
)


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
