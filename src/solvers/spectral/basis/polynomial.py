"""Polynomial spectral methods: Jacobi polynomials, nodes, and transformations."""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import Legendre
from scipy.special import gammaln, eval_jacobi


# =============================================================================
# Jacobi Polynomials
# =============================================================================


def jacobi_poly(xs: np.ndarray, alpha: float, beta: float, N: int) -> np.ndarray:
    """
    Compute Jacobi polynomial :math:`P_N^{(\alpha,\beta)}(x)` using recurrence relation.

    Jacobi polynomials are orthogonal polynomials on :math:`[-1,1]` with weight
    function :math:`w(x) = (1-x)^\alpha (1+x)^\beta`. Special cases include Legendre
    polynomials (:math:`\alpha=\beta=0`) and Chebyshev polynomials.

    Parameters
    ----------
    xs : np.ndarray
        Evaluation points
    alpha : float
        Jacobi parameter alpha
    beta : float
        Jacobi parameter beta
    N : int
        Polynomial degree

    Returns
    -------
    np.ndarray
        Jacobi polynomial values at xs

    Notes
    -----
    The three-term recurrence relation provides a numerically stable method
    for evaluating Jacobi polynomials without computing derivatives or
    explicit polynomial coefficients.

    References
    ----------
    Engsig-Karup, "Lecture 2: Polynomial Methods", p. 12
    """
    jpm2 = xs**0
    jpm1 = 0.5 * (alpha - beta + (alpha + beta + 2) * xs)
    jpm0 = xs * 0

    if N == 0:
        return jpm2
    if N == 1:
        return jpm1

    for n in range(2, N + 1):
        am1 = (2 * ((n - 1) + alpha) * ((n - 1) + beta)) / (
            (2 * (n - 1) + alpha + beta + 1) * (2 * (n - 1) + alpha + beta)
        )
        a0 = (alpha**2 - beta**2) / (
            (2 * (n - 1) + alpha + beta + 2) * (2 * (n - 1) + alpha + beta)
        )
        ap1 = (2 * ((n - 1) + 1) * ((n - 1) + alpha + beta + 1)) / (
            (2 * (n - 1) + alpha + beta + 2) * (2 * (n - 1) + alpha + beta + 1)
        )

        jpm0 = ((a0 + xs) * jpm1 - am1 * jpm2) / ap1
        jpm2 = jpm1
        jpm1 = jpm0

    return jpm0


def normalized_jacobi_poly(
    xs: np.ndarray, alpha: float, beta: float, N: int
) -> np.ndarray:
    """
    Compute normalized Jacobi polynomial.

    Parameters
    ----------
    xs : np.ndarray
        Evaluation points
    alpha : float
        Jacobi parameter alpha
    beta : float
        Jacobi parameter beta
    N : int
        Polynomial degree

    Returns
    -------
    np.ndarray
        Normalized Jacobi polynomial values at xs
    """
    log_c = -0.5 * (
        np.log(2) * (alpha + beta + 1)
        + gammaln(N + alpha + 1)
        + gammaln(N + beta + 1)
        - gammaln(N + 1)
        - np.log(2 * N + alpha + beta + 1)
        - gammaln(N + alpha + beta + 1)
    )
    return np.exp(log_c) * jacobi_poly(xs, alpha, beta, N)


def legendre_polynomials(xs: np.ndarray, degree: int) -> np.ndarray:
    r"""
    Return Legendre polynomials :math:`P_0, \ldots, P_{\text{degree}}` evaluated at xs.

    Parameters
    ----------
    xs : np.ndarray
        Evaluation points
    degree : int
        Maximum polynomial degree

    Returns
    -------
    np.ndarray
        Array of shape (degree+1, len(xs)) containing polynomial values
    """
    xs = np.asarray(xs)
    polys = np.empty((degree + 1, xs.size))
    for n in range(degree + 1):
        polys[n] = jacobi_poly(xs, 0.0, 0.0, n)
    return polys


def grad_jacobi_poly(
    xs: np.ndarray, alpha: float, beta: float, n: int
) -> np.ndarray | float:
    """
    Compute gradient of Jacobi polynomial.

    Parameters
    ----------
    xs : np.ndarray
        Evaluation points
    alpha : float
        Jacobi parameter alpha
    beta : float
        Jacobi parameter beta
    n : int
        Polynomial degree

    Returns
    -------
    np.ndarray | float
        Derivative values at xs (0 if n=0)
    """
    if n == 0:
        return 0
    return 0.5 * (alpha + beta + n + 1) * jacobi_poly(xs, alpha + 1, beta + 1, n - 1)


# =============================================================================
# Quadrature Nodes
# =============================================================================


def legendre_gauss_lobatto_nodes(num_nodes: int) -> np.ndarray:
    r"""
    Compute Legendre-Gauss-Lobatto (LGL) nodes.

    LGL nodes are the roots of :math:`(1-x^2) P'_N(x)`, where :math:`P_N` is the Legendre
    polynomial of degree :math:`N`. They include the endpoints :math:`\pm 1`, making them ideal
    for imposing Dirichlet boundary conditions.

    Parameters
    ----------
    num_nodes : int
        Number of quadrature nodes

    Returns
    -------
    np.ndarray
        LGL nodes on [-1, 1]

    Notes
    -----
    LGL quadrature integrates polynomials of degree up to :math:`2N-3` exactly.
    The inclusion of boundary points makes these nodes particularly well-suited
    for spectral collocation methods with Dirichlet boundary conditions.

    References
    ----------
    Engsig-Karup, "Lecture 2: Polynomial Methods"
    """
    degree = num_nodes - 1
    roots = Legendre.basis(degree).deriv().roots()
    nodes = np.concatenate(([-1.0], roots, [1.0]))
    return np.sort(nodes)


def legendre_gauss_lobatto_weights(num_nodes: int) -> np.ndarray:
    r"""
    Compute Legendre-Gauss-Lobatto (LGL) quadrature weights.

    LGL weights are computed using the formula:

    .. math::

        w_j = \frac{2}{N(N+1) [P_N(x_j)]^2}

    where :math:`P_N` is the Legendre polynomial of degree :math:`N = num\_nodes - 1`.

    Parameters
    ----------
    num_nodes : int
        Number of quadrature nodes (N+1)

    Returns
    -------
    np.ndarray
        LGL quadrature weights on [-1, 1]

    Notes
    -----
    LGL quadrature integrates polynomials of degree up to :math:`2N-1` exactly.
    The weights satisfy :math:`\sum_j w_j = 2` (the length of [-1, 1]).

    References
    ----------
    Kopriva (2009), "Implementing Spectral Methods for PDEs", Eq. (3.44)
    Canuto et al. (2006), "Spectral Methods: Fundamentals", Section 2.3
    """
    N = num_nodes - 1
    if N == 0:
        return np.array([2.0])

    nodes = legendre_gauss_lobatto_nodes(num_nodes)

    # Evaluate Legendre polynomial P_N at nodes
    P_N = jacobi_poly(nodes, 0.0, 0.0, N)

    # LGL weights formula: w_j = 2 / (N(N+1) * P_N(x_j)^2)
    weights = 2.0 / (N * (N + 1) * P_N**2)

    return weights


# =============================================================================
# Vandermonde Matrices
# =============================================================================


def vandermonde(xs: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    r"""
    Construct Vandermonde matrix for Jacobi polynomials.

    The Vandermonde matrix relates modal (polynomial) coefficients to
    nodal values. Element V[i,j] contains the j-th Jacobi polynomial
    evaluated at the i-th node.

    Parameters
    ----------
    xs : np.ndarray
        Evaluation points
    alpha : float
        Jacobi parameter alpha
    beta : float
        Jacobi parameter beta

    Returns
    -------
    np.ndarray
        Vandermonde matrix of shape (N, N)

    Notes
    -----
    The Vandermonde matrix enables transformation between modal and nodal
    representations:

    .. math::

        \mathbf{u}_{\text{nodal}} = V \mathbf{u}_{\text{modal}}

    Its inverse is used for interpolation and constructing spectral operators.

    References
    ----------
    Engsig-Karup, "Lecture 2: Polynomial Methods", p. 55
    """
    N = len(xs)
    V = np.zeros((N, N))

    for n in range(N):
        V[:, n] = jacobi_poly(xs, alpha, beta, n)

    return V


def vandermonde_normalized(xs: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """
    Construct normalized Vandermonde matrix.

    Parameters
    ----------
    xs : np.ndarray
        Evaluation points
    alpha : float
        Jacobi parameter alpha
    beta : float
        Jacobi parameter beta

    Returns
    -------
    np.ndarray
        Normalized Vandermonde matrix of shape (N, N)
    """
    N = len(xs)
    V = np.zeros((N, N))

    for n in range(N):
        V[:, n] = normalized_jacobi_poly(xs, alpha, beta, n)

    return V


def vandermonde_x(xs: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """
    Construct derivative Vandermonde matrix.

    Parameters
    ----------
    xs : np.ndarray
        Evaluation points
    alpha : float
        Jacobi parameter alpha
    beta : float
        Jacobi parameter beta

    Returns
    -------
    np.ndarray
        Derivative Vandermonde matrix of shape (N, N)
    """
    N = len(xs)
    Vx = np.zeros((N, N))

    for n in range(N):
        Vx[:, n] = grad_jacobi_poly(xs, alpha, beta, n)

    return Vx


def generalized_vandermonde(x: np.ndarray, degree: int | None = None) -> np.ndarray:
    """
    Construct generalized Vandermonde matrix using Legendre polynomials.

    Parameters
    ----------
    x : np.ndarray
        Evaluation points
    degree : int, optional
        Maximum polynomial degree (default: len(x) - 1)

    Returns
    -------
    np.ndarray
        Generalized Vandermonde matrix
    """
    if degree is None:
        degree = x.size - 1
    return legendre_polynomials(x, degree).T


# =============================================================================
# Modal-Nodal Transformations
# =============================================================================


def modal_to_nodal(x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Reconstruct function from Legendre coefficients at points x.

    Parameters
    ----------
    x : np.ndarray
        Evaluation points
    coeffs : np.ndarray
        Modal coefficients

    Returns
    -------
    np.ndarray
        Function values at x
    """
    result = np.zeros_like(x)
    for n, cn in enumerate(coeffs):
        Pn = eval_jacobi(n, 0, 0, x)
        result += cn * Pn
    return result


def spectral_interpolate(
    x_nodes: np.ndarray,
    f_values: np.ndarray,
    x_eval: np.ndarray,
    basis: str = "legendre",
) -> np.ndarray:
    """
    Spectrally interpolate function values at new points.

    Uses the Vandermonde matrix approach: compute modal coefficients from
    nodal values, then evaluate the polynomial expansion at new points.
    This preserves spectral accuracy unlike spline interpolation.

    Parameters
    ----------
    x_nodes : np.ndarray
        Original collocation nodes (e.g., Chebyshev-Gauss-Lobatto or LGL nodes)
    f_values : np.ndarray
        Function values at x_nodes
    x_eval : np.ndarray
        Points where to evaluate the interpolant
    basis : str, optional
        Polynomial basis: "legendre" (alpha=beta=0) or "chebyshev" (alpha=beta=-0.5)
        Default is "legendre".

    Returns
    -------
    np.ndarray
        Interpolated values at x_eval

    Notes
    -----
    The interpolation is computed as:

    .. math::

        f(x_{eval}) = V_{eval} V^{-1} f_{nodes}

    where V is the Vandermonde matrix at the original nodes and V_eval is
    the Vandermonde matrix at the evaluation points.

    For Chebyshev nodes, using Legendre basis still works well and avoids
    the weight function singularities of Chebyshev polynomials at endpoints.

    References
    ----------
    Engsig-Karup, "Lecture 2: Polynomial Methods", p. 55-56
    """
    # Set Jacobi parameters based on basis
    if basis.lower() == "legendre":
        alpha, beta = 0.0, 0.0
    elif basis.lower() == "chebyshev":
        alpha, beta = -0.5, -0.5
    else:
        raise ValueError(f"Unknown basis: {basis}. Use 'legendre' or 'chebyshev'.")

    # Map nodes and eval points to [-1, 1] if needed
    x_min, x_max = x_nodes.min(), x_nodes.max()
    if not (np.isclose(x_min, -1.0) and np.isclose(x_max, 1.0)):
        # Affine map to reference domain [-1, 1]
        x_nodes_ref = 2.0 * (x_nodes - x_min) / (x_max - x_min) - 1.0
        x_eval_ref = 2.0 * (x_eval - x_min) / (x_max - x_min) - 1.0
    else:
        x_nodes_ref = x_nodes
        x_eval_ref = x_eval

    # Compute Vandermonde matrix at original nodes
    V = vandermonde(x_nodes_ref, alpha, beta)

    # Compute modal coefficients: f_modal = V^{-1} f_nodal
    f_modal = np.linalg.solve(V, f_values)

    # Build Vandermonde at evaluation points (may have different size)
    N = len(x_nodes)
    V_eval = np.zeros((len(x_eval), N))
    for n in range(N):
        V_eval[:, n] = jacobi_poly(x_eval_ref, alpha, beta, n)

    # Evaluate polynomial: f_interp = V_eval @ f_modal
    return V_eval @ f_modal


def spectral_interpolate_2d(
    x_nodes: np.ndarray,
    y_nodes: np.ndarray,
    f_2d: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    basis: str = "legendre",
) -> np.ndarray:
    """
    Spectrally interpolate a 2D field at new points using tensor product.

    Parameters
    ----------
    x_nodes : np.ndarray
        Original x collocation nodes (1D array of length nx)
    y_nodes : np.ndarray
        Original y collocation nodes (1D array of length ny)
    f_2d : np.ndarray
        2D field values with shape (ny, nx)
    x_eval : np.ndarray
        x coordinates where to evaluate (1D array)
    y_eval : np.ndarray
        y coordinates where to evaluate (1D array, same length as x_eval)
    basis : str, optional
        Polynomial basis: "legendre" or "chebyshev". Default is "legendre".

    Returns
    -------
    np.ndarray
        Interpolated values at (x_eval, y_eval) points

    Notes
    -----
    For tensor product grids, 2D interpolation can be done as:
    1. First interpolate in x-direction for each row
    2. Then interpolate in y-direction at each evaluation point

    Or equivalently using the tensor product of 1D interpolation matrices.
    """
    n_eval = len(x_eval)
    if len(y_eval) != n_eval:
        raise ValueError("x_eval and y_eval must have the same length")

    ny, nx = f_2d.shape

    # Step 1: Interpolate in x-direction for each y-row
    # This gives values at x_eval for each original y_node
    f_x_interp = np.zeros((ny, n_eval))
    for j in range(ny):
        f_x_interp[j, :] = spectral_interpolate(x_nodes, f_2d[j, :], x_eval, basis=basis)

    # Step 2: For each evaluation point, interpolate in y-direction
    result = np.zeros(n_eval)
    for i in range(n_eval):
        # Get the column of x-interpolated values at x_eval[i]
        f_column = f_x_interp[:, i]
        # Interpolate this column at y_eval[i]
        result[i] = spectral_interpolate(y_nodes, f_column, np.array([y_eval[i]]), basis=basis)[0]

    return result


def spectral_interpolate_line(
    x_nodes: np.ndarray,
    y_nodes: np.ndarray,
    f_2d: np.ndarray,
    line_coord: float,
    line_axis: str,
    eval_points: np.ndarray,
    basis: str = "legendre",
) -> np.ndarray:
    """
    Extract a line from a 2D spectral field at an exact coordinate.

    This properly interpolates to the exact line position rather than
    taking the nearest grid slice.

    Parameters
    ----------
    x_nodes : np.ndarray
        Original x collocation nodes (1D array of length nx)
    y_nodes : np.ndarray
        Original y collocation nodes (1D array of length ny)
    f_2d : np.ndarray
        2D field values with shape (ny, nx)
    line_coord : float
        The coordinate value for the line (e.g., x=0.5 for vertical centerline)
    line_axis : str
        Which axis the line_coord refers to: "x" for vertical line, "y" for horizontal
    eval_points : np.ndarray
        Points along the line where to evaluate (y values for vertical, x for horizontal)
    basis : str, optional
        Polynomial basis: "legendre" or "chebyshev". Default is "legendre".

    Returns
    -------
    np.ndarray
        Field values along the line at eval_points

    Examples
    --------
    # Extract u-velocity along vertical centerline x=0.5
    y_line = np.linspace(0, 1, 200)
    u_centerline = spectral_interpolate_line(x, y, U, 0.5, "x", y_line)

    # Extract v-velocity along horizontal centerline y=0.5
    x_line = np.linspace(0, 1, 200)
    v_centerline = spectral_interpolate_line(x, y, V, 0.5, "y", x_line)
    """
    ny, nx = f_2d.shape
    n_eval = len(eval_points)

    if line_axis.lower() == "x":
        # Vertical line at x=line_coord, varying y
        # First interpolate each row (y-slice) to the exact x position
        f_at_x = np.zeros(ny)
        for j in range(ny):
            f_at_x[j] = spectral_interpolate(
                x_nodes, f_2d[j, :], np.array([line_coord]), basis=basis
            )[0]
        # Now interpolate this column to the evaluation y-points
        return spectral_interpolate(y_nodes, f_at_x, eval_points, basis=basis)

    elif line_axis.lower() == "y":
        # Horizontal line at y=line_coord, varying x
        # First interpolate each column (x-slice) to the exact y position
        f_at_y = np.zeros(nx)
        for i in range(nx):
            f_at_y[i] = spectral_interpolate(
                y_nodes, f_2d[:, i], np.array([line_coord]), basis=basis
            )[0]
        # Now interpolate this row to the evaluation x-points
        return spectral_interpolate(x_nodes, f_at_y, eval_points, basis=basis)

    else:
        raise ValueError(f"line_axis must be 'x' or 'y', got {line_axis}")
