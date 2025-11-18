"""Simple gradient computation for structured Cartesian grids.

For structured grids, we can compute gradients using NumPy's built-in gradient function.
"""
import numpy as np


def compute_cell_gradients_structured(mesh, u, pinned_idx=0, use_limiter=True):
    """Compute cell gradients using np.gradient for structured Cartesian grids.

    Parameters
    ----------
    mesh : MeshData2D
        Structured mesh data
    u : ndarray
        Cell-centered field values
    pinned_idx : int
        Cell index to pin gradient to zero (for pressure)
    use_limiter : bool
        Apply Barth-Jespersen limiter to gradients

    Returns
    -------
    grad : ndarray (n_cells, 2)
        Cell gradients [du/dx, du/dy]
    """
    # Check if mesh has structured grid info
    if mesh.nx is None or mesh.ny is None:
        # Fall back to slow version if not a structured mesh
        return _compute_cell_gradients_structured_slow(mesh, u, pinned_idx, use_limiter)

    nx, ny = mesh.nx, mesh.ny
    dx, dy = mesh.dx, mesh.dy

    # Reshape u to 2D grid (cell index = i * ny + j)
    u_grid = u.reshape((nx, ny))

    # Compute gradients using vectorized array operations
    gx_grid = np.zeros((nx, ny), dtype=np.float64)
    gy_grid = np.zeros((nx, ny), dtype=np.float64)

    # Central differences in interior cells
    gx_grid[1:-1, :] = (u_grid[2:, :] - u_grid[:-2, :]) / (2.0 * dx)
    gy_grid[:, 1:-1] = (u_grid[:, 2:] - u_grid[:, :-2]) / (2.0 * dy)

    # One-sided differences at boundaries
    gx_grid[0, :] = (u_grid[1, :] - u_grid[0, :]) / dx
    gx_grid[-1, :] = (u_grid[-1, :] - u_grid[-2, :]) / dx
    gy_grid[:, 0] = (u_grid[:, 1] - u_grid[:, 0]) / dy
    gy_grid[:, -1] = (u_grid[:, -1] - u_grid[:, -2]) / dy

    # Pin gradient at specified cell
    if pinned_idx >= 0:
        i_pin = pinned_idx // ny
        j_pin = pinned_idx % ny
        gx_grid[i_pin, j_pin] = 0.0
        gy_grid[i_pin, j_pin] = 0.0

    # Apply Barth-Jespersen limiter if requested
    if use_limiter:
        phi = np.ones((nx, ny), dtype=np.float64)

        # Compute min/max among neighbors (vectorized)
        umin = u_grid.copy()
        umax = u_grid.copy()

        # Update with neighbors
        umin[1:, :] = np.minimum(umin[1:, :], u_grid[:-1, :])  # West
        umax[1:, :] = np.maximum(umax[1:, :], u_grid[:-1, :])

        umin[:-1, :] = np.minimum(umin[:-1, :], u_grid[1:, :])  # East
        umax[:-1, :] = np.maximum(umax[:-1, :], u_grid[1:, :])

        umin[:, 1:] = np.minimum(umin[:, 1:], u_grid[:, :-1])  # South
        umax[:, 1:] = np.maximum(umax[:, 1:], u_grid[:, :-1])

        umin[:, :-1] = np.minimum(umin[:, :-1], u_grid[:, 1:])  # North
        umax[:, :-1] = np.maximum(umax[:, :-1], u_grid[:, 1:])

        # Compute limiter for each cell (only where needed)
        needs_limiting = (umax > u_grid) | (umin < u_grid)

        # Vectorized limiter computation
        for i in range(nx):
            for j in range(ny):
                cell_idx = i * ny + j
                if cell_idx == pinned_idx or not needs_limiting[i, j]:
                    continue

                u_c = u_grid[i, j]
                gx_c = gx_grid[i, j]
                gy_c = gy_grid[i, j]
                phi_c = 1.0

                # Check all 4 neighbors
                if i > 0:  # West
                    delta_u = gx_c * (-dx)
                    if delta_u > 1e-20:
                        phi_c = min(phi_c, (umax[i, j] - u_c) / delta_u)
                    elif delta_u < -1e-20:
                        phi_c = min(phi_c, (umin[i, j] - u_c) / delta_u)

                if i < nx-1:  # East
                    delta_u = gx_c * dx
                    if delta_u > 1e-20:
                        phi_c = min(phi_c, (umax[i, j] - u_c) / delta_u)
                    elif delta_u < -1e-20:
                        phi_c = min(phi_c, (umin[i, j] - u_c) / delta_u)

                if j > 0:  # South
                    delta_u = gy_c * (-dy)
                    if delta_u > 1e-20:
                        phi_c = min(phi_c, (umax[i, j] - u_c) / delta_u)
                    elif delta_u < -1e-20:
                        phi_c = min(phi_c, (umin[i, j] - u_c) / delta_u)

                if j < ny-1:  # North
                    delta_u = gy_c * dy
                    if delta_u > 1e-20:
                        phi_c = min(phi_c, (umax[i, j] - u_c) / delta_u)
                    elif delta_u < -1e-20:
                        phi_c = min(phi_c, (umin[i, j] - u_c) / delta_u)

                phi[i, j] = phi_c

        # Apply limiter
        gx_grid *= phi
        gy_grid *= phi

    # Flatten back to 1D and combine
    grad = np.column_stack([gx_grid.ravel(), gy_grid.ravel()])

    return grad


def _compute_cell_gradients_structured_slow(mesh, u, pinned_idx=0, use_limiter=True):
    """Fallback slow version for non-structured meshes."""
    n_cells = mesh.cell_centers.shape[0]
    grad = np.zeros((n_cells, 2), dtype=np.float64)

    # Pre-fetch mesh arrays
    cell_faces = mesh.cell_faces
    owner_cells = mesh.owner_cells
    neighbor_cells = mesh.neighbor_cells
    cc = mesh.cell_centers

    for c in range(n_cells):
        if c == pinned_idx:
            grad[c, 0] = grad[c, 1] = 0.0
            continue

        u_c = u[c]
        x_c = cc[c, 0]
        y_c = cc[c, 1]

        # Accumulators for central difference
        du_dx_sum = 0.0
        du_dy_sum = 0.0
        count_x = 0
        count_y = 0

        # For limiter
        umin = u_c
        umax = u_c

        # Loop over cell faces to find neighbors
        for f in cell_faces[c]:
            if f < 0:
                break

            P = owner_cells[f]
            N = neighbor_cells[f]

            if N >= 0:  # Internal face only
                other = N if c == P else P
                if other == pinned_idx:
                    continue

                other_u = u[other]
                other_x = cc[other, 0]
                other_y = cc[other, 1]

                # Determine direction (x or y) based on face orientation
                dx = other_x - x_c
                dy = other_y - y_c

                # For structured Cartesian grid, faces are aligned with axes
                if abs(dx) > abs(dy):  # East-West face
                    distance = abs(dx)
                    if distance > 1e-12:
                        du_dx_sum += (other_u - u_c) / dx
                        count_x += 1
                else:  # North-South face
                    distance = abs(dy)
                    if distance > 1e-12:
                        du_dy_sum += (other_u - u_c) / dy
                        count_y += 1

                # Track min/max for limiter
                if use_limiter:
                    if other_u < umin:
                        umin = other_u
                    if other_u > umax:
                        umax = other_u

        # Average gradients (for central difference, we have 2 faces per direction)
        gx = du_dx_sum / count_x if count_x > 0 else 0.0
        gy = du_dy_sum / count_y if count_y > 0 else 0.0

        # Apply Barth-Jespersen limiter if requested
        phi = 1.0
        if use_limiter and (umax > u_c or umin < u_c):
            for f in cell_faces[c]:
                if f < 0:
                    break

                P = owner_cells[f]
                N = neighbor_cells[f]

                if N >= 0:
                    other = N if c == P else P
                    if other == pinned_idx:
                        continue

                    dx = cc[other, 0] - x_c
                    dy = cc[other, 1] - y_c
                    delta_u = gx * dx + gy * dy

                    if delta_u > 1e-20:
                        phi = min(phi, (umax - u_c) / delta_u)
                    elif delta_u < -1e-20:
                        phi = min(phi, (umin - u_c) / delta_u)

        grad[c, 0] = phi * gx
        grad[c, 1] = phi * gy

    return grad
