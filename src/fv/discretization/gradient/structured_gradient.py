"""Simple gradient computation for structured Cartesian grids.

For structured grids, we can compute gradients using simple central differences
instead of the complex least-squares approach.
"""
import numpy as np


def compute_cell_gradients_structured(mesh, u, pinned_idx=0, use_limiter=True):
    """Compute cell gradients using central differences for structured Cartesian grids.

    This is much simpler and faster than least-squares for regular grids.
    Uses central differences: du/dx = (u_east - u_west) / (2*dx)

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
