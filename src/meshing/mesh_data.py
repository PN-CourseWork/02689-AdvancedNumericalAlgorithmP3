"""
MeshData2D: Core data layout for finite volume CFD (2D, collocated).

This class defines static geometry, connectivity, boundary tagging, and precomputed metrics,
following Moukalled's finite volume formulation.

Indexing Conventions:
- All face-based arrays (e.g., face_normals, owner_cells) use face indexing (0 to n_faces-1).
- All cell-based arrays (e.g., cell_volumes, cell_centers) use cell indexing (0 to n_cells-1).
- Boundary-related arrays (e.g., boundary_values, boundary_types, d_PB) have full-face length (n_faces).
    * Internal faces use sentinel defaults: boundary_types = [-1, -1], boundary_values = [0, 0, 0], d_PB = 0.0

Boundary Condition Metadata:
- boundary_values[f, :] = [u_BC, v_BC, p_BC] for face f. Zero for internal.
- All velocity boundaries use Dirichlet BC with fixed values.
- d_Cb[f] = distance from cell center to boundary face center (used for one-sided gradients)

Fast Boolean Masks:
- face_boundary_mask[f] = 1 if face is boundary, 0 otherwise
- face_flux_mask[f] = 1 if face is active in flux computation, 0 otherwise
"""

class MeshData2D:
    def __init__(
        self,
        cell_volumes,
        cell_centers,
        face_areas,
        face_centers,
        owner_cells,
        neighbor_cells,
        cell_faces,
        vector_S_f,
        vector_d_CE,
        face_interp_factors,
        internal_faces,
        boundary_faces,
        boundary_values,
        d_Cb,
        nx=None,
        ny=None,
        dx=None,
        dy=None,
    ):
        # --- Geometry ---
        self.cell_volumes = cell_volumes
        self.cell_centers = cell_centers
        self.face_areas = face_areas
        self.face_centers = face_centers

        # --- Connectivity ---
        self.owner_cells = owner_cells
        self.neighbor_cells = neighbor_cells
        self.cell_faces = cell_faces

        # --- Vector Geometry ---
        self.vector_S_f = vector_S_f
        self.vector_d_CE = vector_d_CE

        # --- Interpolation Factors ---
        self.face_interp_factors = face_interp_factors

        # --- Topological Info ---
        self.internal_faces = internal_faces
        self.boundary_faces = boundary_faces

        # --- BCs ---
        self.boundary_values = boundary_values
        self.d_Cb = d_Cb

        # --- Structured Grid Info (optional) ---
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
