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

from numba import types
from numba.experimental import jitclass

mesh_data_spec = [
    # --- Cell Geometry ---
    ("cell_volumes", types.float64[:]),  # Cell volumes V_C
    ("cell_centers", types.float64[:, :]),  # Cell centroids x_C (shape: [n_cells, 2])
    # --- Face Geometry ---
    ("face_areas", types.float64[:]),            # Face area magnitudes |S_f| (lengths in 2D)
    ("face_centers", types.float64[:, :]),       # Face centroids x_f [n_faces, 2]

    # --- Connectivity ---
    ("owner_cells", types.int64[:]),             # Owner cell index for each face
    ("neighbor_cells", types.int64[:]),          # Neighbor cell index (–1 for boundary faces)
    ("cell_faces", types.int64[:, :]),           # Padded list of face indices for each cell

    # --- Vector Geometry --- For orthogonal Cartesian grids
    ("vector_S_f", types.float64[:, :]),         # surface vectors S_f = unit_normal * area (serves as both S and E for orthogonal grids)
    ("vector_d_CE", types.float64[:, :]),        # distance vector between centroids of elements C and E (also CF in Moukalled)

    # --- Interpolation Factors ---
    ("face_interp_factors", types.float64[:]),   # g_f = delta_Pf / delta_PN

    # --- Topological Masks ---
    ("internal_faces", types.int64[:]),          # Indices of faces with valid neighbor (N >= 0)
    ("boundary_faces", types.int64[:]),          # Indices of faces with N = –1

    # --- Boundary Conditions ---
    ("boundary_values", types.float64[:, :]),    # BC values per face: [u_BC, v_BC, p_BC]
    ("d_Cb", types.float64[:]),                  # Distance from cell center to boundary face center (Moukalled 8.6.8)
]


@jitclass(mesh_data_spec)
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
