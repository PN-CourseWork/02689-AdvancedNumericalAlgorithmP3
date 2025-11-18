"""Data structures for solver configuration and results.

This module defines the configuration and result data structures
for lid-driven cavity solvers (both FV and spectral).
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import numpy as np

#========================================================
# Mesh Data Classes
# =======================================================

@dataclass
class Mesh:
    """Simple structured mesh.

    Usage: mesh = Mesh(config)
    """
    config: Any

    def __post_init__(self):
        """Create mesh from config."""
        # Create 1D coordinates
        self.x = np.linspace(0, self.config.Lx, self.config.nx)
        self.y = np.linspace(0, self.config.Ly, self.config.ny)

        # Grid spacing
        self.dx = self.x[1] - self.x[0] if self.config.nx > 1 else self.config.Lx
        self.dy = self.y[1] - self.y[0] if self.config.ny > 1 else self.config.Ly

        # 2D meshgrid
        X, Y = np.meshgrid(self.x, self.y)

        # Flattened grid points
        self.grid_points = np.column_stack([X.flatten(), Y.flatten()])

        # Number of cells/points
        self.n_cells = self.config.nx * self.config.ny


#========================================================
# Shared Data Classes
# ======================================================= 




@dataclass
class Fields:
    """Base spatial solution fields."""
    mesh: Any  # Mesh or subclass

    def __post_init__(self):
        """Initialize zero fields from mesh."""
        # Get number of cells (works for both Mesh and FvMesh)
        if hasattr(self.mesh, 'n_cells'):
            n = self.mesh.n_cells
        else:
            n = self.mesh.cell_volumes.shape[0]

        # Solution fields
        self.u = np.zeros(n)
        self.v = np.zeros(n)
        self.p = np.zeros(n)

        # Previous iteration (for under-relaxation)
        self.u_prev_iter = np.zeros(n)
        self.v_prev_iter = np.zeros(n)

        # Grid information (works for both Mesh and FvMesh)
        if hasattr(self.mesh, 'x') and hasattr(self.mesh, 'y'):
            self.x = self.mesh.x
            self.y = self.mesh.y
            self.grid_points = self.mesh.grid_points
        else:
            # Extract from cell centers for FvMesh
            self.x = np.unique(self.mesh.cell_centers[:, 0])
            self.y = np.unique(self.mesh.cell_centers[:, 1])
            self.grid_points = self.mesh.cell_centers


@dataclass
class TimeSeries:
    """Time series data common to all solvers."""
    rel_residual: List[float] = field(default_factory=list)
    u_residual: List[float] = field(default_factory=list)
    v_residual: List[float] = field(default_factory=list)
    #TODO: Add the quantities stuff from the paper


@dataclass
class Meta:
    """Base solver metadata, config and convergence info."""
    # Physics parameters (required)
    Re: float

    # Grid parameters (with defaults)
    nx: int = 64
    ny: int = 64

    # Physics parameters (with defaults)
    lid_velocity: float = 1
    Lx: float = 1
    Ly: float = 1

    # Solver config
    max_iterations: int = 500
    tolerance: float = 1e-4
    method: str = ""

    # Convergence info
    iterations: int = 0
    converged: bool = False
    final_residual: float = 0.0


#=============================================================
# Finite Volume specific data classes
# ============================================================

@dataclass
class FVmeta(Meta):
    """FV-specific metadata with discretization parameters."""
    convection_scheme: str = "Upwind"
    limiter: str = "MUSCL"
    alpha_uv: float = 0.6
    alpha_p: float = 0.4


@dataclass
class FVfields(Fields):
    """FV-specific fields with mass flux."""

    def __post_init__(self):
        """Initialize FV fields including mass flux."""
        super().__post_init__()

        # FV-specific: mass flux on faces
        n_faces = self.mesh.internal_faces.shape[0] + self.mesh.boundary_faces.shape[0]
        self.mdot = np.zeros(n_faces)



