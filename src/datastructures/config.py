"""Configuration and metadata data structures."""
from dataclasses import dataclass, asdict
import pandas as pd


@dataclass
class Info:
    """Base solver metadata, config and convergence info.

    Parameters
    ----------
    Re : float
        Reynolds number.
    nx : int, optional
        Number of grid points in x-direction. Default is 64.
    ny : int, optional
        Number of grid points in y-direction. Default is 64.
    lid_velocity : float, optional
        Velocity of the lid. Default is 1.
    Lx : float, optional
        Domain length in x-direction. Default is 1.
    Ly : float, optional
        Domain length in y-direction. Default is 1.
    max_iterations : int, optional
        Maximum number of iterations. Default is 500.
    tolerance : float, optional
        Convergence tolerance. Default is 1e-4.
    method : str, optional
        Solver method name. Default is None.
    iterations : int, optional
        Actual number of iterations performed. Default is None.
    converged : bool, optional
        Whether the solver converged. Default is False.
    final_residual : float, optional
        Final residual value. Default is None.
    """
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
    method: str = None

    # Convergence info
    iterations: int = None
    converged: bool = False
    final_residual: float = None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert config/metadata to single-row DataFrame.

        Returns
        -------
        pd.DataFrame
            Single-row DataFrame with all configuration and metadata fields.
        """
        return pd.DataFrame([asdict(self)])


@dataclass
class FVinfo(Info):
    """FV-specific metadata with discretization parameters.

    Inherits all parameters from Info and adds FV-specific parameters.

    Parameters
    ----------
    convection_scheme : str, optional
        Convection scheme (e.g., 'Upwind', 'CDS'). Default is 'Upwind'.
    limiter : str, optional
        Flux limiter scheme. Default is 'MUSCL'.
    alpha_uv : float, optional
        Under-relaxation factor for velocity. Default is 0.6.
    alpha_p : float, optional
        Under-relaxation factor for pressure. Default is 0.4.
    """
    convection_scheme: str = "Upwind"
    limiter: str = "MUSCL"
    alpha_uv: float = 0.6
    alpha_p: float = 0.4


@dataclass
class SpectralInfo(Info):
    """Spectral-specific metadata with discretization parameters.

    Inherits all parameters from Info and adds spectral-specific parameters.

    Parameters
    ----------
    Nx : int, optional
        Number of spectral grid points in x-direction. Default is 64.
    Ny : int, optional
        Number of spectral grid points in y-direction. Default is 64.
    differentiation_method : str, optional
        Differentiation method ('fft', 'chebyshev', 'matrix'). Default is 'fft'.
    time_scheme : str, optional
        Time integration scheme. Default is 'rk4'.
    dt : float, optional
        Time step size. Default is 0.001.
    dealiasing : bool, optional
        Whether to use dealiasing. Default is True.
    multigrid : bool, optional
        Whether to use multigrid acceleration. Default is False.
    mg_levels : int, optional
        Number of multigrid levels. Default is 3.
    """
    Nx: int = 64
    Ny: int = 64
    differentiation_method: str = "fft"  # 'fft', 'chebyshev', 'matrix'
    time_scheme: str = "rk4"
    dt: float = 0.001
    dealiasing: bool = True
    multigrid: bool = False
    mg_levels: int = 3
