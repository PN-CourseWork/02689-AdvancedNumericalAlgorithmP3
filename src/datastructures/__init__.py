"""Data structures for solver configuration and results.

This module defines the configuration and result data structures
for lid-driven cavity solvers (both FV and spectral).
"""

from .config import Info, FVinfo, SpectralInfo
from .fields import Fields, FVFields

__all__ = [
    # Configuration and metadata
    "Info",
    "FVinfo",
    "SpectralInfo",
    # Fields
    "Fields",
    "FVFields",
    # Time series
]
