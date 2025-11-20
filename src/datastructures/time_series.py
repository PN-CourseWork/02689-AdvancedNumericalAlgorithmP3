"""Time series data structures."""
from dataclasses import dataclass
from typing import List


@dataclass
class TimeSeries:
    """Time series data common to all solvers.

    Parameters
    ----------
    residual : List[float]
        Overall residual history over iterations.
    u_residual : List[float], optional
        x-velocity residual history. Default is None.
    v_residual : List[float], optional
        y-velocity residual history. Default is None.
    continuity_residual : List[float], optional
        Continuity equation residual history. Default is None.
    """
    residual: List[float]
    u_residual: List[float] = None
    v_residual: List[float] = None
    continuity_residual: List[float] = None
    #TODO: Add the quantities stuff from the paper
