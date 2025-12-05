"""Shared metrics and formatting utilities for solvers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Norms / errors
# -----------------------------------------------------------------------------


def discrete_l2_norm(values: np.ndarray, h: float) -> float:
    """Approximate L2 norm using composite trapezoidal rule."""
    return np.sqrt(h * np.sum(np.abs(values) ** 2))


def discrete_l2_error(
    f_exact: np.ndarray, f_num: np.ndarray, interval_length: float
) -> float:
    """Compute discrete L2 error between exact and numerical solutions."""
    diff = f_num - f_exact
    h = interval_length / f_exact.size
    return np.sqrt(h) * np.linalg.norm(diff)


def discrete_linf_error(f_exact: np.ndarray, f_num: np.ndarray) -> float:
    """Compute discrete L-infinity (maximum) error."""
    return np.max(np.abs(f_num - f_exact))


# -----------------------------------------------------------------------------
# Formatting helpers
# -----------------------------------------------------------------------------


def format_dt_latex(dt: float | str) -> str:
    """Format a timestep value as LaTeX scientific notation."""
    if dt == "?":
        return "?"

    dt_str = f"{float(dt):.2e}"
    mantissa, exp = dt_str.split("e")
    exp_int = int(exp)
    return rf"{mantissa} \times 10^{{{exp_int}}}"


def extract_metadata(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    row_idx: int = 0,
) -> dict[str, Any]:
    """Extract metadata from a DataFrame (assumes constant columns)."""
    if cols is None:
        cols = df.columns.tolist()
    return {col: df[col].iloc[row_idx] for col in cols if col in df.columns}


def format_parameter_range(
    values: list | tuple,
    name: str,
    latex: bool = True,
) -> str:
    """Format a parameter range for display."""
    if len(values) == 0:
        return f"{name} = ?"

    if len(values) == 1:
        val = values[0]
        return rf"${name} = {val}$" if latex else f"{name} = {val}"

    min_val, max_val = min(values), max(values)
    if isinstance(min_val, int) and isinstance(max_val, int):
        range_str = f"[{min_val}, {max_val}]"
    else:
        range_str = f"[{min_val:.1f}, {max_val:.1f}]"

    return rf"${name} \in {range_str}$" if latex else f"{name} ∈ {range_str}"


def build_parameter_string(
    params: dict[str, Any],
    separator: str = ", ",
    latex: bool = True,
) -> str:
    """Build a parameter string from a dictionary."""
    parts = []
    for name, value in params.items():
        if isinstance(value, (list, tuple)):
            parts.append(format_parameter_range(value, name, latex=latex))
        elif "dt" in name.lower() or "delta t" in name:
            value_str = format_dt_latex(value)
            parts.append(
                rf"${name} = {value_str}$" if latex else f"{name} = {value_str}"
            )
        else:
            parts.append(rf"${name} = {value}$" if latex else f"{name} = {value}")
    return separator.join(parts)
