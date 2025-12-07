"""Spectral multigrid variants."""

from solvers.spectral.multigrid.fsg import build_hierarchy, solve_fsg, solve_vmg

__all__ = ["build_hierarchy", "solve_fsg", "solve_vmg"]
