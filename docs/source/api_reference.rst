.. _api_reference:

=============
API Reference
=============

This page provides an overview of the project modules.

Finite Volume Package (``fv``)
================================

Core iteration functions for SIMPLE/PISO algorithms.

.. currentmodule:: fv

.. autosummary::
   :toctree: generated
   :nosignatures:

   simple_step
   initialize_simple_state

Spectral Methods Package (``spectral``)
=========================================

Spectral methods for solving Navier-Stokes equations using Legendre-Gauss-Lobatto grids.

.. currentmodule:: spectral

.. autosummary::
   :toctree: generated
   :nosignatures:

   LegendreLobattoBasis
   legendre_diff_matrix
   get_repo_root

Utilities Package (``utils``)
================================

Utility functions for plotting, data I/O, and validation.

.. currentmodule:: utils

.. autosummary::
   :toctree: generated
   :nosignatures:

   get_project_root
   LDCPlotter
   GhiaValidator
   load_run_data
   load_fields
   load_metadata
   load_multiple_runs
