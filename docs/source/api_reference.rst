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

Spectral methods for solving boundary value problems and time-dependent PDEs.

.. currentmodule:: spectral

BVP Solvers
-----------

.. autosummary::
   :toctree: generated
   :nosignatures:

   BvpProblem
   solve_legendre_collocation
   solve_legendre_tau
   solve_polar_bvp
   solve_bvp

Spectral Bases
--------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   LegendreLobattoBasis
   FourierEquispacedBasis
   legendre_diff_matrix
   legendre_mass_matrix
   fourier_diff_matrix_cotangent
   fourier_diff_matrix_complex
   fourier_diff_matrix_on_interval

Time Integrators
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   TimeIntegrator
   get_time_integrator
   RK3
   RK4

PDE Solvers
-----------

.. autosummary::
   :toctree: generated
   :nosignatures:

   KdVSolver
   soliton
   two_soliton_initial
   ManufacturedSolution

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
