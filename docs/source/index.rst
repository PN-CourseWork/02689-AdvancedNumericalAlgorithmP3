Advanced Numerical Algorithms - Project 3
==========================================

Lid-Driven Cavity Flow: Finite Volume and Spectral Methods

**Authors:** Philip Korsager Nickel, Aske Schrøder Nielsen

This documentation provides computational experiments, API reference, and implementation
details for solving the lid-driven cavity problem using finite volume and spectral methods.

For the full codebase, please visit the `GitHub repository <https://github.com/PN-CourseWork/02689-AdvancedNumericalAlgorithmP3>`_.

Contents
--------

:doc:`example_gallery/index`
   Gallery of computational experiments and visualizations for lid-driven cavity flow.
:doc:`api_reference`
   Complete API reference for finite volume, spectral methods, and utility modules.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Examples

   example_gallery/index

.. toctree::
   :maxdepth: 2
   :hidden:
   :titlesonly:
   :caption: Reference

   api_reference

Installation
------------

The package requires Python 3.12 and uses ``uv`` for dependency management.

Run the setup script from the project root::

    bash setup.sh

This will create a virtual environment and install all dependencies including PETSc and petsc4py.

