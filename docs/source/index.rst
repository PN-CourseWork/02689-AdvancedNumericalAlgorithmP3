Advanced Numerical Algorithms - Project 3
==========================================

Lid-Driven Cavity Flow: Finite Volume and Spectral Methods

**Authors:** Philip Korsager Nickel, Aske Schrøder Nielsen

This documentation provides computational experiments, API reference, and implementation
details for solving the lid-driven cavity problem using finite volume and spectral methods.

For the full codebase, visit the `GitHub repository <https://github.com/PN-CourseWork/02689-AdvancedNumericalAlgorithmP3>`_.

Quick Start
-----------

.. code-block:: bash

   # Install
   uv sync

   # Run FV validation
   uv run python main.py -m +experiment/validation/ghia=fv

   # View results
   uv run mlflow ui

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   usage
   configuration

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api_reference
