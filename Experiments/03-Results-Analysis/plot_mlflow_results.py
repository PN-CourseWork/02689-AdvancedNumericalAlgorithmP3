"""
Fetch Results from MLflow/Databricks
====================================

Retrieve metrics, parameters, and artifacts from MLflow experiments.

Usage::

    uv run python plot_mlflow_results.py
"""

# %%
# Setup
# -----

import mlflow
from utils import load_runs

mlflow.login()

# %%
# Finite Volume Experiment
# ------------------------

df_fv = load_runs("HPC-FV-Solver")
print(f"Finite Volume: {len(df_fv)} converged runs")
df_fv

# %%
# Spectral Experiment
# -------------------

df_spectral = load_runs("HPC-Spectral-Chebyshev")
print(f"Spectral: {len(df_spectral)} converged runs")
df_spectral

