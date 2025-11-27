"""
Download Artifacts from MLflow
==============================

Download HDF5 artifacts from converged runs and place them in data/
with standardized naming (LDC_N{nx}_Re{Re}.h5).

Usage::

    uv run python download_artifacts.py
"""

# %%
# Setup
# -----

import mlflow
from utils import get_project_root, download_artifacts_with_naming

mlflow.login()
project_root = get_project_root()

# %%
# Finite Volume Artifacts
# -----------------------

fv_dir = project_root / "data" / "FV-Solver"
print("Downloading Finite Volume artifacts...")
fv_paths = download_artifacts_with_naming("HPC-FV-Solver", fv_dir)
print(f"Downloaded {len(fv_paths)} files to {fv_dir}\n")

# %%
# Spectral Artifacts
# ------------------

spectral_dir = project_root / "data" / "Spectral-Solver"
print("Downloading Spectral artifacts...")
spectral_paths = download_artifacts_with_naming("HPC-Spectral-Chebyshev", spectral_dir)
print(f"Downloaded {len(spectral_paths)} files to {spectral_dir}")
