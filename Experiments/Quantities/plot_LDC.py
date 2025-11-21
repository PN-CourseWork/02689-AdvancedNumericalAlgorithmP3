"""
Conserved Quantities Visualization
===================================

This script visualizes conserved quantities (energy, enstrophy, palinstrophy)
tracked during the lid-driven cavity flow computation.
"""

# %%
# Setup and Load Data
# -------------------
# Load the computed solution and extract time series data.

import pandas as pd
import h5py
from utils import get_project_root

project_root = get_project_root()
data_dir = project_root / "data" / "Quantities"
figures_dir = project_root / "figures" / "quantities"
figures_dir.mkdir(parents=True, exist_ok=True)

# Load time series data from HDF5
h5_file = data_dir / "LDC_Re100_quantities.h5"
with h5py.File(h5_file, "r") as f:
    energy = f["time_series/energy"][:]
    enstrophy = f["time_series/enstrophy"][:]
    palinstropy = f["time_series/palinstropy"][:]

print(f"Loaded solution from: {h5_file}")

# %%
# Plot conserved quantities vs iteration
# -------------------------------------

# Create DataFrame from loaded time series data
df = pd.DataFrame(
    {
        "Energy": energy,
        "Enstrophy": enstrophy,
        "Palinstropy": palinstropy,
    }
)

# Plot using pandas
axes = df.plot(subplots=True, layout=(1, 3), figsize=(15, 4))

fig = axes[0, 0].figure
fig.suptitle("Lid-driven cavity: conserved quantities vs iteration")

# Save figure
savepath = figures_dir / "ldc_quantities_vs_iteration.pdf"
fig.savefig(savepath)

# %%
# Plot normalized conserved quantities vs iteration
# --------------------------------------------------

# Normalize by first iteration values
df_normalized = df / df.iloc[0]

# Plot using pandas
axes_norm = df_normalized.plot(subplots=True, layout=(1, 3), figsize=(15, 4))

fig_norm = axes_norm[0, 0].figure
fig_norm.suptitle("Lid-driven cavity: normalized conserved quantities vs iteration")

# Save figure
savepath_norm = figures_dir / "ldc_quantities_normalized_vs_iteration.pdf"
fig_norm.savefig(savepath_norm, dpi=200, bbox_inches="tight")

print(f"\nAll figures saved to: {figures_dir}")
