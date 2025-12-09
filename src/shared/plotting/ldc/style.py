"""
Plotting Style Configuration for LDC Plots.

Uses seaborn darkgrid theme with LaTeX rendering.
"""

import logging

import matplotlib.pyplot as plt
import seaborn as sns

log = logging.getLogger(__name__)

# Enable LaTeX rendering
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 12,
        "font.size": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)

# Use seaborn darkgrid theme (after rcParams to preserve LaTeX settings)
sns.set_theme(style="darkgrid", rc={"text.usetex": True})
