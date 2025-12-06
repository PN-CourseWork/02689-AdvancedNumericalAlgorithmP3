"""
Plotting Style Configuration for LDC Plots.

Uses seaborn darkgrid theme.
"""

import logging

import seaborn as sns

log = logging.getLogger(__name__)

# Use seaborn darkgrid theme
sns.set_theme(style="darkgrid")
