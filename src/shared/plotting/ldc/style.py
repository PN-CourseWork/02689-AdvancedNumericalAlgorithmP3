"""
Plotting Style Configuration for LDC Plots.

Uses scientific.mplstyle with seaborn color palette.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

log = logging.getLogger(__name__)

# Load scientific style file
_STYLE_PATH = Path(__file__).parent.parent / "scientific.mplstyle"

if _STYLE_PATH.exists():
    plt.style.use(str(_STYLE_PATH))
    log.debug(f"Loaded scientific style: {_STYLE_PATH}")
else:
    log.warning(f"Scientific style file not found: {_STYLE_PATH}")

# Apply seaborn theme on top for color palette
sns.set_theme()
