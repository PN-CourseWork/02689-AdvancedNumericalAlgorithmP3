"""Validate finite volume solver results against Ghia et al. (1982) benchmark."""

from pathlib import Path
from utils import get_project_root
from utils.ghia_validator import GhiaValidator

# Path to your FV solver results
project_root = get_project_root()
data_dir = project_root / "data" / "FV-Solver"
h5_file = data_dir / "LDC_Re100.h5"

# Create validator and print summary
print(f"\nValidating: {h5_file}")
validator = GhiaValidator(h5_path=h5_file, Re=100)

# Print error metrics summary
validator.print_summary()

# Create validation plots
output_plot = data_dir / "validation_Re100.png"
validator.plot_validation(output_path=output_plot)

# You can also access error metrics programmatically
errors = validator.compute_errors()
print(f"U velocity RMS error: {errors['u_rms']:.6e}")
print(f"V velocity RMS error: {errors['v_rms']:.6e}")
