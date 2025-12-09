"""Test script for PyVista-based LDC plotting functions.

Usage:
    python scripts/test_pyvista_plots.py
    python scripts/test_pyvista_plots.py --vts path/to/solution.vts
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from shared.plotting.ldc.pyvista_fields import generate_pyvista_field_plots

# Default VTS solution from mlruns
DEFAULT_VTS = Path(__file__).parent.parent / "mlruns" / "806262599561726093" / "719e5ef40a454b8d8a60a3f1a222c7f3" / "artifacts" / "solution.vts"


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test PyVista LDC plotting functions")
    parser.add_argument("--vts", type=str, help="Path to solution.vts file")
    parser.add_argument(
        "--output", type=str, default=None, help="Output directory"
    )
    args = parser.parse_args()

    # Find VTS file
    if args.vts:
        vts_path = Path(args.vts)
        if not vts_path.exists():
            print(f"Error: VTS file not found: {vts_path}")
            sys.exit(1)
        print(f"Loading solution from: {vts_path}")
    else:
        if not DEFAULT_VTS.exists():
            print(f"Error: Default VTS not found: {DEFAULT_VTS}")
            print("Please provide --vts path/to/solution.vts")
            sys.exit(1)
        vts_path = DEFAULT_VTS
        print(f"Using default VTS: {vts_path}")

    # Output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(__file__).parent.parent / "figures" / "pyvista_test"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Generate all plots
    print("\nGenerating PyVista plots...")
    paths = generate_pyvista_field_plots(
        vts_path=vts_path,
        output_dir=output_dir,
    )

    print(f"\nGenerated {len(paths)} plots:")
    for name, p in paths.items():
        print(f"  - {name}: {p.name}")


if __name__ == "__main__":
    main()
