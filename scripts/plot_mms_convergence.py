#!/usr/bin/env python
"""Plot MMS convergence results with publication-quality styling."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Enable LaTeX rendering first
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'axes.labelsize': 12,
    'font.size': 11,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})

# Use seaborn darkgrid theme (after rcParams to preserve LaTeX settings)
sns.set_theme(style="darkgrid", rc={"text.usetex": True})


def plot_mms_convergence(parquet_path: Path, output_dir: Path):
    """Create MMS convergence plot from parquet data."""
    # Load data
    df = pd.read_parquet(parquet_path)
    df = df.sort_values('N')

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot u and v errors
    ax.semilogy(df['N'], df['u_error'], 'o-', color='#1f77b4',
                label=r'$\|u - u_{\mathrm{exact}}\|_{L^2}$', markersize=8, linewidth=2)
    ax.semilogy(df['N'], df['v_error'], 's--', color='#ff7f0e',
                label=r'$\|v - v_{\mathrm{exact}}\|_{L^2}$', markersize=8, linewidth=2)

    # Mark non-converged points
    non_converged = df[~df['converged']]
    if len(non_converged) > 0:
        ax.scatter(non_converged['N'], non_converged['u_error'],
                   marker='x', s=150, c='red', zorder=5, linewidths=2,
                   label=r'Did not converge')

    # Labels and title
    ax.set_xlabel(r'$N$ (grid points)')
    ax.set_ylabel(r'$L^2$ Error')

    # Get Re value from data
    Re = int(df['Re'].iloc[0])
    ax.set_title(rf'\textbf{{MMS Spectral Convergence}} --- $\mathrm{{Re}}={Re}$')

    # Legend
    ax.legend(loc='upper right', frameon=True)

    # Set x-axis ticks to show all N values
    ax.set_xticks(df['N'].values)
    ax.set_xticklabels([str(n) for n in df['N'].values])

    # Transparent figure background, but keep darkgrid axes background
    fig.patch.set_alpha(0.0)

    # Tight layout
    plt.tight_layout()

    # Save with transparent background
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'mms_convergence_Re{Re}_styled.pdf'
    fig.savefig(output_file, facecolor=(0, 0, 0, 0), bbox_inches='tight', dpi=300)
    print(f"Saved: {output_file}")

    # Also save PNG for quick viewing
    output_png = output_dir / f'mms_convergence_Re{Re}_styled.png'
    fig.savefig(output_png, facecolor=(0, 0, 0, 0), bbox_inches='tight', dpi=300)
    print(f"Saved: {output_png}")

    plt.close()

    return output_file


def main():
    """Main entry point."""
    # Paths
    data_path = Path('data/MMS/mms_results.parquet')
    output_dir = Path('figures')

    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        print("Run the MMS sweep first: uv run python scripts/test_sg_mms.py -m")
        return

    # Create plot
    plot_mms_convergence(data_path, output_dir)

    # Print summary
    df = pd.read_parquet(data_path)
    print("\nMMS Results Summary:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
