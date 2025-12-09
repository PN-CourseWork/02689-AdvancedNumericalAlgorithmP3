#!/usr/bin/env python
"""Plot MMS convergence results with publication-quality styling."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Enable LaTeX rendering
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})

# Set seaborn style with darkgrid
sns.set_style("darkgrid", {
    'axes.facecolor': 'none',
    'figure.facecolor': 'none',
    'savefig.facecolor': 'none',
})


def plot_mms_convergence(parquet_path: Path, output_dir: Path):
    """Create MMS convergence plot from parquet data."""
    # Load data
    df = pd.read_parquet(parquet_path)
    df = df.sort_values('N')

    # Create figure with transparent background
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

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

    # Add theoretical spectral convergence reference line
    N_ref = np.array([6, 16])
    # Spectral convergence: error ~ C * exp(-alpha * N)
    # Fit from the data roughly
    c0 = df[df['N'] == 6]['u_error'].values[0]
    alpha = 2.0  # approximate spectral decay rate
    spectral_ref = c0 * np.exp(-alpha * (N_ref - 6))
    ax.semilogy(N_ref, spectral_ref, 'k:', linewidth=1.5, alpha=0.7,
                label=r'Spectral: $\mathcal{O}(e^{-\alpha N})$')

    # Labels and title
    ax.set_xlabel(r'$N$ (polynomial degree)', fontsize=12)
    ax.set_ylabel(r'Relative $L^2$ Error', fontsize=12)

    # Get Re value from data
    Re = int(df['Re'].iloc[0])
    ax.set_title(r'\textbf{MMS Spectral Convergence} ($\mathrm{Re} = ' + str(Re) + r'$)',
                 fontsize=14, pad=10)

    # Legend with transparent background
    legend = ax.legend(loc='upper right', framealpha=0.0, edgecolor='none')
    legend.get_frame().set_facecolor('none')

    # Set x-axis ticks to show all N values
    ax.set_xticks(df['N'].values)
    ax.set_xticklabels([str(n) for n in df['N'].values])

    # Grid styling (already set by darkgrid, but ensure visibility)
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)

    # Tight layout
    plt.tight_layout()

    # Save with transparent background
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'mms_convergence_Re{Re}_styled.pdf'
    plt.savefig(output_file, transparent=True, bbox_inches='tight', dpi=300)
    print(f"Saved: {output_file}")

    # Also save PNG for quick viewing
    output_png = output_dir / f'mms_convergence_Re{Re}_styled.png'
    plt.savefig(output_png, transparent=True, bbox_inches='tight', dpi=300)
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
