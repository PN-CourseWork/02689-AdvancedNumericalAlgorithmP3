"""
Conserved Quantities Export + Plotting
======================================

Reads conserved quantities from the HDF5 result file, writes CSVs,
and generates a simple plot of Energy, Enstrophy, and Palinstropy
vs iteration.
"""

from pathlib import Path
import h5py
import csv
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------
# Locate project paths
# ---------------------------------------------------------------------
project_root = Path(__file__).resolve().parents[2]
data_dir = project_root / "data" / "Quantities"
out_dir = data_dir
out_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Load HDF5 file
# ---------------------------------------------------------------------
h5_file = data_dir / "LDC_Re100_quantities.h5"
if not h5_file.exists():
    raise FileNotFoundError(f"HDF5 file not found: {h5_file}")

with h5py.File(h5_file, "r") as f:
    energy = f["time_series/energy"][:] if "time_series/energy" in f else []
    enstrophy = f["time_series/enstrophy"][:] if "time_series/enstrophy" in f else []
    palinstropy = f["time_series/palinstropy"][:] if "time_series/palinstropy" in f else []

print(f"Loaded solution from: {h5_file}")

n = len(energy)
print(f"Number of time points: {n}")

if not (len(energy) == len(enstrophy) == len(palinstropy)):
    raise ValueError("Time series arrays have mismatched lengths.")

# ---------------------------------------------------------------------
# Write CSV files
# ---------------------------------------------------------------------
csv_path = out_dir / "quantities.csv"
with open(csv_path, "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["iteration", "Energy", "Enstrophy", "Palinstropy"])
    for i, (e, en, p) in enumerate(zip(energy, enstrophy, palinstropy)):
        writer.writerow([i, float(e), float(en), float(p)])

csv_norm_path = out_dir / "quantities_normalized.csv"
if n > 0:
    e0 = float(energy[0]) if energy[0] != 0 else 1.0
    en0 = float(enstrophy[0]) if enstrophy[0] != 0 else 1.0
    p0 = float(palinstropy[0]) if palinstropy[0] != 0 else 1.0
else:
    e0 = en0 = p0 = 1.0

with open(csv_norm_path, "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["iteration", "Energy_norm", "Enstrophy_norm", "Palinstropy_norm"])
    for i, (e, en, p) in enumerate(zip(energy, enstrophy, palinstropy)):
        writer.writerow([
            i,
            float(e) / e0,
            float(en) / en0,
            float(p) / p0
        ])

print("\nCSV files written:")
print(f"  Raw values:      {csv_path}")
print(f"  Normalized vals: {csv_norm_path}")

# ---------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------
if n > 0:
    print("\nSummary (first / last):")
    print(f"  Energy:     {float(energy[0]):.6e}  ->  {float(energy[-1]):.6e}")
    print(f"  Enstrophy:  {float(enstrophy[0]):.6e}  ->  {float(enstrophy[-1]):.6e}")
    print(f"  Palinstropy:{float(palinstropy[0]):.6e}  ->  {float(palinstropy[-1]):.6e}")
else:
    print("No data in HDF5 time series.")

# ---------------------------------------------------------------------
# Make a plot of the time series
# ---------------------------------------------------------------------
iters = np.arange(n)

plt.figure(figsize=(12, 6))

plt.plot(iters, energy, label="Energy")
plt.plot(iters, enstrophy, label="Enstrophy")
plt.plot(iters, palinstropy, label="Palinstropy")

plt.xlabel("Iteration")
plt.ylabel("Value")
plt.title("Conserved Quantities vs Iteration")
plt.grid(True)
plt.legend()

png_path = out_dir / "quantities_vs_iteration.png"
pdf_path = out_dir / "quantities_vs_iteration.pdf"

plt.savefig(png_path, dpi=200, bbox_inches="tight")
plt.savefig(pdf_path)

print(f"\nPlots saved to:")
print(f"  {png_path}")
print(f"  {pdf_path}")

plt.show()