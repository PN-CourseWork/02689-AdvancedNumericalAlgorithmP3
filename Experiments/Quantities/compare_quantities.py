#!/usr/bin/env python3
"""
Compare our solver conserved quantities (from quantities_normalized.csv if present,
otherwise from HDF5) with Saad table values read from saad_Re1000_quantities.csv,
and also plot the comparison.

Saves:
 - data/Quantities/compare_quantities_report.csv
 - data/Quantities/compare_quantities.png
 - data/Quantities/compare_quantities.pdf
"""

from pathlib import Path
import csv
import sys

try:
    import numpy as np
except Exception:
    raise SystemExit("This script requires numpy. Install it in your venv: pip install numpy")

# plotting import is optional; script works without it (produces CSV report only)
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

import h5py
import numpy as np

# -----------------------------
# utility that already existed
# -----------------------------
def compute_fv_quantities_from_hdf5(h5_path):
    """
    Compute energy, enstrophy, palinstrophy directly from FV data stored in HDF5.
    This bypasses the base solver and ensures correct FV-style quantities.
    """
    if not h5_path.exists():
        raise FileNotFoundError(f"Cannot compute FV quantities: file not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        u = f["fields/u"][()]
        v = f["fields/v"][()]
        grid = f["fields/grid_points"][()]

    # infer uniform grid spacing
    xs = np.unique(grid[:, 0])
    ys = np.unique(grid[:, 1])
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    dA = dx * dy

    # ----- Energy -----
    energy = 0.5 * np.sum(u * u + v * v) * dA

    # ----- Enstrophy -----
    # finite-difference gradients
    Nx = len(xs)
    Ny = len(ys)

    U = u.reshape(Ny, Nx)
    V = v.reshape(Ny, Nx)

    dudy = (np.roll(U, -1, axis=0) - np.roll(U, 1, axis=0)) / (2 * dy)
    dvdx = (np.roll(V, -1, axis=1) - np.roll(V, 1, axis=1)) / (2 * dx)

    omega = dvdx - dudy
    enstrophy = 0.5 * np.sum(omega * omega) * dA

    # ----- Palinstrophy -----
    omegax = (np.roll(omega, -1, axis=1) - np.roll(omega, 1, axis=1)) / (2 * dx)
    omegay = (np.roll(omega, -1, axis=0) - np.roll(omega, 1, axis=0)) / (2 * dy)

    palinstropy = np.sum(omegax * omegax + omegay * omegay) * dA

    return float(energy), float(enstrophy), float(palinstropy)


# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "Quantities"
SAAD_DIR = PROJECT_ROOT / "data" / "validation" / "ghia"

# prefer repository table; fallback to local copy (created during debugging)
SAAD_TABLE = SAAD_DIR / "saad_Re1000_quantities.csv"
FALLBACK_SAAD_TABLE = Path("/mnt/data/saad_quantities.csv")

OUR_NORMALIZED_CSV = DATA_DIR / "quantities_normalized.csv"
OUR_RAW_CSV = DATA_DIR / "quantities.csv"                 # produced by plot_LDC.py
H5_FILE = DATA_DIR / "LDC_Re100_quantities.h5"
OUT_REPORT = DATA_DIR / "compare_quantities_report.csv"
OUT_PNG = DATA_DIR / "compare_quantities.png"
OUT_PDF = DATA_DIR / "compare_quantities.pdf"


def read_our_normalized_quantities(norm_csv_path):
    """
    Read quantities_normalized.csv and return final normalized values:
      Energy_norm, Enstrophy_norm, Palinstropy_norm
    """
    if not norm_csv_path.exists():
        raise FileNotFoundError(f"Normalized quantities CSV not found: {norm_csv_path}")
    with open(norm_csv_path, "r", newline="") as fh:
        rdr = csv.reader(fh)
        hdr = next(rdr)
        rows = [row for row in rdr if row and any(cell.strip() for cell in row)]
    if not rows:
        raise ValueError(f"No data rows in {norm_csv_path}")
    last = rows[-1]
    idx = {name: i for i, name in enumerate(hdr)}
    # attempt common normalized column names, fallback to indices
    e_norm = float(last[idx.get("Energy_norm", idx.get("Energy_norm".capitalize(), 1))]) if "Energy_norm" in idx or "Energy_norm".capitalize() in idx else float(last[1])
    en_norm = float(last[idx.get("Enstrophy_norm", 2)])
    p_norm = float(last[idx.get("Palinstropy_norm", 3)])
    return e_norm, en_norm, p_norm


def read_our_final_quantities(csv_path):
    """(Legacy) Read our quantities.csv and return final energy, enstrophy, palinstropy (floats)."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Our quantities CSV not found: {csv_path}")
    with open(csv_path, "r", newline="") as fh:
        rdr = csv.reader(fh)
        hdr = next(rdr)
        rows = [row for row in rdr if row and any(cell.strip() for cell in row)]
    if not rows:
        raise ValueError(f"No data rows in {csv_path}")
    last = rows[-1]
    # expected header: iteration, Energy, Enstrophy, Palinstropy (but be robust to ordering)
    idx = {name: i for i, name in enumerate(hdr)}
    # fall back to typical column indices if header names not present
    e = float(last[idx.get("Energy", 1)])
    ens = float(last[idx.get("Enstrophy", 2)])
    pal = float(last[idx.get("Palinstropy", 3)])
    return e, ens, pal


def read_saad_table(path):
    """
    Read a Saad quantities CSV with columns:
      Grid, Energy, Enstrophy, Palinstrophy
    and return the values (floats) for the highest-resolution row (last row).
    """
    if not path.exists():
        raise FileNotFoundError(f"Saad quantities CSV not found: {path}")
    with open(path, "r", newline="") as fh:
        rdr = csv.reader(fh)
        hdr = [h.strip() for h in next(rdr)]
        rows = [row for row in rdr if row and any(cell.strip() for cell in row)]
    if not rows:
        raise ValueError(f"No data rows in {path}")
    # prefer last row (assume increasing resolution order like 64x64 ... 512x512)
    grid, energy_s, enstrophy_s, palinstrophy_s = rows[-1][:4]
    # convert palinstrophy possibly expressed in scientific-like formats
    energy = float(energy_s)
    enstrophy = float(enstrophy_s)
    pal = float(palinstrophy_s)
    return energy, enstrophy, pal


def rel_err(a, b):
    if b == 0:
        return float("inf") if a != 0 else 0.0
    return (a - b) / abs(b)


def make_plot(our_vals, ref_vals, out_png, out_pdf):
    """Create a bar plot comparing our final values and Saad reference estimates."""
    labels = ["Energy", "Enstrophy", "Palinstropy"]
    our = np.array(our_vals, dtype=float)
    ref = np.array(ref_vals, dtype=float)

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
    bars1 = ax.bar(x - width/2, our, width, label="Our (normalized)")
    bars2 = ax.bar(x + width/2, ref, width, label="Saad (table)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Value (units as in code)")
    ax.set_title("Comparison of conserved quantities (Our (normalized) vs Saad table)")
    ax.legend()

    # add numeric labels on bars
    def autolabel(bars):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3e}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    autolabel(bars1)
    autolabel(bars2)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    # Prefer the normalized CSV (produced by plot_LDC2.py). If missing, fall back to recompute+normalize from HDF5.
    our_normalized_exists = OUR_NORMALIZED_CSV.exists()
    if our_normalized_exists:
        try:
            our_e_norm, our_en_norm, our_p_norm = read_our_normalized_quantities(OUR_NORMALIZED_CSV)
            print("Using our normalized values from:", OUR_NORMALIZED_CSV)
        except Exception as e:
            print("Failed reading normalized CSV:", e)
            our_normalized_exists = False

    if not our_normalized_exists:
        # fallback: compute raw quantities from HDF5 and normalize by first time step if available
        try:
            # compute raw from HDF5
            e_raw, en_raw, p_raw = compute_fv_quantities_from_hdf5(H5_FILE)
            # try to read first values from quantities.csv to normalize; else normalize by raw values themselves
            try:
                with open(OUR_RAW_CSV, "r", newline="") as fh:
                    rdr = csv.reader(fh)
                    hdr = next(rdr)
                    rows = [row for row in rdr if row and any(cell.strip() for cell in row)]
                if rows:
                    first = rows[0]
                    idx = {name: i for i, name in enumerate(hdr)}
                    e0 = float(first[idx.get("Energy", 1)])
                    en0 = float(first[idx.get("Enstrophy", 2)])
                    p0 = float(first[idx.get("Palinstropy", 3)])
                else:
                    e0 = e_raw if e_raw != 0 else 1.0
                    en0 = en_raw if en_raw != 0 else 1.0
                    p0 = p_raw if p_raw != 0 else 1.0
            except Exception:
                e0 = e_raw if e_raw != 0 else 1.0
                en0 = en_raw if en_raw != 0 else 1.0
                p0 = p_raw if p_raw != 0 else 1.0

            our_e_norm = e_raw / e0
            our_en_norm = en_raw / en0
            our_p_norm = p_raw / p0
            print("Computed our normalized values from HDF5 (fallback).")
        except Exception as e:
            raise SystemExit(f"Failed to obtain our normalized values (no normalized CSV and HDF5 fallback failed): {e}")

    print("Our (normalized) final quantities:")
    print(f"  Energy_norm     = {our_e_norm:.6e}")
    print(f"  Enstrophy_norm  = {our_en_norm:.6e}")
    print(f"  Palinstropy_norm= {our_p_norm:.6e}\n")

    # Determine Saad table path (repo first, fallback to debug copy)
    table_path = SAAD_TABLE
    if not table_path.exists():
        if FALLBACK_SAAD_TABLE.exists():
            table_path = FALLBACK_SAAD_TABLE
            print(f"Using fallback Saad quantities at: {table_path}")
        else:
            raise FileNotFoundError(
                f"Saad quantities CSV not found at {SAAD_TABLE} and no fallback at {FALLBACK_SAAD_TABLE}"
            )

    # Read Saad table (absolute values)
    e_ref, ens_ref, pal_ref = read_saad_table(table_path)
    print("Saad table values (taken from saad_Re1000_quantities.csv):")
    print(f"  Energy     = {e_ref:.6e}")
    print(f"  Enstrophy  = {ens_ref:.6e}")
    print(f"  Palinstropy= {pal_ref:.6e}\n")

    # Comparison: NOTE we're comparing our *normalized* values to Saad's absolute table.
    print("Comparison (our_normalized vs Saad_table):")
    print("-----------------------------------------------------------")
    print(f"Energy:      our_norm = {our_e_norm:.6e}   saad = {e_ref:.6e}   rel.err = {rel_err(our_e_norm, e_ref):+.4%}")
    print(f"Enstrophy:   our_norm = {our_en_norm:.6e}   saad = {ens_ref:.6e}   rel.err = {rel_err(our_en_norm, ens_ref):+.4%}")
    print(f"Palinstropy: our_norm = {our_p_norm:.6e}   saad = {pal_ref:.6e}   rel.err = {rel_err(our_p_norm, pal_ref):+.4%}")
    print("-----------------------------------------------------------")

    # Save report CSV
    with open(OUT_REPORT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "our_normalized", "saad_table", "relative_error"])
        w.writerow(["energy", our_e_norm, e_ref, rel_err(our_e_norm, e_ref)])
        w.writerow(["enstrophy", our_en_norm, ens_ref, rel_err(our_en_norm, ens_ref)])
        w.writerow(["palinstropy", our_p_norm, pal_ref, rel_err(our_p_norm, pal_ref)])
    print(f"\nReport written to: {OUT_REPORT}")

    # Plot if matplotlib is available
    if HAS_MPL:
        try:
            make_plot((our_e_norm, our_en_norm, our_p_norm), (e_ref, ens_ref, pal_ref), OUT_PNG, OUT_PDF)
            print(f"Figures written to: {OUT_PNG} and {OUT_PDF}")
        except Exception as e:
            print("Plotting failed:", e)
    else:
        print("\nmatplotlib not available in this environment. To enable plots,")
        print("install matplotlib in your venv, e.g.:")
        print("  ./.venv/bin/python -m pip install matplotlib")
        print(f"Report CSV is available at: {OUT_REPORT}")


if __name__ == "__main__":
    main()