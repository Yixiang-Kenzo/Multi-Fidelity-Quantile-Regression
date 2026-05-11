"""Preprocessing for the Materials Project Formation Energy dataset (Section 5.3).

The shipped files
    data/materials_project_real_data.csv             (source, 2500 rows)
    data/materials_project_formation_energy.npz      (cached binary)
are the canonical inputs used in the paper.  This script:

  1. ``csv_to_npz``   converts the CSV to the NPZ array layout the runner expects.
                      This step is exact and verified --- running it on the shipped
                      CSV reproduces the shipped NPZ row-for-row, with zero diff.

  2. ``rebuild_csv``  re-queries the Materials Project API for the 2500 material_ids
                      listed in the shipped CSV and rebuilds the same row format.
                      Use this if you want a fully independent reproduction from MP.

------------------------------------------------------------------
CSV / NPZ SCHEMA
------------------------------------------------------------------
CSV columns (in order):
    band_gap, density, volume, nsites,                    # 4 numerical features
    crystal_Cubic, crystal_Hexagonal, crystal_Monoclinic, # 7-way crystal_system
    crystal_Orthorhombic, crystal_Tetragonal,             #   one-hot indicator
    crystal_Triclinic, crystal_Trigonal,
    formation_energy_pbe,                                 # Y_lf  (low fidelity)
    formation_energy_r2scan,                              # Y_hf  (high fidelity)
    material_id, formula                                  # identifiers (not features)

NPZ arrays:
    X    : float64, shape (2500, 11)  -- first 11 CSV columns
    Y_lf : float64, shape (2500,)     -- formation_energy_pbe
    Y_hf : float64, shape (2500,)     -- formation_energy_r2scan

------------------------------------------------------------------
USAGE
------------------------------------------------------------------
# Rebuild the NPZ from the shipped CSV (no API key needed):
python data_generators/preprocess_materials_project.py csv_to_npz \\
    --csv     data/materials_project_real_data.csv \\
    --output  data/materials_project_formation_energy.npz

# Re-query MP to verify the CSV from scratch (requires MP_API_KEY env var):
export MP_API_KEY="your-key"
python data_generators/preprocess_materials_project.py rebuild_csv \\
    --reference data/materials_project_real_data.csv \\
    --output    data/materials_project_real_data_rebuilt.csv

# Compare a rebuilt CSV against the shipped one:
python data_generators/preprocess_materials_project.py compare_csv \\
    --a data/materials_project_real_data.csv \\
    --b data/materials_project_real_data_rebuilt.csv
"""
import argparse
import os
import sys

import numpy as np


FEATURE_COLS = [
    "band_gap", "density", "volume", "nsites",
    "crystal_Cubic", "crystal_Hexagonal", "crystal_Monoclinic",
    "crystal_Orthorhombic", "crystal_Tetragonal",
    "crystal_Triclinic", "crystal_Trigonal",
]
Y_LF_COL = "formation_energy_pbe"
Y_HF_COL = "formation_energy_r2scan"


def csv_to_npz(csv_path, npz_path):
    """Convert the shipped CSV into the NPZ layout the runner expects."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    missing = [c for c in FEATURE_COLS + [Y_LF_COL, Y_HF_COL] if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: CSV is missing required columns: {missing}")
    X = df[FEATURE_COLS].to_numpy(dtype=np.float64)
    Y_lf = df[Y_LF_COL].to_numpy(dtype=np.float64)
    Y_hf = df[Y_HF_COL].to_numpy(dtype=np.float64)
    np.savez(npz_path, X=X, Y_lf=Y_lf, Y_hf=Y_hf)
    print(f"  {csv_path}  ({len(df)} rows)")
    print(f"  --> {npz_path}  X={X.shape}  Y_lf={Y_lf.shape}  Y_hf={Y_hf.shape}")
    print(f"  r(Y_lf, Y_hf) = {np.corrcoef(Y_lf, Y_hf)[0, 1]:.4f}")


def _require_api_key():
    key = os.environ.get("MP_API_KEY")
    if not key:
        sys.exit(
            "ERROR: MP_API_KEY environment variable not set.\n"
            "  Get a key at https://next-gen.materialsproject.org/api , then:\n"
            '    export MP_API_KEY="your-key"   (bash)\n'
            '    setx   MP_API_KEY "your-key"   (Windows; new terminal afterwards)\n'
        )


def rebuild_csv(reference_csv, output_csv):
    """Re-query Materials Project for the material_ids in `reference_csv` and
    rebuild the same row format.  Useful if you want to verify the shipped
    CSV from scratch using the live MP API.

    Requires MP_API_KEY environment variable and ``mp-api`` installed.
    """
    _require_api_key()
    import pandas as pd
    try:
        from mp_api.client import MPRester
    except ImportError:
        sys.exit("ERROR: mp-api not installed.  Run:  pip install mp-api pymatgen")

    ref = pd.read_csv(reference_csv)
    if "material_id" not in ref.columns:
        sys.exit(f"ERROR: {reference_csv} has no 'material_id' column.")
    mids = ref["material_id"].tolist()
    print(f"  re-querying MP for {len(mids)} material_ids from {reference_csv} ...")

    crystal_levels = ["Cubic", "Hexagonal", "Monoclinic", "Orthorhombic",
                      "Tetragonal", "Triclinic", "Trigonal"]

    rows = []
    with MPRester() as mpr:
        # Pull base properties for the listed materials.  The exact API
        # path may differ between mp-api versions; this targets the
        # widely-supported summary endpoint.
        docs = mpr.materials.summary.search(
            material_ids=mids,
            fields=[
                "material_id", "formula_pretty", "nsites",
                "band_gap", "density", "volume", "symmetry",
                "formation_energy_per_atom",
            ],
        )
        pbe_by_id = {d.material_id: d for d in docs}

        # r2scan formation energies (separate endpoint in mp-api):
        try:
            r2_docs = mpr.materials.r2scan.search(
                material_ids=mids,
                fields=["material_id", "formation_energy_per_atom"],
            )
        except AttributeError:
            sys.exit(
                "ERROR: mpr.materials.r2scan endpoint not available in your "
                "mp-api version.  Inspect dir(mpr.materials) for the correct "
                "endpoint name (was '.r2scan' at time of writing).\n"
            )
        r2_by_id = {d.material_id: d.formation_energy_per_atom for d in r2_docs}

    print(f"  got {len(pbe_by_id)} PBE entries and {len(r2_by_id)} r2scan entries")

    for mid in mids:
        if mid not in pbe_by_id or mid not in r2_by_id:
            continue
        d = pbe_by_id[mid]
        cs = str(d.symmetry.crystal_system).capitalize() if d.symmetry else None
        row = {
            "band_gap": float(d.band_gap or 0.0),
            "density":  float(d.density),
            "volume":   float(d.volume),
            "nsites":   int(d.nsites),
        }
        for cl in crystal_levels:
            row[f"crystal_{cl}"] = (cs == cl)
        row["formation_energy_pbe"]    = float(d.formation_energy_per_atom)
        row["formation_energy_r2scan"] = float(r2_by_id[mid])
        row["material_id"]             = mid
        row["formula"]                 = d.formula_pretty
        rows.append(row)

    out_df = pd.DataFrame(rows, columns=ref.columns.tolist())
    out_df.to_csv(output_csv, index=False)
    print(f"  saved {output_csv}  ({len(out_df)} rows)")
    if len(out_df) < len(ref):
        print(f"  WARNING: {len(ref) - len(out_df)} material_ids were missing from "
              f"the API response (possibly retired entries).")


def compare_csv(a_path, b_path):
    """Compare two CSVs row-by-row, joined on material_id."""
    import pandas as pd
    a = pd.read_csv(a_path).set_index("material_id")
    b = pd.read_csv(b_path).set_index("material_id")
    common = a.index.intersection(b.index)
    print(f"  {a_path}: {len(a)} rows  |  {b_path}: {len(b)} rows  |  common: {len(common)}")
    if not len(common):
        return
    a = a.loc[common]; b = b.loc[common]

    print()
    print(f"  {'column':<30s}  {'max abs diff':>14s}")
    print("  " + "-" * 48)
    for c in FEATURE_COLS + [Y_LF_COL, Y_HF_COL]:
        if c in a.columns and c in b.columns:
            ac = a[c].to_numpy(); bc = b[c].to_numpy()
            try:
                d = float(np.max(np.abs(ac.astype(float) - bc.astype(float))))
                print(f"  {c:<30s}  {d:>14.2e}")
            except Exception as e:
                print(f"  {c:<30s}  not numeric ({e})")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("csv_to_npz", help="Convert CSV to the runner's NPZ layout (no API key needed).")
    p1.add_argument("--csv",    default="data/materials_project_real_data.csv")
    p1.add_argument("--output", default="data/materials_project_formation_energy.npz")

    p2 = sub.add_parser("rebuild_csv", help="Re-query MP API for the material_ids in --reference and rebuild the CSV.")
    p2.add_argument("--reference", default="data/materials_project_real_data.csv")
    p2.add_argument("--output",    default="data/materials_project_real_data_rebuilt.csv")

    p3 = sub.add_parser("compare_csv", help="Compare two CSVs column-by-column on shared material_ids.")
    p3.add_argument("--a", required=True)
    p3.add_argument("--b", required=True)

    args = parser.parse_args()
    if args.cmd == "csv_to_npz":
        csv_to_npz(args.csv, args.output)
    elif args.cmd == "rebuild_csv":
        rebuild_csv(args.reference, args.output)
    elif args.cmd == "compare_csv":
        compare_csv(args.a, args.b)


if __name__ == "__main__":
    main()
