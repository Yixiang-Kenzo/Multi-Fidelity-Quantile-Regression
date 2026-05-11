# Multi-Fidelity Quantile Regression — paper code

Reproduction code for the paper *Multi-Fidelity Quantile Regression*.

This release contains the cleaned-up, paper-bound code only.  Historical exploration scripts have been excluded.

## Layout

```
.
├── mfqr/                          # the library (importable Python package)
│   ├── core.py                    # TransferQuantileRegressor, build_methods
│   ├── cdf_models.py              # ConditionalCDF_QRF (RF backend) + RFF-GP
│   ├── data_utils.py              # create_mf_split, standardise_separately, conformal_calibrate
│   └── models.py                  # public-API re-exports
│
├── runners/
│   ├── run_experiment.py          # §5.3 real-data experiments (Acrolein, Thymine, o-HBDI,
│   │                              #   Burgers, Formation Energy) for RF and GP backends
│   └── run_synthetic.py           # §5.2.1 + §5.2.2 (Informative + Non-informative regimes);
│                                  #   produces data, transform, CQR plots end-to-end
│
├── figures/
│   ├── plot_misinformative.py     # §5.2.3 Misinformative regime: data + 4 CQR plots
│   │                              #   (HF-Only, MFQR, MFQR+OS, MFQR+MS).
│   │                              #   MFQR+MS uses CV-pinball-selected m over m=2..5.
│   ├── plot_boxplots.py           # §5.3 boxplots (5 RF main + 5 GP appendix) from jsonls
│   └── gen_table.py               # tab:results numbers for the paper table
│
├── data_generators/
│   ├── burgers_mf_dataset.py      # Generates burgers_mf_*.npz datasets
│   └── preprocess_materials_project.py
│                                  # Data card + CSV-to-NPZ converter and MP-API rebuilder
│                                  #   for the shipped Materials Project dataset
│
├── data/                          # Curated small datasets shipped directly with the repo
│   ├── materials_project_real_data.csv
│   │                              #   (full source CSV with material_id per row, 354 KB)
│   └── materials_project_formation_energy.npz
│                                  #   (binary cache the runner loads, 255 KB)
│                                  #
│                                  # QeMFi (Acrolein/Thymine/o-HBDI) — too large, downloaded
│                                  #   from Zenodo (see Datasets section).
│                                  # Burgers PDE — generated locally via the script in
│                                  #   data_generators/, no shipped copy.
│
├── paper_figures/                 # The 27 PNGs that appear in the paper
│
├── results/                       # Canonical jsonls produced by the experiments
│   ├── rf_results/                #   (5 RF datasets — §5.3 main paper)
│   └── gp_results/                #   (5 GP datasets — appendix)
│
├── README.md                      # this file
├── LICENSE                        # all rights reserved (under review)
├── requirements.txt               # minimum dependency versions (default install)
└── requirements-frozen.txt        # exact pinned versions used to test the paper
                                   #   (fallback if requirements.txt produces mismatches)
```

## Installation

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

If you hit version-mismatch errors, install the exact versions the paper was tested against:

```bash
pip install -r requirements-frozen.txt
```

## Datasets

The paper uses five datasets in §5.3.  The QeMFi and Materials Project datasets are
downloaded from public sources; the Burgers dataset is generated locally.

| Dataset | Source | Notes |
|---|---|---|
| **QeMFi** (Acrolein, Thymine, o-HBDI) | [Zenodo: 10.5281/zenodo.10782873](https://zenodo.org/records/10782873) | Vinod et al. 2025.  Contains all 9 QeMFi molecules; we use Acrolein, Thymine, and o-HBDI. |
| **Materials Project Formation Energy** | **Shipped in repo**: `data/materials_project_real_data.csv` (full row-level source, with `material_id` per row) and `data/materials_project_formation_energy.npz` (binary cache loaded by the runner).  Querying derived from the [Materials Project](https://next-gen.materialsproject.org/) API. | LF = PBE formation energy per atom; HF = r²SCAN formation energy per atom.  11 features: band_gap, density, volume, nsites, and a 7-way one-hot of crystal system.  N = 2500.  See `data_generators/preprocess_materials_project.py` for the CSV→NPZ converter and an MP-API rebuild script. |
| **Burgers PDE** | Generated locally via `data_generators/burgers_mf_dataset.py` | See below. |

Once downloaded (and pre-processed for Materials Project), place the `.npz` files
anywhere on disk; runners take `--dataset <path>` as a CLI argument.

### Burgers data generation

```bash
python data_generators/burgers_mf_dataset.py --N 5000 --output burgers_mf_N5000.npz
```

## How each paper figure / table is regenerated

### §5.2.1 Informative LF regime + §5.2.2 Non-informative LF regime
Produces all `Informative_*.png` and `Noninformative_*.png` files (data, transform,
and CQR plots for each method).

```bash
python runners/run_synthetic.py --experiment all --output-dir <out-dir>
```

(Internally, `run_synthetic.py` tags the Non-informative experiment as `Break2`
with the legacy label "Misinformative".  The Non-informative output PNGs are
renamed to `Noninformative_*.png` to match the published paper's naming.)

### §5.2.3 Misinformative regime
Produces `Misinformative_data.png`, `Misinformative_cqr_{HF-Only, MFQR, MFQR+OS, MFQR+MS}.png`.
The MFQR+MS plot uses the m selected by cross-fitted pinball loss (paper reports m=5).

```bash
python figures/plot_misinformative.py --output-dir <out-dir>
```

### §5.3 Scientific data
Run experiments first (one command per dataset/backend; produces a jsonl per call).
The 5-method specification matches the paper exactly:

```bash
# Example: Acrolein RF, 20 seeds
python runners/run_experiment.py \
  --backend rf \
  --dataset /path/to/QeMFi_acrolein.npz \
  --alpha 0.10 \
  --seeds 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 \
  --test-frac 0.4 \
  --methods "HF-Only,HF-Only (augment),HF-Only (offset),Transfer (offset),Transfer (offset) + 1Step" \
  --one-step-tune-gamma \
  --one-step-gamma-grid "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0" \
  --one-step-density-source hf \
  --one-step-density-floor 0.1 \
  --output-dir results/rf_results

# Burgers small-HF stress test: n_H is deliberately 60, and we decouple
# the calibration set to n_cal=300 so the conformal margin is stable.
# Use these extra flags:  --test-frac 0.856 --n-hf-override 60 --n-cal-override 300
#
# GP backend: --backend gp  (everything else identical)
```

Then generate boxplots from the resulting jsonls:

```bash
python figures/plot_boxplots.py --results-dir results --output-dir <out-dir>
```
The `--results-dir` should contain `rf_results/` and `gp_results/` subdirectories with one
jsonl per dataset, named e.g. `QeMFi_acrolein_rf_results.jsonl`.

Table values for tab:results are computed from the same jsonls via `gen_table.py`.

### Appendix (GP backend)
Same as §5.3 but with `--backend gp`.  `plot_boxplots.py` produces the five `boxplot_*_GP.png`
files automatically when both `rf_results/` and `gp_results/` are present.

## Reproducibility note

Per-run results may differ at the level of ~1% in aggregate width/coverage due
to floating-point non-determinism in scikit-learn's parallel random forest
implementation
([sklearn glossary on `random_state`](https://scikit-learn.org/stable/glossary.html#term-random_state)).
Qualitative conclusions, ranked method ordering, and bolded best-method cells
in the tables are stable across runs.  Reported numbers in the paper are
averages over 20 random seeds; aggregate statistics are robust to this
parallelism-induced noise.

Setting `n_jobs=1` in `mfqr.core._make_mean_model` would make results
bit-deterministic at significant runtime cost; the paper's numbers were
generated with default `n_jobs=-1` (all cores).

## License

All rights reserved.  The paper is currently under review; this code is shared
for reviewer inspection and reproducibility verification.  A permissive open-source
license will be added once the paper is accepted.

## Citation

Paper: [arXiv:2605.10406](https://arxiv.org/abs/2605.10406)

```bibtex
@article{liu2026mfqr,
  title         = {Multi-Fidelity Quantile Regression},
  author        = {Liu, Yixiang and Zhang, Yao},
  year          = {2026},
  eprint        = {2605.10406},
  archivePrefix = {arXiv},
  primaryClass  = {stat.ML},
  url           = {https://arxiv.org/abs/2605.10406},
}
```
