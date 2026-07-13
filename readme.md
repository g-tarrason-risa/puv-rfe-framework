# PUV — Posterior Urethral Valves prognostic modelling (Jaroy–Lundar 2025)

Clinical-research code for predicting **renal** and **bladder** function outcomes in
children treated for posterior urethral valves (PUV). It uses recursive feature
elimination (RFE) with **nested cross-validation** to identify early, interpretable
clinical predictors in a small, class-imbalanced single-centre cohort.

The deliverable is the analysis itself — the notebooks and the figures/tables they
export to `results/`. There is no application or service.

## Publication

> **TODO — add on acceptance.**
> DOI: [`10.XXXX/XXXXXX`](https://doi.org/10.XXXX/XXXXXX)

If you use this code, please cite:

```bibtex
@article{jaroy_lundar_puv_2025,
  title   = {TODO: article title},
  author  = {Jaroy, Lundar, Risa and others},
  journal = {TODO: journal},
  year    = {2025},
  doi     = {10.XXXX/XXXXXX}
}
```

## Repository layout

```
src/puv/                 # installable modelling package (see "Package" below)
notebooks/
  prep/                  # prep.ipynb — raw data -> per-analysis datasets
  eda/                   # completeness, distributions, creatinine exploration
  analyses/              # one notebook per <dataset> x <estimator>, + supplementary_outputs.py
Makefile                 # regenerate the whole pipeline (devcontainer only)
pyproject.toml           # package metadata (editable install)
data/        (gitignored) # raw + prepped patient data — never committed
results/     (gitignored) # generated figures/tables
.model_cache/(gitignored) # optional joblib cache used by rfe_eval
```

## Data & privacy

**This repository contains code only.** The cohort is sensitive patient data and is
**not** distributed here: no raw records, prepped datasets, or generated outputs are
included, and the notebooks are published with all cell outputs stripped (so no data
is embedded in them). `data/` and `results/` are also **git-ignored** as a second
safeguard.

To reproduce the analysis you must supply your own raw file at
`data/raw/2025_puv_jaroy-lundar_raw-data.xlsx`; running the pipeline then recreates
everything under `data/` and `results/` locally. The data cannot be shared by the
authors for privacy reasons.

## Setup

The project targets an Anaconda/conda base environment (see `.devcontainer/`). The
package is installed in editable mode so notebooks can `import puv`:

```bash
pip install -e .
```

Dependencies are intentionally **not pinned** in `pyproject.toml` — the base
scientific stack is assumed present via conda. The one non-default dependency is
[`dython`](https://pypi.org/project/dython/) (used in `puv.visualise`), installed
via conda-forge in the devcontainer's `postCreateCommand`.

Notebook diffs are kept clean with an [`nbstripout`](https://github.com/kynan/nbstripout)
git filter (`.gitattributes`: `*.ipynb filter=nbstripout`); the devcontainer runs
`nbstripout --install` automatically. Outside the devcontainer, run it once locally
so committed notebooks don't carry output/execution-count noise.

### Path convention (important)

Every notebook hardcodes the project root:

```python
BASE = Path('/workspaces/CODESPACE/').resolve()
```

This only resolves inside the devcontainer (or a checkout at exactly that path).
**The pipeline is devcontainer-only**; adjust this line per-notebook to run elsewhere.

## Reproducing the analysis

A `Makefile` orchestrates the full pipeline in dependency order
(`prep → eda → analyses → supplementary`). **Run inside the devcontainer.**

```bash
make rerun              # refresh everything from cached pickles (fast; no recompute)
make clean-pickles      # remove data/int/*.pkl and .model_cache/ (force recompute)
make clean-results      # remove all files under results/ (keeps the folder tree)
make rebuild            # clean pickles + results, then full recompute from scratch
make help               # list all targets
```

`rfe_eval` caches each model's results as a pickle in `data/int/`; a notebook with an
existing pickle reloads it instead of recomputing, so `make rerun` only refreshes the
plots. Delete the pickle (or `make clean-pickles`) to recompute.

## Data flow

1. **`notebooks/prep/prep.ipynb`** — reads the raw Excel, converts dates to age-in-months,
   drops/renames variables, and writes per-analysis datasets to `data/prep/*.xlsx`:
   `renal_set_infants` (first TUR ≤ 12 mo), `renal_set_children` (> 12 mo),
   `renal_set_ckd5`, and `bladder_set`.
2. **`notebooks/eda/`** — completeness, distributions, and creatinine decision boundaries;
   figures/tables to `results/eda/`.
3. **`notebooks/analyses/`** — one notebook per **dataset × estimator**, each calling
   `rfe_eval(...)` and plotting PR-AUC / ROC-AUC / balanced accuracy / confusion matrix /
   feature importances (or coefficients) to `results/<dataset>/`.
   `supplementary_outputs.py` aggregates the pickles into the reviewer supplementary
   figures/tables (`results/supplementary/`).

**Datasets:** `bladder`, `renal/infants`, `renal/sadia`, `renal/sadia_nocrea`
(the `sadia_nocrea` variant excludes creatinine-derived features).
**Estimators:** `gbc` (GradientBoosting), `lr-l2` (L2 logistic), `lr-np` (unpenalised
logistic), `rf` (RandomForest).

## Package (`src/puv/`)

- **`puv.prep`** — data-cleaning utilities (per-column completeness filtering).
- **`puv.wrangle`** — small DataFrame helpers (e.g. averaging per-fold importances).
- **`puv.visualise`** — EDA plotting helpers (correlation heatmaps, distribution grids).
- **`puv.modelling`** — the core ML pipeline:
  - `impute.GowerKNNImputer` — KNN imputation using Gower distance over mixed
    numeric/categorical features; imputes against a fitted training "anchor".
  - `scaling.NormalityAwareScaler` — per-column scaler picking Standard vs Robust scaling
    from a Shapiro–Wilk test. (The paper models pass an explicit single `StandardScaler`
    for cross-fold comparability of coefficients.)
  - `preprocess._preprocessor` — wires imputation → scaling → one-hot encoding into one
    leakage-safe `Pipeline`.
  - `rfe.rfe_eval` — the main entry point: leakage-safe RFE with nested CV, optional
    inner-CV hyperparameter search, and aggregation of accuracy / balanced-accuracy /
    F1 / ROC-AUC / PR-AUC, threshold analysis, confusion matrices, and per-fold feature
    importances. Returns one results `dict`, pickled to `data/int/`.
  - `utils` — column-type inference, positive-label resolution, standard-error helpers.

## Tooling notes

There is no test suite, linter, or formatter configured — this is a research codebase.

## License

MIT (see `pyproject.toml`).
