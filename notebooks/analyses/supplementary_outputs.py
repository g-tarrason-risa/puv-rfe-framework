"""
Supplementary outputs for the reviewer response.

Generates, from the cached rfe_eval pickles in data/int/, two deliverables:

  Task 1 (Reviewer 2, comment 6) - ROC-AUC reported alongside PR-AUC:
      * per-model ROC-AUC bar charts (mean +/- SEM vs number of features), styled as
        drop-in companions to the existing PR-AUC figures, and
      * a table of ROC-AUC / PR-AUC at the selected model, for PURK-style comparison.

  Task 3 (Reviewer 2, major comment 1) - model stability across folds:
      * a per-fold metrics table at the selected model,
      * a figure showing fold-to-fold spread of PR-AUC and ROC-AUC, and
      * a feature-selection frequency table (how often each feature is picked across folds).

The "selected model" is the k=1 model, matching the `best_k = 1` choice hardcoded in
every analysis notebook. Change SELECTED_K below if that ever changes.

Reads only the pickles (no sklearn / no raw data needed); safe to run anywhere.
Any pickle that is temporarily missing (e.g. mid-regeneration) is skipped with a warning.
"""
from __future__ import annotations
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 15, "axes.labelsize": 13,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
})

BASE = Path(__file__).resolve().parents[2]
INT = BASE / "data" / "int"
RES = BASE / "results" / "supplementary"
RES.mkdir(parents=True, exist_ok=True)

SELECTED_K = 1  # matches `best_k = 1` in the analysis notebooks

# (display label, filename stem, short axis code, filename code) in display order
DATASETS = [
    ("Bladder dysfunction", "bladder_set_rfe", "Bladder", "bladder"),
    ("Renal - infants (TUR <= 12 mo)", "renal_set_infants_rfe", "Infants", "infants"),
    ("Renal - Sadia", "renal_set_sadia_rfe", "Sadia", "sadia"),
    ("Renal - Sadia (no creatinine)", "renal_set_sadia_nocrea_rfe", "Sadia-nc", "sadia_nocrea"),
]
ESTIMATORS = {"lr_l2": "LR-L2", "lr_np": "LR-NP", "gbc": "GBC", "rf": "RF"}

# ---- project figure style (matches the PR-AUC / feature-importance plots) -----
BAR_KW = dict(color="skyblue", edgecolor="black", alpha=0.8, capsize=4)
JITTER_KW = dict(color="grey", alpha=0.5, zorder=3)


def load(stem: str, est: str):
    """Return the results dict, or None if the pickle is not present yet."""
    p = INT / f"{stem}_{est}.pkl"
    if not p.exists():
        print(f"  ! skipping {p.name} (not found)")
        return None
    with p.open("rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Task 1a - per-model ROC-AUC bar chart (one file per model, PR-AUC figure style)
# ---------------------------------------------------------------------------
def roc_auc_per_model() -> None:
    out_dir = RES / "roc_auc"
    out_dir.mkdir(exist_ok=True)
    for label, stem, _short, code in DATASETS:
        for est, est_label in ESTIMATORS.items():
            d = load(stem, est)
            if d is None:
                continue
            summary = d["Summary"].sort_values("N_Features")
            per_fold = d["PerFold"]

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(summary["N_Features"], summary["ROC_AUC_Mean"],
                   yerr=np.nan_to_num(summary["ROC_AUC_SEM"].to_numpy()),
                   label="Mean ± SEM", **BAR_KW)

            # jittered fold-level points
            xs, ys = [], []
            for k, folds in per_fold.items():
                jitter = np.random.uniform(-0.15, 0.15, size=len(folds))
                xs.extend(k + jitter)
                ys.extend([f["ROC_AUC"] for f in folds])
            ax.scatter(xs, ys, label="Fold scores (jittered)", **JITTER_KW)

            # mean (black) and ±SEM (red) labels above each bar
            for x, mean, sem in zip(summary["N_Features"],
                                    summary["ROC_AUC_Mean"], summary["ROC_AUC_SEM"]):
                base_y = mean + (0 if np.isnan(sem) else sem) + 0.1
                ax.text(x, base_y + 0.12, f"{mean:.2f}", ha="center", va="bottom",
                        fontsize=11, color="black")
                if not np.isnan(sem):
                    ax.text(x, base_y, f"±{sem:.2f}", ha="center", va="bottom",
                            fontsize=11, color="red")

            ax.grid(False)
            ax.set_ylim(0, 1.5)
            ax.set_yticks(np.arange(0, 1.1, 0.2))
            ax.set_xticks(summary["N_Features"])
            ax.set_xlabel("# features")
            ax.set_ylabel("ROC-AUC")
            ax.set_title(f"ROC-AUC by model feature count\n{label} — {est_label}")
            ax.legend(loc="lower center", framealpha=1.0)
            fig.tight_layout()
            out = out_dir / f"{code}_rfe_{est}_roc_auc.png"
            fig.savefig(out, dpi=300)
            plt.close(fig)
            print("wrote", out.relative_to(BASE))


# ---------------------------------------------------------------------------
# Task 1b - ROC-AUC / PR-AUC at the selected model (table)
# ---------------------------------------------------------------------------
def selected_model_metrics_table() -> pd.DataFrame:
    rows = []
    for label, stem, _short, _code in DATASETS:
        for est, est_label in ESTIMATORS.items():
            d = load(stem, est)
            if d is None:
                continue
            s = d["Summary"]
            r = s.loc[s["N_Features"] == SELECTED_K].iloc[0]
            rows.append({
                "Dataset": label, "Estimator": est_label, "N_Features": SELECTED_K,
                "ROC_AUC_Mean": round(r["ROC_AUC_Mean"], 3), "ROC_AUC_SEM": round(r["ROC_AUC_SEM"], 3),
                "PR_AUC_Mean": round(r["PR_AUC_Mean"], 3), "PR_AUC_SEM": round(r["PR_AUC_SEM"], 3),
                "BalancedAccuracy_Mean": round(r["BalancedAccuracy_Mean"], 3),
                "BalancedAccuracy_SEM": round(r["BalancedAccuracy_SEM"], 3),
            })
    df = pd.DataFrame(rows)
    out = RES / "metrics_at_selected_model.csv"
    df.to_csv(out, index=False)
    print("wrote", out.relative_to(BASE))
    return df


# ---------------------------------------------------------------------------
# Task 3a - per-fold metrics at the selected model (table)
# ---------------------------------------------------------------------------
METRICS = ["PR_AUC", "ROC_AUC", "BalancedAccuracy"]


def perfold_table() -> pd.DataFrame:
    rows = []
    for label, stem, _short, _code in DATASETS:
        for est, est_label in ESTIMATORS.items():
            d = load(stem, est)
            if d is None:
                continue
            folds = d["PerFold"][SELECTED_K]
            for metric in METRICS:
                vals = [f[metric] for f in folds]
                row = {"Dataset": label, "Estimator": est_label, "Metric": metric}
                row.update({f"fold{f['fold']}": round(f[metric], 3) for f in folds})
                row["Mean"] = round(float(np.nanmean(vals)), 3)
                row["SD"] = round(float(np.nanstd(vals, ddof=1)), 3)
                rows.append(row)
    df = pd.DataFrame(rows)
    out = RES / "perfold_metrics_at_selected_model.csv"
    df.to_csv(out, index=False)
    print("wrote", out.relative_to(BASE))
    return df


# ---------------------------------------------------------------------------
# Task 3b - fold-to-fold spread of PR-AUC and ROC-AUC (bar + jittered folds)
# ---------------------------------------------------------------------------
def stability_figure() -> None:
    labels, groups = [], []
    for label, stem, short, _code in DATASETS:
        for est, est_label in ESTIMATORS.items():
            d = load(stem, est)
            if d is None:
                continue
            labels.append(f"{short}\n{est_label}")
            groups.append(d["PerFold"][SELECTED_K])
    if not groups:
        print("  ! stability_figure: no models available, skipped")
        return

    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    rng = np.random.default_rng(0)
    for ax, metric in zip(axes, ["PR_AUC", "ROC_AUC"]):
        means = [float(np.nanmean([f[metric] for f in g])) for g in groups]
        sems = [float(np.nanstd([f[metric] for f in g], ddof=1) / np.sqrt(len(g))) for g in groups]
        ax.bar(x, means, yerr=sems, label="Mean ± SEM", **BAR_KW)
        for i, g in enumerate(groups):
            vals = np.array([f[metric] for f in g], dtype=float)
            ax.scatter(i + rng.uniform(-0.15, 0.15, vals.size), vals, **JITTER_KW)
        ax.set_ylabel(metric.replace("_", "-"))
        ax.set_ylim(0, 1.05)
        ax.grid(False)
    axes[0].set_title(f"Per-fold stability at the selected model (k={SELECTED_K}); "
                      "bar = mean ± SEM, points = folds")
    axes[0].legend(loc="upper left", framealpha=1.0)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, fontsize=10)
    fig.tight_layout()
    out = RES / "perfold_stability.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("wrote", out.relative_to(BASE))


# ---------------------------------------------------------------------------
# Task 3c - feature-selection frequency across folds (table)
# ---------------------------------------------------------------------------
def selection_frequency_table() -> pd.DataFrame:
    rows = []
    for label, stem, _short, _code in DATASETS:
        for est, est_label in ESTIMATORS.items():
            d = load(stem, est)
            if d is None:
                continue
            per_fold_sel = d["Selected_Features"][SELECTED_K]
            n_folds = len(per_fold_sel)
            counts: dict[str, int] = {}
            for sel in per_fold_sel:
                for feat in sel:
                    counts[feat] = counts.get(feat, 0) + 1
            for feat, c in sorted(counts.items(), key=lambda kv: -kv[1]):
                rows.append({"Dataset": label, "Estimator": est_label,
                             "Feature": feat, "Folds_selected": f"{c}/{n_folds}"})
    df = pd.DataFrame(rows)
    out = RES / "feature_selection_frequency_at_selected_model.csv"
    df.to_csv(out, index=False)
    print("wrote", out.relative_to(BASE))
    return df


if __name__ == "__main__":
    print(f"Reading pickles from {INT.relative_to(BASE)} | selected k = {SELECTED_K}\n")
    roc_auc_per_model()
    selected_model_metrics_table()
    perfold_table()
    stability_figure()
    selection_frequency_table()
    print("\nAll supplementary outputs written to", RES.relative_to(BASE))
