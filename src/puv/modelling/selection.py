# src/puv/modelling/selection.py
"""Parsimony-aware selection of the RFE feature count (k).

Implements the rule described in the manuscript. A repeated-measures ANOVA of
PR-AUC across candidate feature counts (folds treated as the within-subject
factor) gates the decision: if no subset size differs significantly, the
single-feature model is chosen. Otherwise the highest-mean-PR-AUC subset is
compared against the best-performing *smaller* subset with a paired Wilcoxon
signed-rank test, and the larger set is retained only if it is significantly
better; failing that, the choice falls back to the parsimonious single-feature
model.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def _perfold_long(results: Dict[str, Any], score: str) -> pd.DataFrame:
    """Long-form (k, fold, score) frame from an rfe_eval results dict."""
    per = results["PerFold"]
    rows: List[Dict[str, float]] = []
    for k, folds in per.items():
        for fold_idx, fold in enumerate(folds):
            rows.append({"k": int(k), "fold": fold_idx, "score": float(fold[score])})
    return pd.DataFrame(rows)


def _rm_anova_pvalue(long_df: pd.DataFrame) -> float:
    """Repeated-measures ANOVA p-value for score ~ k, folds as subjects."""
    try:
        from statsmodels.stats.anova import AnovaRM
    except ImportError as exc:  # pragma: no cover - env guard
        raise ImportError(
            "select_best_k requires statsmodels for the repeated-measures ANOVA. "
            "It ships with Anaconda; otherwise `pip install statsmodels`."
        ) from exc
    fit = AnovaRM(long_df, depvar="score", subject="fold",
                  within=["k"], aggregate_func="mean").fit()
    return float(fit.anova_table["Pr > F"].iloc[0])


def select_best_k(results: Dict[str, Any], alpha: float = 0.05,
                  score: str = "PR_AUC") -> int:
    """Choose the RFE feature count with the ANOVA -> Wilcoxon parsimony rule.

    Parameters
    ----------
    results : dict
        Dictionary returned by ``rfe_eval`` (must contain ``PerFold``).
    alpha : float
        Significance threshold for both the ANOVA gate and the Wilcoxon test.
    score : str
        Per-fold metric key to select on (default ``"PR_AUC"``).

    Returns
    -------
    int
        Selected number of features. Defaults to the single-feature model
        unless a larger subset is significantly better than the best-performing
        smaller subset.
    """
    long_df = _perfold_long(results, score)
    ks = sorted(int(k) for k in long_df["k"].unique())
    if len(ks) <= 1:
        return ks[0] if ks else 1
    floor = 1 if 1 in ks else ks[0]

    # Stage 1 - ANOVA gate: is any subset size different at all?
    p_anova = _rm_anova_pvalue(long_df)
    if not (np.isfinite(p_anova) and p_anova < alpha):
        return floor

    means = long_df.groupby("k")["score"].mean()
    k_top = int(means.idxmax())
    if k_top == floor:
        return floor

    # Stage 2 - best subset vs best-performing smaller subset.
    smaller = [k for k in ks if k < k_top]
    k_small = int(means.loc[smaller].idxmax())
    a = long_df[long_df["k"] == k_top].sort_values("fold")["score"].to_numpy()
    b = long_df[long_df["k"] == k_small].sort_values("fold")["score"].to_numpy()
    if np.allclose(a, b):
        return floor
    mask = ~np.isclose(a - b, 0)
    p_wilcoxon = wilcoxon(a[mask], b[mask], zero_method="wilcox",
                          alternative="two-sided").pvalue
    if p_wilcoxon < alpha and means[k_top] > means[k_small]:
        return k_top
    return floor
