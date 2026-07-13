# src/puv/modelling/impute.py
from __future__ import annotations
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class GowerKNNImputer(BaseEstimator, TransformerMixin):
    """
    KNN imputer using Gower distance for mixed data (numeric + categorical).

    - Computes pairwise Gower distances between samples (handling numeric and categorical features differently).
    - Uses k nearest neighbours from the *training anchor* to fill in missing values.
    - Can weight neighbour contributions by inverse distance ("distance") or equally ("uniform").

    Parameters
    ----------
    n_neighbors : int, default=3
        Number of nearest neighbours to consider when imputing.

    weights : {'uniform','distance'}, default='distance'
        How to weight neighbour contributions when imputing.

    numeric_cols, categorical_cols : list[str], optional
        Explicit lists of numeric and categorical columns. If None, inferred at fit().

    feature_weights : dict[str,float], optional
        Per-feature weights to scale their contribution to the Gower distance.

    fallback_numeric : {'median','mean'}, default='median'
        How to fill a numeric feature if neighbours have no valid (non-NaN) values.

    fallback_categorical : {'most_frequent'}, default='most_frequent'
        How to fill a categorical feature if neighbours have no valid values.

    random_state : int, optional
        For reproducibility (not currently used).
    """

    def __init__(
        self,
        n_neighbors: int = 3,
        weights: str = "distance",
        numeric_cols: Optional[Sequence[str]] = None,
        categorical_cols: Optional[Sequence[str]] = None,
        feature_weights: Optional[Dict[str, float]] = None,
        fallback_numeric: str = "median",
        fallback_categorical: str = "most_frequent",
        random_state: Optional[int] = None,
    ):
        # ---- Input validation ----
        if weights not in {"uniform", "distance"}:
            raise ValueError("weights must be 'uniform' or 'distance'")
        if fallback_numeric not in {"median", "mean"}:
            raise ValueError("fallback_numeric must be 'median' or 'mean'")
        if fallback_categorical not in {"most_frequent"}:
            raise ValueError("fallback_categorical must be 'most_frequent'")

        # ---- Store public parameters ----
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.feature_weights = feature_weights
        self.fallback_numeric = fallback_numeric
        self.fallback_categorical = fallback_categorical
        self.random_state = random_state

        # ---- Internal attributes created at fit() ----
        self._cols_: List[str] = []               # all column names in training data
        self._numeric_cols_: List[str] = []       # resolved numeric col names
        self._categorical_cols_: List[str] = []   # resolved categorical col names
        self._num_ranges_: Dict[str, float] = {}  # numeric ranges (max-min) used for scaling
        self._fallback_stats_: Dict[str, Any] = {}# per-column fallback values

        # ---- Anchor and precomputed arrays ----
        self._X_fit_: Optional[pd.DataFrame] = None
        self._fit_num_arrs_: Dict[str, np.ndarray] = {}  # numeric columns as np arrays
        self._fit_cat_arrs_: Dict[str, np.ndarray] = {}  # categorical columns as np arrays

    # ----------------------------------------------------------------------
    #  fit()
    # ----------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y=None):
        """
        Learn column partitions, numeric ranges, fallback statistics,
        and store the TRAINING anchor for anchor-based imputation.
        """
        # Ensure DataFrame input
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Keep column order to restore later
        self._cols_ = list(X.columns)

        # Infer numeric / categorical partitions if not given
        if self.numeric_cols is None or self.categorical_cols is None:
            # categorical = object, category, or bool
            cats = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
            # numeric = everything else
            nums = [c for c in X.columns if c not in cats]
            # store resolved lists
            self._numeric_cols_ = nums if self.numeric_cols is None else list(self.numeric_cols)
            self._categorical_cols_ = cats if self.categorical_cols is None else list(self.categorical_cols)
        else:
            # if explicitly provided, just store
            self._numeric_cols_ = list(self.numeric_cols)
            self._categorical_cols_ = list(self.categorical_cols)

        # ---- Compute numeric ranges for Gower normalization ----
        self._num_ranges_.clear()
        for c in self._numeric_cols_:
            s = pd.to_numeric(X[c], errors="coerce")
            rng = s.max(skipna=True) - s.min(skipna=True)
            # If a feature has constant values, range = 0 → mark as 0.0 to avoid div/0
            self._num_ranges_[c] = float(rng) if pd.notna(rng) and rng != 0 else 0.0

        # ---- Compute fallback values for missing neighbour cases ----
        self._fallback_stats_.clear()
        # For numeric columns: use median or mean
        for c in self._numeric_cols_:
            s = pd.to_numeric(X[c], errors="coerce")
            self._fallback_stats_[c] = float(s.median(skipna=True)) if self.fallback_numeric == "median" else float(s.mean(skipna=True))
        # For categorical columns: use most frequent mode
        for c in self._categorical_cols_:
            s = X[c].astype("object")
            mode = s.mode(dropna=True)
            self._fallback_stats_[c] = mode.iloc[0] if not mode.empty else None

        # ---- Store the training anchor (used for all imputations) ----
        self._X_fit_ = X[self._cols_].copy()
        # Pre-extract numpy arrays for performance in distance loops
        self._fit_num_arrs_ = {c: pd.to_numeric(self._X_fit_[c], errors="coerce").to_numpy() for c in self._numeric_cols_}
        self._fit_cat_arrs_ = {c: self._X_fit_[c].astype("object").to_numpy() for c in self._categorical_cols_}

        return self

    # ----------------------------------------------------------------------
    #  fit_transform()
    # ----------------------------------------------------------------------
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Fit on X, then impute X against itself, excluding each row as its own neighbour.
        (Used on training data to avoid self-leakage.)
        """
        self.fit(X, y)
        return self._impute_against_anchor(X, exclude_self=True)

    # ----------------------------------------------------------------------
    #  transform()
    # ----------------------------------------------------------------------
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values in new data using the training anchor
        (learned during fit). Self-exclusion is not applied.
        """
        # Ensure DataFrame and same column order as training
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self._cols_)
        else:
            X = X[self._cols_].copy()
        return self._impute_against_anchor(X, exclude_self=False)

    # ----------------------------------------------------------------------
    #  _cross_gower()
    # ----------------------------------------------------------------------
    def _cross_gower(self, A: pd.DataFrame) -> np.ndarray:
        """
        Compute pairwise Gower distances between rows of A and rows of the anchor (fit set).
        Returns an (nA x nB) matrix of distances in [0,1].
        """
        assert self._X_fit_ is not None, "GowerKNNImputer must be fit before transform."
        nA = len(A)
        nB = len(self._X_fit_)
        D = np.zeros((nA, nB), dtype=float)       # distance matrix to fill
        fw = self.feature_weights or {}           # optional per-feature weights

        # Pre-convert A to numeric / categorical arrays
        A_num = {c: pd.to_numeric(A[c], errors="coerce").to_numpy() for c in self._numeric_cols_}
        A_cat = {c: A[c].astype("object").to_numpy() for c in self._categorical_cols_}

        # Anchor arrays (already extracted at fit time)
        B_num = self._fit_num_arrs_
        B_cat = self._fit_cat_arrs_

        # ---- Compute distances row by row ----
        for i in range(nA):
            # Initialize accumulators for numeric and categorical parts
            num_sum = np.zeros(nB, dtype=float)
            num_cnt = np.zeros(nB, dtype=float)

            # Numeric part: normalized absolute difference
            for c in self._numeric_cols_:
                ai = A_num[c][i]
                if np.isnan(ai):
                    continue  # skip if the value in A is missing
                rng = self._num_ranges_[c]
                if rng == 0.0:
                    continue  # skip constant feature
                bj = B_num[c]
                mask = ~np.isnan(bj)  # ignore NaN in anchor
                contrib = np.zeros_like(bj, dtype=float)
                contrib[mask] = np.abs(ai - bj[mask]) / rng
                w = fw.get(c, 1.0)
                num_sum += w * contrib
                num_cnt += w * mask.astype(float)

            # Categorical part: 0 if same, 1 if different (only where not NaN)
            cat_sum = np.zeros(nB, dtype=float)
            cat_cnt = np.zeros(nB, dtype=float)
            for c in self._categorical_cols_:
                ai = A_cat[c][i]
                bj = B_cat[c]
                mask = ~pd.isna(ai) & ~pd.isna(bj)
                if np.any(mask):
                    diff = np.zeros_like(bj, dtype=float)
                    diff[mask] = (bj[mask] != ai).astype(float)
                    w = fw.get(c, 1.0)
                    cat_sum += w * diff
                    cat_cnt += w * mask.astype(float)

            # Combine numeric and categorical parts.
            # Divide total distance contribution by number of valid comparisons per row pair.
            denom = num_cnt + cat_cnt
            D[i, :] = np.divide(
                num_sum + cat_sum,
                denom,
                out=np.ones_like(denom),     # if denom==0, distance defaults to 1
                where=(denom != 0.0),
            )

        return D

    # ----------------------------------------------------------------------
    #  _weighted_mode()
    # ----------------------------------------------------------------------
    @staticmethod
    def _weighted_mode(values: np.ndarray, weights: np.ndarray) -> Any:
        """
        Compute the mode of a 1D array using weighted frequencies.
        Returns the value with the highest total weight.
        """
        agg: Dict[Any, float] = {}
        for v, w in zip(values, weights):
            agg[v] = agg.get(v, 0.0) + float(w)
        # Break ties by string order to ensure deterministic output
        return max(agg.items(), key=lambda kv: (kv[1], str(kv[0])))[0]

    # ----------------------------------------------------------------------
    #  _impute_against_anchor()
    # ----------------------------------------------------------------------
    def _impute_against_anchor(self, X: pd.DataFrame, exclude_self: bool) -> pd.DataFrame:
        """
        Core imputation logic.

        - Computes Gower distances between each row in X and all anchor rows.
        - For each missing value, finds the k nearest neighbours.
        - Fills numeric values with weighted (or unweighted) averages.
        - Fills categorical values with weighted (or unweighted) modes.
        """
        assert self._X_fit_ is not None, "Imputer not fit."

        X_out = X.copy()                # output DataFrame
        D = self._cross_gower(X)        # nA x nB distance matrix

        # If imputing training data itself, avoid "self" neighbour by
        # setting diagonal distances to infinity (so it can never be chosen).
        if exclude_self and len(X) == len(self._X_fit_) and X_out.index.equals(self._X_fit_.index):
            np.fill_diagonal(D, np.inf)

        nA, nB = D.shape
        eps = 1e-12                     # tiny constant to avoid div/0

        # ---- Loop through each row to impute ----
        for r in range(nA):
            # Determine neighbour indices for current sample
            k = min(self.n_neighbors, nB)
            neigh_idx = np.argpartition(D[r], k - 1)[:k]
            dists = D[r, neigh_idx]

            # Compute weights (uniform or inverse-distance)
            w = np.ones_like(dists) if self.weights == "uniform" else 1.0 / (dists + eps)
            w = w / (w.sum() + eps)     # normalize to sum=1

            # ---- Impute numeric features ----
            for c in self._numeric_cols_:
                # Skip already non-missing
                if pd.notna(X_out.iat[r, X_out.columns.get_loc(c)]):
                    continue
                neigh_vals = self._X_fit_.iloc[neigh_idx][c].astype(float).to_numpy()
                mask = ~np.isnan(neigh_vals)
                if mask.any():
                    # Weighted average if distance weighting, else simple mean
                    X_out.at[X_out.index[r], c] = (
                        float(np.average(neigh_vals[mask], weights=w[mask]))
                        if self.weights == "distance"
                        else float(neigh_vals[mask].mean())
                    )
                else:
                    # All neighbours missing → fallback (median/mean)
                    X_out.at[X_out.index[r], c] = self._fallback_stats_.get(c, np.nan)

            # ---- Impute categorical features ----
            for c in self._categorical_cols_:
                # Skip already non-missing
                if pd.notna(X_out.iat[r, X_out.columns.get_loc(c)]):
                    continue
                neigh_vals = self._X_fit_.iloc[neigh_idx][c].astype("object").to_numpy()
                mask = ~pd.isna(neigh_vals)
                if mask.any():
                    if self.weights == "distance":
                        # Weighted mode for distance weighting
                        X_out.at[X_out.index[r], c] = self._weighted_mode(neigh_vals[mask], w[mask])
                    else:
                        # Simple unweighted mode
                        mode = Counter(neigh_vals[mask]).most_common(1)[0][0]
                        X_out.at[X_out.index[r], c] = mode
                else:
                    # All neighbours missing → fallback to most frequent
                    X_out.at[X_out.index[r], c] = self._fallback_stats_.get(c, None)

        return X_out