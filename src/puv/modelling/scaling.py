# src/puv/modelling/scaling.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy.stats import shapiro

# ---------- normality-aware scaler --------------------------------------------
# This transformer scales each numeric column independently based on a
# normality check (Shapiro–Wilk). If a column looks Gaussian (p >= alpha),
# it uses StandardScaler; otherwise it either uses RobustScaler (median/IQR)
# or leaves the column unscaled, depending on non_normal_strategy.

@dataclass(eq=False)
class NormalityAwareScaler(BaseEstimator, TransformerMixin):
    # Significance level for the Shapiro–Wilk test; higher alpha => easier to call "normal".
    alpha: float = 0.05
    # Strategy when a column fails normality: "robust" => RobustScaler, "none" => passthrough.
    non_normal_strategy: Literal["robust", "none"] = "robust"

    # Per-column fitted scalers (None means passthrough for that column).
    _col_scalers: List[Optional[TransformerMixin]] = field(default_factory=list, init=False, repr=False)
    # Per-column Shapiro p-values (for diagnostics / introspection).
    _col_pvalues: List[Optional[float]] = field(default_factory=list, init=False, repr=False)
    # Per-column chosen strategy label: "standard" | "robust" | "none".
    _col_choices: List[str] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        # Basic input validation for alpha and strategy.
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        if self.non_normal_strategy not in {"robust", "none"}:
            raise ValueError("non_normal_strategy must be 'robust' or 'none'")

    def fit(self, X, y=None):
        # Convert input to numpy array (accepts lists, pandas, etc.).
        X = np.asarray(X)
        n_cols = X.shape[1]
        # Reset any previous state.
        self._col_scalers.clear()
        self._col_pvalues.clear()
        self._col_choices.clear()

        # Decide a scaler per column.
        for j in range(n_cols):
            # Extract j-th column as float vector; keep a 2D view for sklearn scalers.
            col = X[:, j].astype(float)
            col2d = col.reshape(-1, 1)
            n = len(col)

            # Run Shapiro–Wilk normality test when it is statistically valid:
            # - at least 3 unique values
            # - sample size between 3 and 5000 (Shapiro is defined/recommended in this range)
            if np.unique(col).size >= 3 and 3 <= n <= 5000:
                try:
                    p = float(shapiro(col).pvalue)
                except Exception:
                    # If test fails (e.g., due to NaNs/degenerate data),
                    # treat as non-normal by forcing p < alpha.
                    p = 0.0
            else:
                # Outside the recommended range or not enough variability:
                # treat as non-normal.
                p = 0.0

            # Choose scaler based on p-value and configured strategy.
            if p >= self.alpha:
                # Looks normal => use StandardScaler (mean=0, std=1).
                scaler = StandardScaler().fit(col2d)
                choice = "standard"
            elif self.non_normal_strategy == "robust":
                # Not normal and strategy=robust => use RobustScaler (median/IQR).
                scaler = RobustScaler().fit(col2d)
                choice = "robust"
            else:
                # Not normal and strategy=none => no scaling (passthrough).
                scaler = None
                choice = "none"

            # Persist the decision and diagnostics.
            self._col_scalers.append(scaler)
            self._col_pvalues.append(p)
            self._col_choices.append(choice)

        return self

    def transform(self, X):
        # Apply the per-column choice made during fit().
        X = np.asarray(X)
        cols = []
        for j, scaler in enumerate(self._col_scalers):
            col2d = X[:, j].reshape(-1, 1)
            # If scaler is None, passthrough the raw values; else transform.
            cols.append(col2d if scaler is None else scaler.transform(col2d))
        # Reconstruct the full matrix by horizontally stacking the columns.
        return np.hstack(cols)

    def get_feature_names_out(self, input_features=None):
        # Keep feature names unchanged; required for compatibility with ColumnTransformer.
        return np.array(input_features) if input_features is not None else None