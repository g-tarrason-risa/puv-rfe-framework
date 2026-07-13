# src/puv/modelling/preprocess.py

from __future__ import annotations
from typing import List, Optional, Literal
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from joblib import Memory

from .impute import GowerKNNImputer
from .scaling import NormalityAwareScaler


def _preprocessor(
    num_cols: List[str],
    cat_cols: List[str],
    use_normality_scaler: bool = True,
    normality_alpha: float = 0.05,
    non_normal_strategy: str = "robust",
    numeric_scaler: Optional[BaseEstimator] = None,
    gower_n_neighbors: int = 3,
    gower_weights: Literal["uniform", "distance"] = "distance",
    memory: Optional[Memory] = None,
) -> Pipeline:
    """
    Build a composite preprocessing pipeline for mixed-type data.

    Steps:
    1. Mixed-type imputation using Gower distance (handles numeric + categorical)
    2. Separate transformation of numeric and categorical columns:
       - Numeric: optionally scaled depending on normality test
       - Categorical: one-hot encoded with binary columns dropped
    3. Wraps everything into a scikit-learn Pipeline with optional caching

    Parameters
    ----------
    num_cols : list of str
        Names of numeric columns.
    cat_cols : list of str
        Names of categorical columns.
    use_normality_scaler : bool, default=True
        Whether to apply the NormalityAwareScaler to numeric features.
    normality_alpha : float, default=0.05
        Alpha level for Shapiro–Wilk test inside NormalityAwareScaler.
    non_normal_strategy : {"robust","none"}, default="robust"
        How to handle non-normal numeric columns.
    gower_n_neighbors : int, default=3
        Number of neighbours used by GowerKNNImputer.
    gower_weights : {"uniform","distance"}, default="distance"
        How to weight neighbours during imputation.
    memory : joblib.Memory, optional
        Optional caching directory for the whole pipeline (improves speed).

    Returns
    -------
    sklearn.pipeline.Pipeline
        Complete preprocessing pipeline combining imputation and transformation.
    """

    # ---------------------------------------------------------------------
    # Step 1: Define a mixed-type imputer
    # ---------------------------------------------------------------------
    mixed_imputer = GowerKNNImputer(
        n_neighbors=gower_n_neighbors,         # number of neighbours for imputation
        weights=gower_weights,                 # weighting scheme: "uniform" or "distance"
        numeric_cols=num_cols,                 # numeric column list
        categorical_cols=cat_cols,             # categorical column list
        fallback_numeric="median",             # fallback for numeric if neighbours all NaN
        fallback_categorical="most_frequent",  # fallback for categorical
    )

    # ---------------------------------------------------------------------
    # Step 2a: Numeric transformation pipeline
    # ---------------------------------------------------------------------
    num_steps = []
    if numeric_scaler is not None:
        # Explicit single scaler applied to every numeric column. Cloned so each
        # fold/pipeline gets its own unfitted instance (no cross-fold state sharing).
        # Preferred when coefficients must be comparable across folds.
        num_steps.append(("scaler", clone(numeric_scaler)))
    elif use_normality_scaler:
        # Add a NormalityAwareScaler step that chooses
        # between StandardScaler or RobustScaler based on Shapiro–Wilk p-value
        num_steps.append((
            "scaler",
            NormalityAwareScaler(alpha=normality_alpha, non_normal_strategy=non_normal_strategy)
        ))

    # If we have numeric steps, create a pipeline; otherwise passthrough (no scaling)
    num_pipe = Pipeline(steps=num_steps) if num_steps else "passthrough"

    # ---------------------------------------------------------------------
    # Step 2b: Categorical transformation pipeline
    # ---------------------------------------------------------------------
    cat_pipe = Pipeline(steps=[
        (
            "oh",
            OneHotEncoder(
                handle_unknown="ignore",  # ignore unseen categories at transform time
                drop="if_binary",         # drop one level of binary variables to avoid redundancy
                sparse_output=False,       # return dense numpy array (not sparse matrix)
            )
        )
    ])

    # ---------------------------------------------------------------------
    # Step 3: Combine numeric + categorical transformers
    # ---------------------------------------------------------------------
    ct = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),  # apply numeric pipeline to numeric columns
            ("cat", cat_pipe, cat_cols),  # apply categorical pipeline to categorical columns
        ],
        remainder="drop",                 # drop any columns not listed above
        verbose_feature_names_out=False,  # keep clean feature names (no prefixes)
    )

    # ---------------------------------------------------------------------
    # Step 4: Full pipeline (imputation → column transforms)
    # ---------------------------------------------------------------------
    # The top-level pipeline first imputes missing values using GowerKNNImputer,
    # then passes the completed data to the ColumnTransformer for encoding/scaling.
    # The optional 'memory' argument enables joblib caching for expensive fit() calls.
    return Pipeline(
        steps=[
            ("impute_mixed", mixed_imputer),  # step 1: impute missing values
            ("pre", ct),                      # step 2: scale + encode columns
        ],
        memory=memory,                        # optional caching for speed
    )