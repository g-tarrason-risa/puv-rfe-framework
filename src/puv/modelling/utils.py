# src/puv/modelling/utils.py

from __future__ import annotations
from typing import Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype, is_categorical_dtype, is_integer_dtype,
    is_object_dtype, is_float_dtype
)

# ---------------------------------------------------------------------
# Utility functions used throughout the modelling pipeline.
# These provide type inference for columns, positive-class resolution,
# and standard error calculations.
# ---------------------------------------------------------------------

def _infer_column_types(
    df: pd.DataFrame,
    target_col: str,
    low_int_cardinality: int = 2,
    treat_binary_floats_as_categorical: bool = True,
) -> Tuple[List[str], List[str]]:
    """
    Automatically infer which columns in a DataFrame are numeric vs categorical.

    Rules:
      1. Columns of dtype object, category, or bool → categorical
      2. Integer columns with few unique values (<= low_int_cardinality) → categorical
      3. Optionally, float columns that contain only 0/1 (binary) values → categorical
      4. Everything else → numeric

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    target_col : str
        Column name to exclude (the supervised target).
    low_int_cardinality : int, default=2
        Threshold for treating low-cardinality integers as categorical.
    treat_binary_floats_as_categorical : bool, default=True
        Whether to classify float columns with only {0.0, 1.0} as categorical.

    Returns
    -------
    num_cols : list of str
        Names of columns inferred as numeric.
    cat_cols : list of str
        Names of columns inferred as categorical.
    """
    assert target_col in df.columns, f"target_col '{target_col}' not in DataFrame"
    features = df.drop(columns=[target_col])  # only infer on predictor features

    cat_cols: List[str] = []

    # Step 1: Explicitly categorical types (object, category, bool)
    for c in features.columns:
        s = features[c]
        if is_object_dtype(s) or is_categorical_dtype(s) or is_bool_dtype(s):
            cat_cols.append(c)

    # Step 2: Integer columns with small unique cardinality → categorical
    for c in features.columns:
        if c in cat_cols:
            continue
        s = features[c]
        if is_integer_dtype(s) and s.nunique(dropna=True) <= low_int_cardinality:
            cat_cols.append(c)

    # Step 3: Float columns that behave like binary flags → categorical (optional)
    if treat_binary_floats_as_categorical:
        for c in features.columns:
            if c in cat_cols:
                continue
            s = features[c]
            if is_float_dtype(s):
                uniq = pd.unique(s.dropna())
                # if the column contains only 0.0 and/or 1.0 values, mark as categorical
                if len(uniq) <= 2 and set(map(float, uniq)).issubset({0.0, 1.0}):
                    cat_cols.append(c)

    # Finalize categorical/numeric splits
    cat_set = set(cat_cols)
    num_cols = [c for c in features.columns if c not in cat_set]
    cat_cols = [c for c in features.columns if c in cat_set]
    return num_cols, cat_cols


def _positive_label(y: pd.Series, pos_label: Optional[Any] = None) -> Any:
    """
    Determine the positive class label for a binary classification target.

    If pos_label is provided, return it directly. Otherwise,
    default to the last label in descending frequency order.

    Parameters
    ----------
    y : pd.Series
        Target labels.
    pos_label : optional
        User-specified positive class.

    Returns
    -------
    label : Any
        The chosen positive class.
    """
    if pos_label is not None:
        return pos_label
    vc = y.value_counts()
    # By default, the least frequent (last in value_counts order) is considered positive
    return vc.index[-1]


def _sem(series: pd.Series) -> float:
    """
    Compute the standard error of the mean (SEM).

    SEM = standard deviation / sqrt(sample size)

    Parameters
    ----------
    series : pd.Series
        Numeric series.

    Returns
    -------
    float
        The standard error of the mean.
    """
    return series.std(ddof=1) / np.sqrt(len(series))