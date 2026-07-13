# src/puv/wrangle.py
from __future__ import annotations
import pandas as pd


def process_feature_importance(df):
    # Average the per-fold importance columns, then sort descending
    df = df.copy()
    df['importance_folds_avg'] = df.select_dtypes(include='number').mean(axis=1)
    return df.sort_values(by='importance_folds_avg', ascending=False).reset_index(drop=True)