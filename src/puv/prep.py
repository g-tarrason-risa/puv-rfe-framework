# src/puv/prep.py
from __future__ import annotations

"""
Data preparation utilities for PUV.

This module emits logs via a module-specific logger. Configure logging
(level, handlers, formatting) in your notebook or CLI entry point.
"""
import logging
logger = logging.getLogger(__name__)  # module-scoped logger

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import matplotlib.pyplot as plt



def filter_and_review_completeness(
    df: pd.DataFrame,
    threshold: float,
    out_path: Path | str,
    title: str,
    cols_to_remove: Optional[Iterable[str]] = None,
    plot: bool = True,
) -> pd.DataFrame:
    
    """
    Compute per-column completeness (% non-null), export low-completeness features,
    return a filtered DataFrame of columns above a threshold, and optionally plot.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    threshold : float
        Minimum completeness percentage to keep a column (0–100).
        Columns with completeness <= threshold are exported and excluded.
    out_path : Path | str
        Base path (without extension) for outputs:
          - '{out_path}.xlsx' for low-completeness list
          - '{out_path}.png' for the plot (if plot=True)
    title : str
        Plot title.
    cols_to_remove : Iterable[str], optional
        Case-insensitive substrings; any kept column containing one is removed.
    plot : bool, default True
        Whether to render/save the bar plot.

    Returns
    -------
    pd.DataFrame
        A copy of `df` with only the selected columns.
    """
    if df is None or df.empty:
        logger.warning("Input DataFrame is empty; returning empty DataFrame.")
        return df.copy()

    if not (0 <= threshold <= 100):
        raise ValueError(f"'threshold' must be in [0, 100], got {threshold}.")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a copy to avoid mutating the original DataFrame
    df_copy = df.copy()

    # % completeness per column = 100 - % nulls
    completeness_pct = (100 - (df_copy.isnull().mean() * 100)).round(2)

    df_complete = (
        pd.DataFrame(
            {"Feature_name": completeness_pct.index, "Completeness %": completeness_pct.values}
        )
        .sort_values(by="Completeness %", ascending=True)
        .reset_index(drop=True)
    )

    # # Export low-completeness features
    # df_low = df_complete[df_complete["Completeness %"] <= threshold]
    # excel_path = out_path.with_suffix(".xlsx")
    # df_low.to_excel(excel_path, index=False)
    # logger.info("Exported low-completeness features (<= %.2f%%) to %s", threshold, excel_path)

    # Select high-completeness features
    high_feats = df_complete.loc[df_complete["Completeness %"] > threshold, "Feature_name"].tolist()

    # Optional exclusion by substring (case-insensitive)
    if cols_to_remove:
        lowered = [s.lower() for s in cols_to_remove]
        kept_feats = [
            f for f in high_feats
            if not any(substr in f.lower() for substr in lowered)
        ]
        logger.info("Removed features containing any of %s (case-insensitive).", cols_to_remove)
    else:
        kept_feats = high_feats

    df_filtered = df_copy[kept_feats].copy()
    if df_filtered.shape[1] == 0:
        logger.warning("All columns were filtered out; returning empty DataFrame.")

    # Prepare plot table without mutating df_complete
    df_plot = df_complete[df_complete["Feature_name"] != "Patient number"]

    if plot:
        kept = set(kept_feats)
        df_plot = df_plot.assign(
            Color=df_plot["Feature_name"].apply(lambda x: "#ADD8E6" if x in kept else "#FFB6C1")
        )

        ax = df_plot.plot(
            kind="bar",
            x="Feature_name",
            y="Completeness %",
            rot=90,
            figsize=(20, 10),
            fontsize=10,
            color=df_plot["Color"],
            legend=False,
        )
        ax.set_title(title, fontsize=25)
        ax.set_xlabel("Feature")
        ax.set_ylabel("Completeness (%)")
        ax.axhline(y=threshold, linestyle="dashed", color="r", linewidth=1)

        plt.tight_layout()
        fig_path = out_path.with_suffix(".png")
        plt.savefig(fig_path, bbox_inches="tight")
        # plt.close()
        logger.info("Saved completeness plot to %s", fig_path)

    logger.info("# of features before data review: %d", len(df_copy.columns))
    logger.info("# of features after data review:  %d", len(df_filtered.columns))

    return df_filtered