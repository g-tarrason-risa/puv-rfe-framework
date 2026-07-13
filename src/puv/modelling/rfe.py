# # src/puv/modelling/rfe.py
# src/puv/modelling/rfe.py

from __future__ import annotations

import logging
logger = logging.getLogger(__name__)  # module-scoped logger


import contextlib
from typing import Any, Callable, Dict, List, Optional, Union, Literal
import numpy as np
import pandas as pd
from joblib import Memory
from tqdm.auto import tqdm
import warnings

from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, recall_score, roc_auc_score,
    precision_recall_curve, auc, confusion_matrix,
)

from .utils import _infer_column_types, _positive_label, _sem
from .preprocess import _preprocessor


def _strip_binary_suffix(name: str) -> str:
    """Remove trailing `_1` or `_1.0` that come from drop-if-binary OHE columns."""
    if "__" not in name:
        target = name
        prefix = ""
    else:
        prefix, target = name.split("__", 1)

    for suffix in ("_1.0", "_1"):
        if target.endswith(suffix):
            target = target[: -len(suffix)]
            break

    return f"{prefix}__{target}" if prefix else target

def rfe_eval(
    df_raw: pd.DataFrame,
    target_col: str,
    max_feature_eval: int = 100,         # upper bound for number of features to evaluate (post-preprocessing)
    n_splits: int = 5,                  # outer CV folds
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    estimator: Optional[BaseEstimator] = None,   # base estimator for both RFE and final clf
    pos_label: Optional[Any] = None,             # explicit positive label (if None, inferred)
    random_state: int = 42,
    use_normality_scaler: bool = True,
    normality_alpha: float = 0.05,
    non_normal_strategy: str = "robust",
    numeric_scaler: Optional[BaseEstimator] = None,  # explicit single scaler; overrides normality-aware choice
    gower_n_neighbors: int = 3,
    gower_weights: Literal["uniform","distance"] = "distance",
    tune_hyperparams: bool = True,       # whether to run inner-CV hyperparameter search
    tuner: str = "random",               # "random" or "grid"
    inner_splits: int = 5,               # inner CV folds for tuning
    scoring: Optional[Union[str, Callable]] = "average_precision",  # refit/scoring metric for tuner
    n_iter: int = 20,                    # iterations for RandomizedSearchCV
    param_grid: Optional[Dict[str, List[Any]]] = None,
    param_distributions: Optional[Dict[str, Any]] = None,
    show_progress: bool = True,          # display tqdm progress bars
    verbose: int = 1,                    # 0: quiet per-k summary; 1: show tuner best; 2: per-fold metrics
    sklearn_verbose: int = 0,            # verbosity passed to Grid/RandomizedSearchCV
    suppress_warnings: bool = True,      # silence common warnings for cleaner logs
    target_recall: float = 1.0,          # threshold analysis: aim to achieve at least this recall
    fbeta_beta: float = 2.0,             # F-beta emphasis (beta=2 favors recall)

    rfe_step: Union[int, float] = 1,     # RFE elimination step (int or fraction of remaining features)
    cache_dir: Optional[str] = None,     # joblib.Memory path (enables caching)
) -> Dict[str, Any]:
    """
    Leakage-safe RFE with nested CV and threshold analysis.

    High-level flow:
      1) Split data with outer StratifiedKFold.
      2) Inside each outer fold, build a pipeline:
             preprocessor -> RFE(estimator) -> classifier
         Optionally hyperparameter tune this pipeline with an inner CV.
      3) Evaluate metrics on the held-out fold and collect selected features.
      4) Repeat for k=1..max_k number of selected features.
      5) Aggregate results (means, SEMs, per-fold details, thresholds).
    """

    ctx = warnings.catch_warnings() if suppress_warnings else contextlib.nullcontext()
    with ctx:
        if suppress_warnings:
            # Silence common sklearn/scipy warnings to keep output readable
            warnings.simplefilter("ignore", UserWarning)
            warnings.simplefilter("ignore", RuntimeWarning)

        log_level = logging.INFO if verbose >= 1 else logging.WARNING
        logger.setLevel(log_level)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)
            logger.propagate = False  # avoid duplicate output through root handlers

        # Work on a copy so we don't mutate the caller's DataFrame
        df = df_raw.copy()
        assert target_col in df.columns, f"target_col '{target_col}' not in DataFrame"

        # Separate features/target
        y = df[target_col]
        X = df.drop(columns=[target_col])

        # Auto-infer numeric/categorical columns unless provided
        if numeric_cols is None or categorical_cols is None:
            num_cols, cat_cols = _infer_column_types(df_raw, target_col)
        else:
            num_cols, cat_cols = numeric_cols, categorical_cols

        # Default estimator is class-balanced logistic regression (robust baseline)
        if estimator is None:
            base_est = LogisticRegression(
                penalty="l2", solver="liblinear", max_iter=2000,
                class_weight="balanced", random_state=random_state
            )
        else:
            base_est = estimator

        # Choose positive class (used in metrics); can be user-specified
        pos = _positive_label(y, pos_label)

        # Ensure we have enough samples per class for the requested number of folds
        min_class_count = y.value_counts().min()
        if min_class_count < 2:
            raise ValueError("Stratified CV needs at least two samples in every class.")
        
        if min_class_count < n_splits:
            warnings.warn(
                f"Not enough samples per class for {n_splits}-fold CV. "
                f"Minimum class count is {min_class_count}. "
                f"Reducing n_splits to {min_class_count}."
            )
            n_splits = min_class_count

        # Outer CV that controls unbiased performance estimation
        cv_outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        # Optional joblib cache across fits/transforms (speeds up repeated runs)
        memory = Memory(location=cache_dir, verbose=0) if cache_dir else None

        # ---- Pre-probe: how many features exist AFTER preprocessing? ----
        # Fit the preprocessor on ALL X just to compute the transformed dimensionality.
        # This does NOT leak into modelling since it's not used to train the final models.
        pre_probe = _preprocessor(
            num_cols, cat_cols,
            use_normality_scaler=use_normality_scaler,
            normality_alpha=normality_alpha,
            non_normal_strategy=non_normal_strategy,
            numeric_scaler=numeric_scaler,
            gower_n_neighbors=gower_n_neighbors,
            gower_weights=gower_weights,
            memory=memory,
        )
        pre_probe.fit(X)
        total_pre_features = pre_probe.transform(X.iloc[:1]).shape[1]  # transform one row to get width
        max_k = min(max_feature_eval, total_pre_features)               # cap user request to available features

        if max_k < 1:
            raise ValueError("Preprocessing produced zero usable features; cannot run RFE.")

        # ---- Default tuning space if user didn't pass any ----
        if tune_hyperparams and param_grid is None and param_distributions is None:
            if isinstance(base_est, LogisticRegression):
                try:
                    from scipy.stats import loguniform
                except ImportError as exc:
                    raise ImportError(
                        "tune_hyperparams=True with a LogisticRegression baseline requires SciPy. "
                        "Install scipy or provide param_distributions explicitly."
                    ) from exc
                param_distributions = {
                    "clf__C": loguniform(1e-3, 1e3),
                    "rfe__estimator__C": loguniform(1e-3, 1e3),
                    "rfe__step": [rfe_step],
                }
            else:
                # For other estimators propagate the RFE step
                param_distributions = {"rfe__step": [rfe_step]}

        # Containers to accumulate results across k and folds
        summary_rows = []
        anova_acc_rows = []
        anova_prauc_rows = []
        perfold: Dict[int, List[Dict[str, float]]] = {}
        selected_features: Dict[int, List[List[str]]] = {}
        best_params: Dict[int, List[Dict[str, Any]]] = {}
        threshold_analysis: Dict[int, List[Dict[str, Any]]] = {}
        confusion_mats: Dict[int, List[Dict[str, int]]] = {}
        feature_importances: Dict[int, List[Dict[str, float]]] = {}

        # Iterate number of features selected by RFE: k = 1..max_k
        k_iter = range(1, max_k + 1)
        k_bar = tqdm(k_iter, desc="RFE: #features", disable=not show_progress)

        for k in k_bar:
            # Per-k accumulators
            fold_metrics: List[Dict[str, float]] = []
            fold_selected: List[List[str]] = []
            fold_bestparams: List[Dict[str, Any]] = []
            fold_threshold_info: List[Dict[str, Any]] = []
            fold_confusions: List[Dict[str, int]] = []
            fold_importances: List[Dict[str, float]] = []

            # Materialize outer folds up-front (so we can progress-bar them)
            folds = list(cv_outer.split(X, y))
            fold_bar = tqdm(folds, desc=f"#features={k} evaluation process", leave=False, disable=not show_progress)

            for fold_idx, (tr, te) in enumerate(fold_bar, start=1):
                # Split into train/test for this outer fold
                X_tr, X_te = X.iloc[tr], X.iloc[te]
                y_tr, y_te = y.iloc[tr], y.iloc[te]

                # Fit preprocessor on TRAIN ONLY to compute how many transformed features exist in this fold
                pre_tr = _preprocessor(
                    num_cols, cat_cols,
                    use_normality_scaler=use_normality_scaler,
                    normality_alpha=normality_alpha,
                    non_normal_strategy=non_normal_strategy,
                    numeric_scaler=numeric_scaler,
                    gower_n_neighbors=gower_n_neighbors,
                    gower_weights=gower_weights,
                    memory=memory,
                )
                pre_tr.fit(X_tr)
                n_pre_feats_tr = pre_tr.transform(X_tr.iloc[:1]).shape[1]
                k_fold = min(k, n_pre_feats_tr)  # RFE cannot select more than available this fold

                # Build the modelling pipeline for this fold:
                #   preprocessor -> RFE(estimator=base_est) -> classifier=base_est
                # Using clone() prevents cross-fold contamination of fitted state.
                base_pipe = Pipeline(
                    steps=[
                        ("pre", _preprocessor(
                            num_cols, cat_cols,
                            use_normality_scaler=use_normality_scaler,
                            normality_alpha=normality_alpha,
                            non_normal_strategy=non_normal_strategy,
                            numeric_scaler=numeric_scaler,
                            gower_n_neighbors=gower_n_neighbors,
                            gower_weights=gower_weights,
                            memory=memory,
                        )),
                        ("rfe", RFE(estimator=clone(base_est), n_features_to_select=k_fold, step=rfe_step)),  
                        ("clf", clone(base_est)),
                    ],
                    memory=memory,
                )

                pipe = base_pipe  # may be replaced by search.best_estimator_ below

                # Optional nested hyperparameter tuning (inner CV)
                if tune_hyperparams:
                    min_inner_count = y_tr.value_counts().min()
                    if min_inner_count < 2:
                        pipe.fit(X_tr, y_tr)
                        fold_bestparams.append({"tuning_skipped": "insufficient class counts"})
                    else:
                        cv_inner = StratifiedKFold(
                            # Inner CV folds limited by smallest class size to avoid errors
                            n_splits=min(inner_splits, max(2, y_tr.value_counts().min())),
                            shuffle=True, random_state=random_state
                        )
                        if tuner == "grid":
                            if not param_grid:
                                raise ValueError(
                                    "tuner='grid' requires a non-empty param_grid. "
                                    "Supply the grid you want GridSearchCV to explore.")
                            search = GridSearchCV(
                                estimator=pipe,
                                param_grid=param_grid,
                                scoring=scoring,
                                cv=cv_inner,
                                n_jobs=-1,
                                refit=True,              # refit best pipeline on full training split
                                verbose=sklearn_verbose,
                            )
                        else:
                            search = RandomizedSearchCV(
                                estimator=pipe,
                                param_distributions=param_distributions if param_distributions is not None else {},
                                n_iter=n_iter,
                                scoring=scoring,
                                cv=cv_inner,
                                n_jobs=-1,
                                random_state=random_state,
                                refit=True,
                                verbose=sklearn_verbose,
                            )
            
                        # Fit inner search and use the best estimator for evaluation on outer test
                        search.fit(X_tr, y_tr)
                        pipe = search.best_estimator_
                        fold_bestparams.append(search.best_params_)
                    
                        if verbose >= 2:
                            logger.info(f"[k={k} fold={fold_idx}] best score ({scoring}): {getattr(search,'best_score_',np.nan):.4f}")
                            logger.info(f"[k={k} fold={fold_idx}] best params: {search.best_params_}")
                else:
                    # No tuning: just fit the pipeline on the training split
                    pipe.fit(X_tr, y_tr)
                    fold_bestparams.append({})

                # ---- Predict on the held-out outer test split ----
                y_pred = pipe.predict(X_te)

                labels = list(pipe.named_steps["clf"].classes_)
                if labels != [pos]:
                    if pos in labels and labels.index(pos) != 1:
                        labels = [lab for lab in labels if lab != pos] + [pos]
                cm = confusion_matrix(y_te, y_pred, labels=labels)
                fold_confusions.append(
                    dict(TN=int(cm[0, 0]), FP=int(cm[0, 1]),
                        FN=int(cm[1, 0]), TP=int(cm[1, 1]))
                )


                # Continuous positive-class probability for PR/ROC and threshold metrics.
                # We require predict_proba so scores are calibrated and comparable across
                # folds; wrap decision-function-only estimators in CalibratedClassifierCV.
                clf_step = pipe.named_steps["clf"]
                if not hasattr(clf_step, "predict_proba"):
                    raise ValueError(
                        f"{type(clf_step).__name__} has no predict_proba; wrap it in "
                        "CalibratedClassifierCV so RFE eval gets calibrated scores."
                    )
                probs = pipe.predict_proba(X_te)
                pos_idx = list(clf_step.classes_).index(pos)
                y_pos = probs[:, pos_idx]

                # Standard classification metrics
                acc = accuracy_score(y_te, y_pred)
                bacc = balanced_accuracy_score(y_te, y_pred)
                rec_pos = recall_score(y_te, y_pred, pos_label=pos, zero_division=0)
                f1_pos = f1_score(y_te, y_pred, pos_label=pos, zero_division=0)
                try:
                    roc = roc_auc_score((y_te == pos).astype(int), y_pos)
                except ValueError:
                    roc = np.nan  # ROC-AUC undefined if only one class is present

                # Precision–Recall curve & PR-AUC for imbalanced-aware assessment
                y_true_bin = (y_te == pos).astype(int).to_numpy()
                precision, recall, thresholds = precision_recall_curve(y_true_bin, y_pos)
                pr_auc = auc(recall, precision)

                # ---- Threshold analysis ----
                thr_at_target = None
                prec_at_target = np.nan
                rec_at_target = np.nan

                # Find a threshold that achieves at least 'target_recall'
                if len(thresholds) > 0:
                    # recall has length len(thresholds) + 1; skip the first point when aligning
                    idx_candidates = np.where(recall[1:] >= target_recall)[0] + 1
                    if idx_candidates.size > 0:
                        i = idx_candidates[-1]  # farthest-right index that still meets recall target
                        thr_at_target = float(thresholds[i - 1])
                        prec_at_target = float(precision[i])
                        rec_at_target = float(recall[i])
                    else:
                        # No threshold reaches the target recall: "predict all positive"
                        thr_at_target = "always_positive"
                        prec_at_target = float(precision[0])
                        rec_at_target = float(recall[0])
                else:
                    # Degenerate curve: also "predict all positive"
                    thr_at_target = "always_positive"
                    prec_at_target = float(precision[0])
                    rec_at_target = float(recall[0])

                # Best F_beta over the PR curve (default beta=2 emphasizes recall)
                beta2 = fbeta_beta ** 2
                with np.errstate(divide="ignore", invalid="ignore"):
                    fbeta_vals = (1 + beta2) * (precision * recall) / (beta2 * precision + recall + 1e-12)
                best_i = int(np.nanargmax(fbeta_vals))
                best_fbeta = float(fbeta_vals[best_i])
                best_fbeta_prec = float(precision[best_i])
                best_fbeta_rec = float(recall[best_i])
                if best_i >= 1 and len(thresholds) >= best_i:
                    best_fbeta_thr = float(thresholds[best_i - 1])
                else:
                    best_fbeta_thr = "always_positive"

                # Record per-fold scalar metrics
                fold_metrics.append(
                    dict(
                        fold=fold_idx,
                        Accuracy=acc,
                        BalancedAccuracy=bacc,
                        Recall_pos=rec_pos,
                        F1_pos=f1_pos,
                        ROC_AUC=roc,
                        PR_AUC=pr_auc,
                    )
                )

                # Collect the feature names kept by RFE for this fold/k
                ct_fit: Any = pipe.named_steps["pre"].named_steps["pre"]  # ColumnTransformer fitted inside preprocessor
                rfe_fit = pipe.named_steps["rfe"]
                pre_names = [_strip_binary_suffix(n) for n in ct_fit.get_feature_names_out()]
                support = rfe_fit.support_                   # boolean mask of kept features
                sel_names = [n for n, keep in zip(pre_names, support) if keep]
                fold_selected.append(sel_names)

                # Store confusion matrix for this fold/k
                clf = pipe.named_steps["clf"]
                if hasattr(clf, "coef_"):
                    weights = clf.coef_.ravel()
                elif hasattr(clf, "feature_importances_"):
                    weights = clf.feature_importances_
                else:
                    weights = None
                if weights is not None:
                    fold_importances.append(dict(zip(sel_names, weights)))

                # Store threshold diagnostics for this fold/k
                fold_threshold_info.append(
                    dict(
                        threshold_at_target_recall=thr_at_target,
                        precision_at_target_recall=prec_at_target,
                        recall_at_target_recall=rec_at_target,
                        best_fbeta_threshold=best_fbeta_thr,
                        best_fbeta=best_fbeta,
                        best_fbeta_precision=best_fbeta_prec,
                        best_fbeta_recall=best_fbeta_rec,
                    )
                )

                # Optional verbose per-fold logging
                if verbose >= 2:
                    logger.info(
                        f"[k={k} fold={fold_idx}] "
                        f"Acc={acc:.3f} | BAcc={bacc:.3f} | F1+={f1_pos:.3f} | "
                        f"PR-AUC={pr_auc:.3f} | ROC-AUC={roc:.3f} || "
                        f"thr@rec>={target_recall:.2f}: {thr_at_target} "
                        f"(P={prec_at_target:.3f}, R={rec_at_target:.3f}); "
                        f"best F{fbeta_beta:.1f}={best_fbeta:.3f} @ thr={best_fbeta_thr}"
                    )

            # ---- Aggregate over folds for this k ----
            df_k = pd.DataFrame(fold_metrics)
            k_summary = dict(
                N_Features=k,
                Accuracy_Mean=df_k["Accuracy"].mean(),
                Accuracy_SEM=_sem(df_k["Accuracy"]),
                BalancedAccuracy_Mean=df_k["BalancedAccuracy"].mean(),
                BalancedAccuracy_SEM=_sem(df_k["BalancedAccuracy"]),
                Recall_pos_Mean=df_k["Recall_pos"].mean(),
                Recall_pos_SEM=_sem(df_k["Recall_pos"]),
                F1_pos_Mean=df_k["F1_pos"].mean(),
                F1_pos_SEM=_sem(df_k["F1_pos"]),
                ROC_AUC_Mean=df_k["ROC_AUC"].mean(skipna=True),
                ROC_AUC_SEM=_sem(df_k["ROC_AUC"].dropna()) if df_k["ROC_AUC"].notna().any() else np.nan,
                PR_AUC_Mean=df_k["PR_AUC"].mean(),
                PR_AUC_SEM=_sem(df_k["PR_AUC"]),
            )
            summary_rows.append(k_summary)

            # Update outer progress-bar postfix with quick view of current k
            if show_progress:
                k_bar.set_postfix({"k": k, "Acc": f"{k_summary['Accuracy_Mean']:.3f}", "PR-AUC": f"{k_summary['PR_AUC_Mean']:.3f}"})

            # Long-format rows for potential repeated-measures ANOVA/plots
            for r in fold_metrics:
                anova_acc_rows.append({"n_features": k, "fold": r["fold"], "score": r["Accuracy"]})
                anova_prauc_rows.append({"n_features": k, "fold": r["fold"], "score": r["PR_AUC"]})

            # Persist per-k artifacts
            perfold[k] = fold_metrics
            selected_features[k] = fold_selected
            best_params[k] = fold_bestparams
            threshold_analysis[k] = fold_threshold_info
            confusion_mats[k] = fold_confusions
            feature_importances[k] = fold_importances

            # Optional concise summary per k
            if verbose >= 1:
                logger.info(
                    f"[k={k}] "
                    f"Acc={k_summary['Accuracy_Mean']:.3f}±{k_summary['Accuracy_SEM']:.3f}  |  "
                    f"PR-AUC={k_summary['PR_AUC_Mean']:.3f}±{k_summary['PR_AUC_SEM']:.3f}"
                )

        # ---- Final output tables/data ----
        summary_df = pd.DataFrame(summary_rows).sort_values("N_Features").reset_index(drop=True)
        acc_anova_df = pd.DataFrame(anova_acc_rows)
        prauc_anova_df = pd.DataFrame(anova_prauc_rows)

        # Package everything into a results dict
        results = {
            "Summary": summary_df,                      # per-k means/SEMs
            "Accuracy_ANOVA_Data": acc_anova_df,        # long-form accuracy data
            "PR_AUC_ANOVA_Data": prauc_anova_df,        # long-form PR-AUC data
            "PerFold": perfold,                         # raw metrics per fold
            "Selected_Features": selected_features,     # names kept by RFE per fold
            "Metadata": {                               # run configuration for provenance
                "target_col": target_col,
                "numeric_cols": num_cols,
                "categorical_cols": cat_cols,
                "max_feature_eval_requested": max_feature_eval,
                "max_feature_eval_used": max_k,
                "n_splits": n_splits,
                "estimator": type(base_est).__name__,
                "random_state": random_state,
                "class_counts": y.value_counts().to_dict(),
                "positive_label": pos,
                "use_normality_scaler": use_normality_scaler,
                "normality_alpha": normality_alpha,
                "non_normal_strategy": non_normal_strategy,
                "numeric_scaler": type(numeric_scaler).__name__ if numeric_scaler is not None else None,
                "gower_n_neighbors": gower_n_neighbors,
                "gower_weights": gower_weights,
                "tune_hyperparams": tune_hyperparams,
                "tuner": tuner,
                "inner_splits": inner_splits,
                "scoring": scoring,
                "n_iter": n_iter if tuner == "random" else None,
                "sklearn_verbose": sklearn_verbose,
                "target_recall": target_recall,
                "fbeta_beta": fbeta_beta,
                "rfe_step": rfe_step,
                "cache_dir": cache_dir,
            },
            "BestParams": best_params if tune_hyperparams else {},  # only populated when tuning
            "ThresholdAnalysis": threshold_analysis,                 # per-fold threshold diagnostics
            "ConfusionMatrices": confusion_mats,                     # per-fold confusion matrices
            "FeatureImportances": feature_importances,               # per-fold feature importances
        }
        return results
