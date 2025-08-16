from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from stresstest.data import load_dataset, train_test_split_by_date, TARGET_COL, DATE_COL
from stresstest.models import build_pipeline
from stresstest.metrics import compute_metrics

def time_series_cv_auc(pipe, X, y, n_splits=5):
    """Simple TimeSeriesSplit CV returning ROC-AUC per fold."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for train_idx, val_idx in tscv.split(X):
        Xtr, Xval = X.iloc[train_idx], X.iloc[val_idx]
        ytr, yval = y.iloc[train_idx], y.iloc[val_idx]
        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xval)[:,1]
        # roc_auc_score inline to avoid extra import here
        from sklearn.metrics import roc_auc_score
        scores.append(float(roc_auc_score(yval, p)))
    return scores

def main():
    ap = argparse.ArgumentParser(description="Train stress prediction model (logistic or random_forest).")
    ap.add_argument("--data", type=Path, required=True, help="Input CSV with features + label")
    ap.add_argument("--model", type=str, default="logistic", choices=["logistic", "random_forest"])
    ap.add_argument("--date_cutoff", type=str, default="2019-12-31", help="Train <= cutoff; test > cutoff")
    ap.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    ap.add_argument("--cv_splits", type=int, default=5)
    args = ap.parse_args()

    df = load_dataset(str(args.data))
    if TARGET_COL not in df.columns:
        raise ValueError(f"TARGET_COL '{TARGET_COL}' missing.")

    # Train/test split by time
    train_df, test_df = train_test_split_by_date(df, cutoff=args.date_cutoff)

    # Separate features/target
    FEATURES = [c for c in df.columns if c not in {TARGET_COL, DATE_COL, "bank_id"}]  # exclude bank_id to avoid leakage
    X_train, y_train = train_df[FEATURES], train_df[TARGET_COL].astype(int)
    X_test, y_test = test_df[FEATURES], test_df[TARGET_COL].astype(int)

    pipe, tag = build_pipeline(args.model)

    # Time-aware CV on train
    cv_scores = time_series_cv_auc(pipe, X_train.reset_index(drop=True), y_train.reset_index(drop=True), n_splits=args.cv_splits)
    print(f"TimeSeriesSplit ROC-AUC ({args.cv_splits} folds): {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f} | {cv_scores}")

    # Fit on full train
    pipe.fit(X_train, y_train)

    # Evaluate on test
    y_prob = pipe.predict_proba(X_test)[:,1]
    metrics = compute_metrics(y_test.values, y_prob, threshold=0.5)
    print("Holdout metrics:", json.dumps(metrics, indent=2))

    # Save artifacts
    args.artifacts.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, args.artifacts / "pipeline.joblib")
    with open(args.artifacts / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Feature importance / coefficients (if available)
    try:
        importances = None
        model = pipe.named_steps["clf"]
        pre = pipe.named_steps["pre"]
        feature_names = list(pre.get_feature_names_out())

        if hasattr(model, "coef_"):
            importances = pd.Series(model.coef_.ravel(), index=feature_names).sort_values(key=lambda s: s.abs(), ascending=False)
        elif hasattr(model, "feature_importances_"):
            importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)

        if importances is not None:
            importances.to_csv(args.artifacts / "feature_importance.csv", header=["importance"])
            print(f"Saved feature importances to {args.artifacts / 'feature_importance.csv'}")
    except Exception as e:
        print(f"Feature importance not saved: {e}")

    print(f"Saved pipeline to {args.artifacts / 'pipeline.joblib'}")
    print(f"Saved metrics to {args.artifacts / 'metrics.json'}")

if __name__ == "__main__":
    main()
