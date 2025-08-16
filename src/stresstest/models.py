from __future__ import annotations
from typing import Dict, Tuple
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from .data import build_preprocessor

def build_pipeline(model_name: str) -> Tuple[Pipeline, str]:
    """
    Assemble a sklearn Pipeline with preprocessing + chosen model.
    Returns (pipeline, tag) where tag is a short identifier used in filenames.
    """
    pre = build_preprocessor()

    if model_name.lower() in {"logistic", "logit", "logreg"}:
        clf = LogisticRegression(
            max_iter=2000,
            n_jobs=None,
            class_weight="balanced",
            solver="lbfgs"
        )
        tag = "logistic"
    elif model_name.lower() in {"rf", "random_forest", "random-forest"}:
        clf = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=42
        )
        tag = "random_forest"
    else:
        raise ValueError(f"Unknown model '{model_name}'. Use 'logistic' or 'random_forest'.")

    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
    return pipe, tag
