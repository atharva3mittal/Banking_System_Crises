from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import pandas as pd
from stresstest.data import load_dataset, DATE_COL, TARGET_COL

def main():
    ap = argparse.ArgumentParser(description="Score new data with a saved pipeline.")
    ap.add_argument("--pipeline", type=Path, required=True, help="Path to artifacts/pipeline.joblib")
    ap.add_argument("--data", type=Path, required=True, help="CSV with the same feature schema used in training")
    ap.add_argument("--out", type=Path, default=Path("artifacts/scores.csv"))
    args = ap.parse_args()

    pipe = joblib.load(args.pipeline)
    df = load_dataset(str(args.data))

    # if target present, we won't use it for scoring
    FEATURES = [c for c in df.columns if c not in {TARGET_COL, DATE_COL, "bank_id"}]
    X = df[FEATURES]
    scores = pipe.predict_proba(X)[:,1]
    out = df[[DATE_COL, "bank_id"]].copy() if "bank_id" in df.columns else df[[DATE_COL]].copy()
    out["prob_high_risk"] = scores
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote scores to {args.out}")

if __name__ == "__main__":
    main()
