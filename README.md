# Banking System Stress Testing (ML)

A compact, production-ready baseline to **predict high-risk liquidity stress** in banks using supervised learning.
Implements **Logistic Regression** and **Random Forest** within scikit-learn Pipelines and supports time-aware
validation, artifact saving, and reproducible synthetic data generation (if you don't have proprietary data).

> This repo ships with a _synthetic_ dataset generator that mimics plausible banking risk signals.

## Features
- Clean train/evaluate/score commands
- TimeSeriesSplit cross-validation
- Class imbalance handling (`class_weight='balanced'`)
- Metrics: ROC-AUC, PR-AUC, F1, Balanced Accuracy, Brier score & more
- Model + pipeline persisted with `joblib`

## Quickstart
```bash
# 1) Create synthetic data
python scripts/generate_data.py --out data/gsib_synth.csv

# 2) Train a model (logistic or random_forest)
python scripts/train.py --data data/gsib_synth.csv --model logistic

# 3) Score new observations
python scripts/score.py --pipeline artifacts/pipeline.joblib --data data/gsib_synth.csv --out artifacts/scores.csv
```

## Project Layout
```
.
├── data/                      # input CSVs (synthetic generator provided)
├── artifacts/                 # saved pipeline, metrics, feature importances
├── scripts/                   # CLIs: generate_data, train, score
└── src/stresstest/            # library code
```


