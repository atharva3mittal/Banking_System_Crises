from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# Core feature set used by the pipeline
NUMERIC_FEATURES: List[str] = [
    "lcr", "nsfr", "loan_to_deposit", "st_wholesale_funding_ratio",
    "uninsured_deposit_ratio", "deposit_concentration_hhi",
    "cds_5y_bp", "equity_drawdown_30d", "vol_30d",
    "duration_gap_years", "hqla_to_assets", "liquidity_coverage_gap",
    "deposits_outflow_rate_7d", "market_sentiment",
    "policy_rate_change_bp", "yield_curve_slope",
    "funding_cost_spread_bp", "tier1_leverage_ratio",
    "npl_ratio"
]

CATEGORICAL_FEATURES: List[str] = ["region", "size_bucket"]

TARGET_COL = "high_risk_liquidity_event"
DATE_COL = "date"

def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer for numeric scaling + categorical one-hot encoding."""
    num_pipe = StandardScaler(with_mean=True, with_std=True)
    cat_pipe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, NUMERIC_FEATURES),
            ("cat", cat_pipe, CATEGORICAL_FEATURES)
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    return preprocessor

def load_dataset(path: str) -> pd.DataFrame:
    """Load CSV dataset and enforce dtypes."""
    df = pd.read_csv(path)
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    return df

def train_test_split_by_date(df: pd.DataFrame, cutoff: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train/test by a date cutoff (inclusive for train)."""
    cutoff_ts = pd.to_datetime(cutoff)
    train = df[df[DATE_COL] <= cutoff_ts].copy()
    test = df[df[DATE_COL] > cutoff_ts].copy()
    return train, test

# ---------------------------- Synthetic Data Generator ----------------------------

@dataclass
class SynthConfig:
    n_banks: int = 30
    start: str = "2010-01-01"
    end: str = "2024-12-31"
    freq: str = "W"  # weekly
    seed: int = 42

def _sigmoid(x):
    return 1 / (1 + np.exp(-x))

def generate_synthetic(config: SynthConfig) -> pd.DataFrame:
    """
    Generate a realistic-looking panel dataset for stress prediction.
    Each row ~ (bank, week) with engineered risk drivers + binary event label.
    """
    rng = np.random.default_rng(config.seed)
    dates = pd.date_range(config.start, config.end, freq=config.freq)
    bank_ids = [f"BANK_{i:02d}" for i in range(config.n_banks)]
    regions = ["NA", "EU", "APAC"]
    sizes = ["GSIB_Large", "GSIB_Medium"]

    records = []
    for b in bank_ids:
        region = rng.choice(regions, p=[0.45, 0.35, 0.20])
        size = rng.choice(sizes, p=[0.6, 0.4])
        # bank-level baseline risk appetite & franchise stickiness
        base_uninsured = rng.uniform(0.20, 0.60)
        stickiness = rng.uniform(0.3, 0.9)  # higher = more stable deposits

        # time-varying macro shocks (global, shared across banks but with idiosyncratic noise)
        macro_drift = rng.normal(0.0, 0.03, size=len(dates)).cumsum()

        for t, dt in enumerate(dates):
            # Core prudential ratios
            lcr = np.clip(rng.normal(130 - 10*np.sin(t/60), 10), 50, 300)
            nsfr = np.clip(rng.normal(115 - 8*np.cos(t/50), 8), 70, 180)
            loan_to_deposit = np.clip(rng.normal(0.95 + 0.1*np.sin(t/40), 0.08), 0.5, 1.5)

            # Structural funding risks
            st_wholesale = np.clip(rng.normal(0.20 + 0.05*np.sin(t/30), 0.04), 0.05, 0.6)
            uninsured_ratio = np.clip(base_uninsured + rng.normal(0, 0.04), 0.05, 0.95)
            deposit_conc = np.clip(rng.normal(0.15 + 0.05*(1-stickiness), 0.03), 0.05, 0.5)

            # Market signals
            cds = np.clip(80 + 100*macro_drift[t] + rng.normal(0, 30), 20, 800)  # bps
            eq_dd = np.clip(rng.normal(0.0 + 0.3*macro_drift[t], 0.10), -0.6, 0.6)  # 30d drawdown
            vol_30d = np.clip(rng.normal(0.25 + 0.2*abs(macro_drift[t]), 0.05), 0.05, 1.0)

            # Balance sheet structure
            dur_gap = np.clip(rng.normal(1.5 + 0.7*np.sin(t/90), 0.4), -1.0, 4.0)
            hqla_assets = np.clip(rng.normal(0.18 + 0.03*np.cos(t/25), 0.03), 0.05, 0.5)
            liq_cov_gap = np.clip(rng.normal(0.0, 0.2), -0.6, 0.6)

            # Flows & sentiment
            outflow_7d = np.clip(rng.normal(0.01 + 0.03*vol_30d, 0.015), 0.0, 0.25)
            sentiment = np.clip(rng.normal(0.0 - 0.5*macro_drift[t], 0.6), -2.5, 2.5)

            # Rates & curve
            policy_chg = np.clip(rng.normal(0 + 15*np.sin(t/70), 15), -100, 100)  # bps, weekly change
            slope = np.clip(rng.normal(0.8 - 0.6*np.sin(t/80), 0.4), -1.5, 2.0)
            funding_spread = np.clip(30 + 60*vol_30d + rng.normal(0, 15), 0, 500)  # bps

            # Capital & credit
            tier1_lev = np.clip(rng.normal(6.0 - 0.4*macro_drift[t], 0.8), 3.0, 12.0)
            npl = np.clip(rng.normal(0.025 + 0.02*max(macro_drift[t], 0), 0.01), 0.0, 0.15)

            # Liquidity event probability (stylized logit)
            logit = (
                -4.0
                - 0.012*(lcr-100)
                - 0.010*(nsfr-100)
                + 1.8*st_wholesale
                + 2.0*uninsured_ratio
                + 1.4*deposit_conc
                + 0.004*(cds-120)
                + 0.9*eq_dd
                + 0.7*vol_30d
                + 0.15*dur_gap
                - 0.7*hqla_assets
                + 0.9*liq_cov_gap
                + 2.5*outflow_7d
                - 0.10*sentiment
                + 0.003*policy_chg
                - 0.20*slope
                + 0.004*funding_spread
                - 0.10*(tier1_lev-6.0)
                + 1.2*npl
                + (0.2 if region == "EU" else 0.0)  # mild region effect
            )
            p_event = _sigmoid(logit)
            y = rng.binomial(1, p_event)

            records.append({
                "date": dt, "bank_id": b, "region": region, "size_bucket": size,
                "lcr": lcr, "nsfr": nsfr, "loan_to_deposit": loan_to_deposit,
                "st_wholesale_funding_ratio": st_wholesale,
                "uninsured_deposit_ratio": uninsured_ratio,
                "deposit_concentration_hhi": deposit_conc,
                "cds_5y_bp": cds, "equity_drawdown_30d": eq_dd, "vol_30d": vol_30d,
                "duration_gap_years": dur_gap, "hqla_to_assets": hqla_assets,
                "liquidity_coverage_gap": liq_cov_gap,
                "deposits_outflow_rate_7d": outflow_7d,
                "market_sentiment": sentiment,
                "policy_rate_change_bp": policy_chg,
                "yield_curve_slope": slope,
                "funding_cost_spread_bp": funding_spread,
                "tier1_leverage_ratio": tier1_lev,
                "npl_ratio": npl,
                "high_risk_liquidity_event": y
            })

    df = pd.DataFrame.from_records(records)
    df.sort_values(["date", "bank_id"], inplace=True)
    return df
