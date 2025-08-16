from __future__ import annotations
import argparse
from pathlib import Path
from stresstest.data import SynthConfig, generate_synthetic

def main():
    ap = argparse.ArgumentParser(description="Generate synthetic G-SIB-like panel data for stress prediction.")
    ap.add_argument("--out", type=Path, default=Path("data/gsib_synth.csv"), help="Output CSV path")
    ap.add_argument("--n_banks", type=int, default=30)
    ap.add_argument("--start", type=str, default="2010-01-01")
    ap.add_argument("--end", type=str, default="2024-12-31")
    ap.add_argument("--freq", type=str, default="W")
    args = ap.parse_args()

    cfg = SynthConfig(n_banks=args.n_banks, start=args.start, end=args.end, freq=args.freq)
    df = generate_synthetic(cfg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows to {args.out}")

if __name__ == "__main__":
    main()
