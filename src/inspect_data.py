"""Phase 1 dataset inspection; never trains on unverified columns."""
import argparse
from pathlib import Path
import pandas as pd


def inspect_csv(path: str) -> None:
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, encoding="latin1", low_memory=False)
    print(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
    print("\nColumns:")
    for name in df.columns:
        print(f" - {name}")
    print("\nMissing values (top 15):")
    print(df.isna().sum().sort_values(ascending=False).head(15).to_string())
    if "Late_delivery_risk" in df:
        print("\nTarget distribution:")
        print(df["Late_delivery_risk"].value_counts(dropna=False).to_string())
    if "Order Id" in df:
        print(f"\nUnique Order Id values: {df['Order Id'].nunique():,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()
    inspect_csv(args.csv)
