#!/usr/bin/env python3
"""Quick EDA report for any CSV file.

Usage:
    python quick-eda.py data.csv target_column
    python quick-eda.py data.csv target_column --task classification
"""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def quick_eda(filepath, target_col, task="auto"):
    df = pd.read_csv(filepath)

    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"\nColumn types:\n{df.dtypes.value_counts().to_string()}")

    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_report = pd.DataFrame({"count": missing, "pct": missing_pct})
    missing_report = missing_report[missing_report["count"] > 0].sort_values(
        "pct", ascending=False
    )
    if len(missing_report) > 0:
        print(f"\nMissing values:\n{missing_report.to_string()}")
    else:
        print("\nNo missing values.")

    # Target analysis
    print("\n" + "=" * 60)
    print("TARGET ANALYSIS")
    print("=" * 60)

    if task == "auto":
        if df[target_col].nunique() <= 20:
            task = "classification"
        else:
            task = "regression"
        print(f"Auto-detected task: {task}")

    if task == "classification":
        counts = df[target_col].value_counts()
        ratios = df[target_col].value_counts(normalize=True)
        print(f"\nClass distribution:")
        for cls in counts.index:
            print(f"  {cls}: {counts[cls]} ({ratios[cls]:.1%})")
        imbalance_ratio = counts.min() / counts.max()
        if imbalance_ratio < 0.2:
            print(f"\n** SEVERE IMBALANCE (ratio: {imbalance_ratio:.3f})")
        elif imbalance_ratio < 0.4:
            print(f"\n** MODERATE IMBALANCE (ratio: {imbalance_ratio:.3f})")
    else:
        target = df[target_col]
        print(f"Mean: {target.mean():.4f}, Median: {target.median():.4f}")
        print(f"Std: {target.std():.4f}, Skew: {target.skew():.4f}")
        if abs(target.skew()) > 1:
            print("** WARNING: Target is highly skewed. Consider log transform.")

    # Feature analysis
    print("\n" + "=" * 60)
    print("FEATURE ANALYSIS")
    print("=" * 60)

    numeric = df.select_dtypes(include=["int64", "float64"]).columns.drop(
        target_col, errors="ignore"
    )
    categorical = df.select_dtypes(include=["object", "category"]).columns

    print(f"Numeric features ({len(numeric)}): {list(numeric)}")
    print(f"Categorical features ({len(categorical)}): {list(categorical)}")

    # High cardinality
    for col in categorical:
        n_unique = df[col].nunique()
        if n_unique > 50:
            print(f"** HIGH CARDINALITY: {col} ({n_unique} unique values)")

    # Constant columns
    constant = [col for col in df.columns if df[col].nunique() <= 1]
    if constant:
        print(f"** CONSTANT COLUMNS (drop): {constant}")

    # ID-like columns
    id_like = [col for col in numeric if df[col].nunique() == len(df)]
    if id_like:
        print(f"** POSSIBLE ID COLUMNS (drop): {id_like}")

    # Duplicates
    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        print(f"** DUPLICATE ROWS: {n_dupes} ({n_dupes/len(df)*100:.1f}%)")

    # Top correlations with target
    if len(numeric) > 0 and target_col in df.select_dtypes(include="number").columns:
        print("\n" + "=" * 60)
        print("TOP CORRELATIONS WITH TARGET")
        print("=" * 60)
        corrs = df[numeric].corrwith(df[target_col]).abs().sort_values(ascending=False)
        print(corrs.head(10).to_string())

    print(f"\n{'=' * 60}")
    print(f"Task: {task} | Target: {target_col} | Ready for pipeline building")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick EDA for tabular data")
    parser.add_argument("filepath", help="Path to CSV file")
    parser.add_argument("target", help="Target column name")
    parser.add_argument(
        "--task",
        choices=["classification", "regression", "auto"],
        default="auto",
        help="Task type",
    )
    args = parser.parse_args()
    quick_eda(args.filepath, args.target, args.task)
