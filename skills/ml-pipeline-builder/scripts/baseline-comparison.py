#!/usr/bin/env python3
"""Compare baseline models for classification or regression.

Usage:
    python baseline-comparison.py data.csv target_column --task classification
    python baseline-comparison.py data.csv target_column --task regression
"""

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


def run_baseline_comparison(filepath, target_col, task):
    df = pd.read_csv(filepath)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Auto-detect feature types
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Drop ID-like columns
    id_cols = [col for col in numeric_features if X[col].nunique() == len(X)]
    if id_cols:
        print(f"Dropping ID-like columns: {id_cols}")
        numeric_features = [c for c in numeric_features if c not in id_cols]

    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")

    # Build preprocessor
    transformers = []
    if numeric_features:
        transformers.append((
            "num",
            Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
            numeric_features,
        ))
    if categorical_features:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            categorical_features,
        ))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    # Split data
    if task == "classification":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        cv = StratifiedKFold(5, shuffle=True, random_state=42)
        scoring = "roc_auc" if y.nunique() == 2 else "f1_weighted"

        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
            "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
        }
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        cv = KFold(5, shuffle=True, random_state=42)
        scoring = "r2"

        models = {
            "Ridge": Ridge(random_state=42),
            "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
        }

    # Try to add XGBoost if available
    try:
        from xgboost import XGBClassifier, XGBRegressor
        if task == "classification":
            models["XGBoost"] = XGBClassifier(
                n_estimators=200, random_state=42, use_label_encoder=False, eval_metric="logloss"
            )
        else:
            models["XGBoost"] = XGBRegressor(n_estimators=200, random_state=42)
    except ImportError:
        print("(XGBoost not installed, skipping)")

    # Try to add LightGBM if available
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
        if task == "classification":
            models["LightGBM"] = LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)
        else:
            models["LightGBM"] = LGBMRegressor(n_estimators=200, random_state=42, verbose=-1)
    except ImportError:
        print("(LightGBM not installed, skipping)")

    # Run comparisons
    print(f"\n{'=' * 60}")
    print(f"BASELINE COMPARISON ({task.upper()}, scoring={scoring})")
    print(f"{'=' * 60}")
    print(f"{'Model':<25} {'CV Mean':>10} {'CV Std':>10}")
    print("-" * 45)

    results = {}
    for name, model in models.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        results[name] = {"mean": scores.mean(), "std": scores.std()}
        print(f"{name:<25} {scores.mean():>10.4f} {scores.std():>10.4f}")

    best = max(results, key=lambda k: results[k]["mean"])
    print(f"\nBest baseline: {best} ({scoring}={results[best]['mean']:.4f})")
    print("Next step: Hyperparameter tuning on the best model(s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline model comparison")
    parser.add_argument("filepath", help="Path to CSV file")
    parser.add_argument("target", help="Target column name")
    parser.add_argument("--task", choices=["classification", "regression"], required=True)
    args = parser.parse_args()
    run_baseline_comparison(args.filepath, args.target, args.task)
