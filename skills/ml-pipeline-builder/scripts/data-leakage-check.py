#!/usr/bin/env python3
"""Static analysis to detect common data leakage patterns in ML code.

NOTE: This checker analyzes single lines only and may miss multi-line expressions.
It is a best-effort heuristic, not a substitute for manual code review.

Usage:
    python data-leakage-check.py path/to/your/script.py
    python data-leakage-check.py path/to/notebook.py
"""

import argparse
import re
import sys
from pathlib import Path


LEAKAGE_PATTERNS = [
    {
        "pattern": r"\.fit_transform\(.*[Xx]\b",
        "before_split": True,
        "message": "fit_transform() called — ensure this is INSIDE a Pipeline, not before train/test split",
        "severity": "HIGH",
    },
    {
        "pattern": r"StandardScaler\(\)\.fit\(",
        "before_split": True,
        "message": "Scaler fitted outside Pipeline — risk of fitting on full data before split",
        "severity": "HIGH",
    },
    {
        "pattern": r"LabelEncoder\(\)",
        "before_split": False,
        "message": "LabelEncoder used — use OrdinalEncoder for features, LabelEncoder is only for targets",
        "severity": "MEDIUM",
    },
    {
        "pattern": r"\.fillna\(.*\.mean\(\)|\.fillna\(.*\.median\(\)",
        "before_split": True,
        "message": "Imputation with global statistics — should use SimpleImputer inside Pipeline",
        "severity": "HIGH",
    },
    {
        "pattern": r"SMOTE.*\.fit_resample\(",
        "before_split": True,
        "message": "SMOTE applied outside Pipeline — must use imblearn.pipeline.Pipeline",
        "severity": "HIGH",
    },
    {
        "pattern": r"\.score\(.*[Xx]_train",
        "before_split": False,
        "message": "Evaluating on training data — use cross_val_score or test set instead",
        "severity": "MEDIUM",
    },
    {
        "pattern": r"accuracy_score|scoring=['\"]accuracy['\"]",
        "before_split": False,
        "message": "Using accuracy metric — verify data is balanced, otherwise use f1/roc_auc",
        "severity": "LOW",
    },
    {
        "pattern": r"train_test_split(?!.*stratify)",
        "before_split": False,
        "message": "train_test_split without stratify — add stratify=y for classification tasks",
        "severity": "LOW",
    },
    {
        "pattern": r"train_test_split(?!.*random_state)",
        "before_split": False,
        "message": "train_test_split without random_state — results not reproducible",
        "severity": "LOW",
    },
    {
        "pattern": r"\.drop_duplicates\(\)",
        "before_split": True,
        "message": "Dropping duplicates before split — duplicates could span train/test causing leakage",
        "severity": "MEDIUM",
    },
    {
        "pattern": r"SelectKBest|SelectFromModel|RFE",
        "before_split": True,
        "message": "Feature selection — ensure it's inside Pipeline/CV, not fitted on full data",
        "severity": "HIGH",
    },
    {
        "pattern": r"\.get_dummies\(",
        "before_split": True,
        "message": "pd.get_dummies() — use OneHotEncoder inside Pipeline instead for consistency",
        "severity": "MEDIUM",
    },
]


def check_file(filepath):
    content = Path(filepath).read_text(encoding="utf-8", errors="replace")
    lines = content.split("\n")

    findings = []
    split_line = None

    # Find where train_test_split happens
    for i, line in enumerate(lines):
        if "train_test_split" in line:
            split_line = i
            break

    for i, line in enumerate(lines):
        # Skip comments and strings
        stripped = line.strip()
        if stripped.startswith("#"):
            continue

        for check in LEAKAGE_PATTERNS:
            if re.search(check["pattern"], line):
                is_before_split = split_line is not None and i < split_line
                if check["before_split"] and not is_before_split:
                    continue  # Only flag if it's before split

                findings.append({
                    "line": i + 1,
                    "severity": check["severity"],
                    "message": check["message"],
                    "code": stripped[:80],
                })

    return findings


def main():
    parser = argparse.ArgumentParser(description="Check for data leakage patterns")
    parser.add_argument("filepath", help="Path to Python file to check")
    args = parser.parse_args()

    if not Path(args.filepath).exists():
        print(f"File not found: {args.filepath}")
        sys.exit(1)

    findings = check_file(args.filepath)

    if not findings:
        print("No data leakage patterns detected.")
        return

    print(f"Found {len(findings)} potential issue(s):\n")
    for f in sorted(findings, key=lambda x: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}[x["severity"]]):
        print(f"[{f['severity']}] Line {f['line']}: {f['message']}")
        print(f"  > {f['code']}")
        print()


if __name__ == "__main__":
    main()
