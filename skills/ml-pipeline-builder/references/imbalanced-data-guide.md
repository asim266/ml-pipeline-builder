# Handling Imbalanced Data

## Severity Assessment

| Minority Ratio | Severity | Recommended Approach |
|---------------|----------|---------------------|
| 20-40% | Mild | `class_weight='balanced'` |
| 5-20% | Moderate | Class weights + threshold tuning, or SMOTE |
| 1-5% | Severe | SMOTE + undersampling combo, or cost-sensitive learning |
| <1% | Extreme | Anomaly detection approach, or heavy oversampling + ensemble |

## Technique 1: Class Weights (Simplest)

```python
# Most sklearn models support this
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Automatically adjusts weights inversely proportional to class frequency
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model = RandomForestClassifier(class_weight='balanced', random_state=42)

# XGBoost equivalent
from xgboost import XGBClassifier
# scale_pos_weight = count(negative) / count(positive)
scale = y_train.value_counts()[0] / y_train.value_counts()[1]
model = XGBClassifier(scale_pos_weight=scale, random_state=42)
```

## Technique 2: SMOTE (Synthetic Oversampling)

**CRITICAL: Use imblearn Pipeline, not sklearn Pipeline**

```python
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek, SMOTEENN

# Basic SMOTE
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('model', RandomForestClassifier(random_state=42))
])

# SMOTE + Tomek links (cleans decision boundary)
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smotetomek', SMOTETomek(random_state=42)),
    ('model', RandomForestClassifier(random_state=42))
])

# SMOTE + ENN (more aggressive cleaning)
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smoteenn', SMOTEENN(random_state=42)),
    ('model', RandomForestClassifier(random_state=42))
])
```

**SMOTE variants:**
- `SMOTE`: Standard, works well for most cases
- `BorderlineSMOTE`: Only oversamples near the decision boundary (often better)
- `ADASYN`: Adaptive, generates more samples in harder regions
- `SMOTETomek`: SMOTE + removes Tomek links (noisy samples)
- `SMOTEENN`: SMOTE + removes samples misclassified by ENN (most aggressive)

## Technique 3: Threshold Tuning

**IMPORTANT:** Find the optimal threshold on a VALIDATION set, not the test set. Using test labels to select the threshold is data leakage.

```python
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split

# Split training data into train + validation
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Train and find threshold on validation set
pipeline.fit(X_tr, y_tr)
y_val_proba = pipeline.predict_proba(X_val)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_proba)
# Drop last element (precisions/recalls have len = thresholds + 1)
f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
optimal_idx = f1_scores.argmax()
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal threshold: {optimal_threshold:.4f}")
print(f"F1 at optimal threshold: {f1_scores[optimal_idx]:.4f}")

# Re-train on full training set, then apply threshold to TEST set (evaluated once)
pipeline.fit(X_train, y_train)
y_test_proba = pipeline.predict_proba(X_test)[:, 1]
y_pred_custom = (y_test_proba >= optimal_threshold).astype(int)
```

## Technique 4: Ensemble Methods for Imbalance

```python
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier

# Balanced Random Forest (undersamples each bootstrap)
model = BalancedRandomForestClassifier(n_estimators=200, random_state=42)

# Easy Ensemble (ensemble of AdaBoost on balanced subsets)
model = EasyEnsembleClassifier(n_estimators=10, random_state=42)
```

## Evaluation for Imbalanced Data

**NEVER use accuracy. Use these instead:**

```python
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)

def evaluate_imbalanced(y_test, y_pred, y_proba=None):
    """Comprehensive evaluation for imbalanced classification."""
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")

    if y_proba is not None:
        print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
        print(f"PR-AUC: {average_precision_score(y_test, y_proba):.4f}")

    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
```

**Key metrics for imbalanced data:**
- **PR-AUC** (`average_precision_score`): Better than ROC-AUC for heavy imbalance
- **F1 Score**: Harmonic mean of precision and recall
- **Recall**: When false negatives are costly (fraud detection, disease screening)
- **Precision**: When false positives are costly (spam filtering)

## Strategy Comparison Template

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

strategies = {
    'baseline': Pipeline([('pre', preprocessor), ('model', LogisticRegression(max_iter=1000))]),
    'class_weight': Pipeline([('pre', preprocessor), ('model', LogisticRegression(class_weight='balanced', max_iter=1000))]),
    'smote': ImbPipeline([('pre', preprocessor), ('smote', SMOTE(random_state=42)), ('model', LogisticRegression(max_iter=1000))]),
    'smote_tomek': ImbPipeline([('pre', preprocessor), ('st', SMOTETomek(random_state=42)), ('model', LogisticRegression(max_iter=1000))]),
}

cv = StratifiedKFold(5, shuffle=True, random_state=42)
for name, pipe in strategies.items():
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='f1')
    print(f"{name:20s}: F1 = {scores.mean():.4f} (+/- {scores.std():.4f})")
```
