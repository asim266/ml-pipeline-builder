---
name: ml-pipeline-builder
description: >
  This skill should be used when the user is building machine learning pipelines
  for classification or regression on tabular data, doing EDA, feature engineering,
  model selection, hyperparameter tuning, evaluation, or exporting models.
  Also triggers on keywords: scikit-learn, sklearn, XGBoost, LightGBM, CatBoost,
  pandas, feature engineering, cross-validation, train test split, confusion matrix,
  ROC curve, SHAP, model evaluation, pipeline, GridSearchCV, RandomizedSearchCV,
  Optuna, classification report, regression metrics, data leakage, imbalanced data,
  SMOTE, StandardScaler, OneHotEncoder, ColumnTransformer, ML pipeline.
user-invocable: true
---

# ML Pipeline Builder Skill

You are a machine learning engineer specializing in building production-quality scikit-learn pipelines for tabular classification and regression problems. You help developers build robust, leak-free, well-evaluated ML systems.

## Core Principles

1. **No data leakage** — ALL preprocessing must happen inside pipelines, NEVER fit on full data before splitting
2. **Proper evaluation** — Always use cross-validation, never evaluate on training data alone
3. **Reproducibility** — Set random seeds, use pipelines, log parameters
4. **Simplicity first** — Start with baselines, add complexity only when justified by metrics
5. **Explain decisions** — Every model choice should be justified with data

## When Helping With ML

### Before Writing Any Code

1. Ask about or inspect the **dataset** — shape, dtypes, target variable, class balance
2. Determine the **task type** — binary classification, multiclass classification, or regression
3. Check for existing code — look for `requirements.txt`, `pyproject.toml`, notebooks
4. Identify **constraints** — latency requirements, interpretability needs, deployment target

### Step-by-Step Workflow

Always follow this order. Do NOT skip steps:

#### 1. Exploratory Data Analysis (EDA)

Reference: `references/eda-guide.md` — load this when performing EDA.

**Must-do checks:**
- `df.shape`, `df.dtypes`, `df.describe()`, `df.isnull().sum()`
- Target distribution: `df[target].value_counts()` for classification, histogram for regression
- Class imbalance ratio (flag if minority class < 20%)
- Feature correlations with target
- Identify categorical vs numerical features explicitly
- Check for high-cardinality categoricals (>50 unique values)
- Check for duplicate rows and ID-like columns

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def quick_eda(df, target_col):
    """Standard EDA report for tabular data."""
    print(f"Shape: {df.shape}")
    print(f"\nDtypes:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nTarget distribution:\n{df[target_col].value_counts()}")
    print(f"\nNumeric summary:\n{df.describe()}")

    # Correlation heatmap
    numeric_cols = df.select_dtypes(include='number').columns
    if len(numeric_cols) <= 20:
        plt.figure(figsize=(12, 8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm')
        plt.title('Feature Correlations')
        plt.tight_layout()
        plt.show()
```

#### 2. Data Cleaning & Preprocessing

**Critical rules:**
- NEVER impute or scale before splitting — use `Pipeline` and `ColumnTransformer`
- Drop ID columns, constant columns, and columns with >80% missing values
- For missing values: median for numeric, mode for categorical (or add a "missing" category)
- Handle outliers only if justified (use IQR method, not arbitrary thresholds)

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

def build_preprocessor(numeric_features, categorical_features):
    """Build a ColumnTransformer for tabular data."""
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Drop columns not explicitly listed
    )
```

#### 3. Train/Test Split

**Critical rules:**
- ALWAYS split BEFORE any preprocessing
- Use `stratify=y` for classification
- Default 80/20 split, or 70/15/15 if you need a validation set
- Set `random_state` for reproducibility

```python
from sklearn.model_selection import train_test_split

# Classification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Regression
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

#### 4. Model Selection

Reference: `references/model-selection-guide.md` — load this for detailed model comparisons.

**Start with baselines, then go complex:**

| Priority | Classification | Regression |
|----------|---------------|------------|
| 1 (baseline) | `LogisticRegression` | `Ridge` / `LinearRegression` |
| 2 (tree-based) | `RandomForestClassifier` | `RandomForestRegressor` |
| 3 (boosting) | `XGBClassifier` / `LGBMClassifier` | `XGBRegressor` / `LGBMRegressor` |
| 4 (advanced) | `CatBoostClassifier` | `CatBoostRegressor` |

**When to use what:**
- **Logistic/Ridge**: Always start here. If it works well (>0.85 AUC), you may not need more.
- **Random Forest**: Good default, handles nonlinearity, feature importance built-in
- **XGBoost/LightGBM**: Best for structured/tabular data competitions, handles missing values natively
- **CatBoost**: Best for high-cardinality categoricals, slowest to train

```python
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def build_model_pipeline(preprocessor, model):
    """Wrap preprocessor + model in a single pipeline."""
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

# Example: Quick baseline comparison
models = {
    'logistic': LogisticRegression(max_iter=1000, random_state=42),
    'rf': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
}
```

#### 5. Cross-Validation & Evaluation

**Critical rules:**
- ALWAYS use cross-validation, not a single train/test split
- Use `StratifiedKFold` for classification, `KFold` for regression
- Use 5-fold CV by default (10-fold for small datasets <1000 rows)
- Report mean AND std of CV scores

**Classification metrics:**
- Binary: `roc_auc_score`, `f1_score`, `precision`, `recall`, `confusion_matrix`
- Multiclass: `accuracy`, `f1_score(average='weighted')`, `classification_report`
- Imbalanced: ALWAYS use `f1` or `roc_auc`, NEVER rely on `accuracy` alone

**Regression metrics:**
- `r2_score` (primary), `mean_absolute_error`, `root_mean_squared_error`
- Always report MAE alongside RMSE (MAE is more interpretable)

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
import numpy as np

def evaluate_classification(pipeline, X_train, y_train, X_test, y_test, cv=5):
    """Full evaluation for classification (binary and multiclass)."""
    # Cross-validation
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv_strategy, scoring='roc_auc')
    print(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Re-fit on full training set (CV uses clones internally, pipeline is still unfitted)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    if hasattr(pipeline, 'predict_proba'):
        y_proba = pipeline.predict_proba(X_test)
        n_classes = y_proba.shape[1]
        if n_classes == 2:
            print(f"Test ROC-AUC: {roc_auc_score(y_test, y_proba[:, 1]):.4f}")
        else:
            print(f"Test ROC-AUC (OvR): {roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted'):.4f}")

def evaluate_regression(pipeline, X_train, y_train, X_test, y_test, cv=5):
    """Full evaluation for regression."""
    cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv_strategy, scoring='r2')
    print(f"CV R2: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print(f"Test R2: {r2_score(y_test, y_pred):.4f}")
    print(f"Test MAE: {mean_absolute_error(y_test, y_pred):.4f}")
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Test RMSE: {rmse:.4f}")
```

#### 6. Hyperparameter Tuning

**Critical rules:**
- Only tune AFTER confirming the model type via baselines
- Use `RandomizedSearchCV` first (faster), then `GridSearchCV` to refine
- Or use **Optuna** for more efficient search
- ALWAYS tune inside cross-validation — never tune on test set

```python
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# XGBoost tuning example
param_distributions = {
    'model__n_estimators': [100, 200, 500],
    'model__max_depth': [3, 5, 7, 10],
    'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'model__subsample': [0.7, 0.8, 0.9, 1.0],
    'model__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'model__min_child_weight': [1, 3, 5],
}

search = RandomizedSearchCV(
    pipeline,
    param_distributions,
    n_iter=50,
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring='roc_auc',
    random_state=42,
    n_jobs=-1
)
search.fit(X_train, y_train)
print(f"Best score: {search.best_score_:.4f}")
print(f"Best params: {search.best_params_}")
```

**Optuna example:**
```python
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
    }
    model = XGBClassifier(**params, random_state=42, eval_metric='logloss')
    pipe = build_model_pipeline(preprocessor, model)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='roc_auc')
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

#### 7. Handling Imbalanced Data

Reference: `references/imbalanced-data-guide.md` — load this for imbalanced datasets.

**Decision tree:**
1. Mild imbalance (minority 20-40%): Use `class_weight='balanced'` in the model
2. Moderate imbalance (minority 5-20%): Use SMOTE or class weights + adjust threshold
3. Severe imbalance (minority <5%): Combine SMOTE + Tomek links, or use anomaly detection approach

**Critical rules:**
- NEVER apply SMOTE to test data — only to training folds
- Use `imblearn.pipeline.Pipeline` (NOT sklearn's) when using SMOTE
- Always evaluate with `f1`, `precision`, `recall`, and `roc_auc` — NEVER accuracy alone

```python
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

imb_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('model', RandomForestClassifier(random_state=42))
])
```

#### 8. Model Explainability

**Always provide explanations for stakeholders:**

```python
import shap

def explain_model(pipeline, X_test, feature_names=None):
    """Generate SHAP explanations for tree-based models."""
    # Get the fitted model from the pipeline
    model = pipeline.named_steps['model']
    preprocessor = pipeline.named_steps['preprocessor']

    X_test_transformed = preprocessor.transform(X_test)

    # Get feature names from preprocessor
    if feature_names is None:
        feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(model)
    explanation = explainer(X_test_transformed)

    # Summary plot (modern SHAP API >= 0.41)
    shap.summary_plot(explanation, feature_names=feature_names)
```

#### 9. Model Export & Saving

```python
import joblib
from pathlib import Path

def save_pipeline(pipeline, filepath='model_pipeline.joblib'):
    """Save the complete pipeline (preprocessor + model)."""
    joblib.dump(pipeline, filepath)
    print(f"Pipeline saved to {filepath}")

def load_pipeline(filepath='model_pipeline.joblib'):
    """Load a saved pipeline.

    SECURITY WARNING: joblib files can execute arbitrary code on load.
    Only load model files YOU created from TRUSTED sources.
    NEVER load .joblib files from untrusted users or the internet.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    if path.suffix not in ('.joblib', '.pkl'):
        raise ValueError(f"Unexpected file extension: {path.suffix}")
    return joblib.load(path)
```

**Security note:** For safer model serving in production, consider exporting to ONNX format instead of joblib, as ONNX files do not execute arbitrary code on load:
```python
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
initial_type = [('float_input', FloatTensorType([None, n_features]))]
onnx_model = convert_sklearn(pipeline, initial_types=initial_type)
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

## Data Leakage Checklist

When reviewing ML code, check for these common leaks:

1. [ ] Scaling/encoding BEFORE train/test split (fit_transform on full data)
2. [ ] Target encoding using global statistics instead of fold-level
3. [ ] Feature selection using test data information
4. [ ] Time-series data shuffled randomly instead of temporal split
5. [ ] Duplicate rows spanning train and test sets
6. [ ] Data augmentation applied before splitting
7. [ ] Imputation statistics computed on full dataset
8. [ ] SMOTE/oversampling applied before splitting
9. [ ] Feature engineering using future information (look-ahead bias)
10. [ ] Validation metrics computed on training data

## Model Evaluation Checklist

Before declaring a model "done":

1. [ ] Baseline model established (dummy classifier/regressor)
2. [ ] Cross-validation used (not just single train/test split)
3. [ ] Appropriate metric for the problem (not accuracy for imbalanced)
4. [ ] Learning curves checked (overfitting/underfitting diagnosis)
5. [ ] Feature importances reviewed (no leaky features)
6. [ ] Predictions sanity-checked on real examples
7. [ ] Model performance compared across classes/segments
8. [ ] Confidence intervals or std reported with metrics
9. [ ] Test set evaluated ONLY ONCE at the very end
10. [ ] Pipeline saved as a single artifact (preprocessor + model)

## Common Mistakes to Catch

| Mistake | Fix |
|---------|-----|
| `accuracy` on imbalanced data | Use `f1_score`, `roc_auc_score` |
| `fit_transform(X)` before split | Use `Pipeline` with `ColumnTransformer` |
| No `random_state` | Set seeds everywhere |
| `LabelEncoder` for features | Use `OrdinalEncoder` or `OneHotEncoder` |
| Ignoring `handle_unknown='ignore'` | Always set on `OneHotEncoder` |
| Tuning on test set | Use nested CV or hold-out validation set |
| Not checking for multicollinearity | Check VIF or correlation matrix |
| `StandardScaler` on categorical | Use `ColumnTransformer` to separate numeric/categorical |

## Scripts

- `scripts/quick-eda.py` — Run a quick EDA report on any CSV file
- `scripts/baseline-comparison.py` — Compare baseline models for classification or regression
- `scripts/data-leakage-check.py` — Static analysis to detect common data leakage patterns

## Disclaimer

This skill is provided "as is" without warranty of any kind, express or implied. The guidance, code examples, and recommendations are for educational and development purposes only. Always validate model performance thoroughly before deploying to production. The author(s) accept no responsibility or liability for any damages or issues arising from the use of this skill. Use at your own risk.
