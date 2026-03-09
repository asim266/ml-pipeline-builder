# Model Selection Guide

## Decision Framework

### Classification

```
Start
  |
  v
Is the dataset small (<1000 rows)?
  YES -> LogisticRegression or SVC (with kernel='rbf')
  NO  -> Continue
  |
  v
Is interpretability critical?
  YES -> LogisticRegression (linear) or DecisionTreeClassifier (nonlinear)
  NO  -> Continue
  |
  v
Are there many categorical features with high cardinality?
  YES -> CatBoostClassifier (handles categoricals natively)
  NO  -> Continue
  |
  v
Default recommendation:
  1. Start with LogisticRegression (baseline)
  2. Try RandomForestClassifier
  3. Try XGBClassifier or LGBMClassifier
  4. Pick the best via cross-validation
```

### Regression

```
Start
  |
  v
Is the relationship likely linear?
  YES -> Ridge or ElasticNet
  NO  -> Continue
  |
  v
Is the dataset small (<1000 rows)?
  YES -> Ridge, SVR, or KNeighborsRegressor
  NO  -> Continue
  |
  v
Default recommendation:
  1. Start with Ridge (baseline)
  2. Try RandomForestRegressor
  3. Try XGBRegressor or LGBMRegressor
  4. Pick the best via cross-validation
```

## Model Comparison Table

### Classification Models

| Model | Pros | Cons | Best For |
|-------|------|------|----------|
| `LogisticRegression` | Fast, interpretable, good baseline, built-in regularization | Linear decision boundary only | Baseline, interpretable models, sparse data |
| `RandomForestClassifier` | Handles nonlinearity, feature importance, robust to outliers | Can overfit on noisy data, slow prediction | General purpose, feature selection |
| `GradientBoostingClassifier` | Strong performance, handles mixed features | Slow training, needs careful tuning | When sklearn-only is required |
| `XGBClassifier` | Best tabular performance, handles missing values, GPU support | Many hyperparameters, can overfit | Competitions, production systems |
| `LGBMClassifier` | Fastest boosting, low memory, handles large datasets | Leaf-wise growth can overfit on small data | Large datasets, fast training needed |
| `CatBoostClassifier` | Handles categoricals natively, ordered boosting reduces leakage | Slowest to train | High-cardinality categoricals |
| `SVC` | Strong with RBF kernel on small data | Does not scale (O(n^2)), no native probabilities | Small datasets, clear margin problems |

### Regression Models

| Model | Pros | Cons | Best For |
|-------|------|------|----------|
| `Ridge` | Fast, interpretable, handles multicollinearity | Linear only | Baseline, interpretable models |
| `ElasticNet` | Built-in feature selection (L1+L2) | Linear only | Sparse features, feature selection |
| `RandomForestRegressor` | Handles nonlinearity, robust | Prediction range limited to training data range | General purpose |
| `XGBRegressor` | Best tabular performance, flexible | Many hyperparameters | Best overall for tabular regression |
| `LGBMRegressor` | Fast, memory efficient | Needs min_data_in_leaf tuning | Large datasets |
| `CatBoostRegressor` | Handles categoricals | Slow training | High-cardinality categoricals |
| `SVR` | Good on small data with RBF | Does not scale | Small datasets |

## Hyperparameter Search Spaces

### Logistic Regression
```python
{
    'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'model__penalty': ['l1', 'l2'],
    'model__solver': ['liblinear', 'saga'],
}
```

### Random Forest
```python
{
    'model__n_estimators': [100, 200, 500],
    'model__max_depth': [5, 10, 20, None],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__max_features': ['sqrt', 'log2', None],
}
```

### XGBoost
```python
{
    'model__n_estimators': [100, 200, 500, 1000],
    'model__max_depth': [3, 5, 7, 10],
    'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'model__subsample': [0.7, 0.8, 0.9, 1.0],
    'model__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'model__min_child_weight': [1, 3, 5, 7],
    'model__gamma': [0, 0.1, 0.2],
    'model__reg_alpha': [0, 0.01, 0.1],
    'model__reg_lambda': [0.1, 1, 10],
}
```

### LightGBM
```python
{
    'model__n_estimators': [100, 200, 500, 1000],
    'model__max_depth': [-1, 5, 10, 20],
    'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'model__num_leaves': [15, 31, 63, 127],
    'model__subsample': [0.7, 0.8, 0.9, 1.0],
    'model__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'model__min_child_samples': [5, 10, 20, 50],
    'model__reg_alpha': [0, 0.01, 0.1],
    'model__reg_lambda': [0, 0.01, 0.1],
}
```

### CatBoost
```python
{
    'model__iterations': [200, 500, 1000],
    'model__depth': [4, 6, 8, 10],
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__l2_leaf_reg': [1, 3, 5, 7],
    'model__border_count': [32, 64, 128],
}
```

## Ensemble Strategies

### Stacking (when you need the best performance)
```python
from sklearn.ensemble import StackingClassifier

estimators = [
    ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
    ('xgb', XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False, eval_metric='logloss')),
    ('lgbm', LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)),
]

stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5,
    stack_method='predict_proba'
)
```

### Voting (simpler ensemble)
```python
from sklearn.ensemble import VotingClassifier

voting = VotingClassifier(
    estimators=estimators,
    voting='soft'  # Use predicted probabilities
)
```

## When to Stop

- **Baseline is already good** (>0.95 AUC): Check for data leakage before celebrating
- **Diminishing returns**: If tuning improves <0.5% over default, stop
- **Interpretability needed**: Stick with Logistic/Ridge or single decision tree
- **Training time constraint**: Use LightGBM over XGBoost, or reduce hyperparameter space
