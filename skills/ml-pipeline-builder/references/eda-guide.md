# EDA Guide for Tabular Data

## Step-by-Step EDA Process

### 1. Dataset Overview
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data.csv')

print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print(f"\nColumn types:\n{df.dtypes.value_counts()}")
print(f"\nFirst 5 rows:\n{df.head()}")
```

### 2. Missing Value Analysis
```python
def missing_value_report(df):
    """Generate a missing value summary."""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    report = pd.DataFrame({
        'missing_count': missing,
        'missing_pct': missing_pct.round(2)
    })
    return report[report['missing_count'] > 0].sort_values('missing_pct', ascending=False)

# Decision rules:
# - >80% missing: DROP the column
# - 20-80% missing: Consider if missingness is informative, add indicator column
# - <20% missing: Impute (median for numeric, mode for categorical)
```

### 3. Target Variable Analysis

**Classification:**
```python
def analyze_classification_target(df, target_col):
    counts = df[target_col].value_counts()
    ratios = df[target_col].value_counts(normalize=True)
    print("Class distribution:")
    for cls in counts.index:
        print(f"  {cls}: {counts[cls]} ({ratios[cls]:.1%})")

    imbalance_ratio = counts.min() / counts.max()
    if imbalance_ratio < 0.2:
        print(f"\n** WARNING: Severe imbalance detected (ratio: {imbalance_ratio:.3f})")
        print("   Consider: class_weight='balanced', SMOTE, or threshold tuning")
    elif imbalance_ratio < 0.4:
        print(f"\n** NOTICE: Moderate imbalance (ratio: {imbalance_ratio:.3f})")
        print("   Consider: class_weight='balanced'")
```

**Regression:**
```python
def analyze_regression_target(df, target_col):
    target = df[target_col]
    print(f"Mean: {target.mean():.4f}")
    print(f"Median: {target.median():.4f}")
    print(f"Std: {target.std():.4f}")
    print(f"Skewness: {target.skew():.4f}")
    print(f"Kurtosis: {target.kurtosis():.4f}")

    if abs(target.skew()) > 1:
        print("\n** WARNING: Target is highly skewed")
        print("   Consider: log transform, Box-Cox, or Yeo-Johnson")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    target.hist(bins=50, ax=axes[0])
    axes[0].set_title('Target Distribution')
    from scipy import stats
    stats.probplot(target, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot')
    plt.tight_layout()
    plt.show()
```

### 4. Feature Analysis

```python
def analyze_features(df, target_col):
    """Categorize and analyze features."""
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.drop(target_col, errors='ignore').tolist()
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

    print(f"Numeric features ({len(numeric_features)}): {numeric_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

    # High cardinality check
    for col in categorical_features:
        n_unique = df[col].nunique()
        if n_unique > 50:
            print(f"\n** WARNING: {col} has {n_unique} unique values (high cardinality)")
            print("   Consider: target encoding, frequency encoding, or grouping rare categories")

    # Constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        print(f"\n** WARNING: Constant columns (drop these): {constant_cols}")

    # ID-like columns
    id_cols = [col for col in numeric_features if df[col].nunique() == len(df)]
    if id_cols:
        print(f"\n** WARNING: Possible ID columns (drop these): {id_cols}")

    return numeric_features, categorical_features
```

### 5. Correlation Analysis

```python
def correlation_analysis(df, target_col, numeric_features):
    """Analyze feature correlations with target and each other."""
    # Correlation with target
    correlations = df[numeric_features + [target_col]].corr()[target_col].drop(target_col)
    correlations = correlations.abs().sort_values(ascending=False)
    print("Top correlations with target:")
    print(correlations.head(15))

    # Multicollinearity check
    corr_matrix = df[numeric_features].corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_pairs = [
        (col, row, corr_matrix.loc[row, col])
        for col in upper_triangle.columns
        for row in upper_triangle.index
        if upper_triangle.loc[row, col] > 0.9
    ]
    if high_corr_pairs:
        print("\n** WARNING: Highly correlated feature pairs (>0.9):")
        for col, row, corr in high_corr_pairs:
            print(f"   {col} <-> {row}: {corr:.3f}")
        print("   Consider dropping one from each pair")
```

### 6. Outlier Detection

```python
def detect_outliers_iqr(df, numeric_features):
    """Detect outliers using IQR method."""
    outlier_report = {}
    for col in numeric_features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        if n_outliers > 0:
            outlier_report[col] = {
                'count': n_outliers,
                'pct': (n_outliers / len(df)) * 100,
                'lower_bound': lower,
                'upper_bound': upper
            }
    if outlier_report:
        print("Outlier summary (IQR method):")
        for col, info in sorted(outlier_report.items(), key=lambda x: x[1]['pct'], reverse=True):
            print(f"  {col}: {info['count']} outliers ({info['pct']:.1f}%)")
    return outlier_report
```

### 7. Visualization Templates

```python
def plot_numeric_distributions(df, numeric_features, target_col, task='classification'):
    """Plot distributions of numeric features split by target."""
    n_cols = 3
    n_rows = (len(numeric_features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_features):
        if task == 'classification':
            for cls in df[target_col].unique():
                df[df[target_col] == cls][col].hist(alpha=0.5, bins=30, ax=axes[i], label=str(cls))
            axes[i].legend()
        else:
            axes[i].scatter(df[col], df[target_col], alpha=0.3, s=5)
            axes[i].set_ylabel(target_col)
        axes[i].set_title(col)

    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.show()
```
