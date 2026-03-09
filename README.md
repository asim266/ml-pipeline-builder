# ML Pipeline Builder - Claude Code Plugin

A Claude Code skill that provides expert guidance for building production-quality machine learning pipelines for tabular classification and regression problems.

## What It Does

- **EDA Automation** - Generates comprehensive exploratory data analysis with missing value reports, target distribution analysis, correlation checks, outlier detection, and high-cardinality warnings
- **Leak-Free Pipelines** - Builds scikit-learn `Pipeline` + `ColumnTransformer` setups that prevent data leakage by keeping all preprocessing inside the pipeline
- **Model Selection** - Decision frameworks for choosing between LogisticRegression, RandomForest, XGBoost, LightGBM, CatBoost with hyperparameter search spaces
- **Imbalanced Data** - SMOTE, class weights, threshold tuning, and imblearn pipeline integration with proper evaluation metrics
- **Evaluation** - Cross-validation, proper metrics (ROC-AUC, F1, MAE, RMSE), SHAP explainability, and learning curve analysis
- **Data Leakage Detection** - Static analysis script and 10-point checklist to catch common leakage patterns

## Installation

```
/plugin install gh:asim266/ml-pipeline-builder
```

## Triggers On

Any mention of: `scikit-learn`, `sklearn`, `XGBoost`, `LightGBM`, `CatBoost`, `pandas`, `feature engineering`, `cross-validation`, `train test split`, `confusion matrix`, `ROC curve`, `SHAP`, `model evaluation`, `pipeline`, `GridSearchCV`, `RandomizedSearchCV`, `Optuna`, `classification report`, `regression metrics`, `data leakage`, `imbalanced data`, `SMOTE`, `StandardScaler`, `OneHotEncoder`, `ColumnTransformer`, or `ML pipeline`.

## Bundled Scripts

- `scripts/quick-eda.py` - Run a quick EDA report on any CSV file (`python quick-eda.py data.csv target_col`)
- `scripts/baseline-comparison.py` - Compare baseline models for classification or regression
- `scripts/data-leakage-check.py` - Static analysis to detect common data leakage patterns in Python files

## Reference Guides

- `references/eda-guide.md` - Step-by-step EDA process with visualization templates
- `references/model-selection-guide.md` - Decision trees, model comparison tables, hyperparameter search spaces, ensemble strategies
- `references/imbalanced-data-guide.md` - SMOTE variants, class weights, threshold tuning, evaluation for imbalanced data

## Disclaimer

This plugin is provided "as is" without warranty of any kind, express or implied. The guidance, code examples, and recommendations are for educational and development purposes only. Always review and test generated code thoroughly before deploying to production. The author(s) accept no responsibility or liability for any damages or other issues arising from the use of this plugin. **Use at your own risk.**

## License

MIT
