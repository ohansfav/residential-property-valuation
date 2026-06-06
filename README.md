<div align="center">

# Residential Property Valuation

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=24&pause=1200&color=0F766E&center=true&vCenter=true&width=900&lines=Kaggle+House+Prices+Competition+Solution;Public+Leaderboard+Score%3A+0.12254;Gradient+Boosting+Selected+via+Cross-Validation;Portfolio-Ready+Tabular+ML+Project" alt="Typing banner" />

![Kaggle](https://img.shields.io/badge/Kaggle-House%20Prices-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)
![Public Score](https://img.shields.io/badge/Public%20Score-0.12254-0F766E?style=for-the-badge)
![Best Model](https://img.shields.io/badge/Best%20Model-Gradient%20Boosting-1D4ED8?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Submitted-success?style=for-the-badge)

</div>

This repository solves the Kaggle **House Prices - Advanced Regression Techniques** competition with a reproducible ML pipeline, model benchmarking, and a verified leaderboard submission.

## Achievement Snapshot

- Competition solved end to end from local pipeline to live Kaggle submission.
- **Public leaderboard score:** `0.12254`
- **Best cross-validated model:** `GradientBoostingRegressor`
- **Best CV RMSLE:** `0.12437` using 3-fold validation
- **Models benchmarked:** Ridge, Elastic Net, Gradient Boosting, Random Forest
- **Submission artifact generated:** `outputs/submission.csv`

## Why This Repo Stands Out

- Uses the real Kaggle competition dataset, not a toy housing sample.
- Handles mixed tabular data with explicit preprocessing for numeric and categorical features.
- Optimizes for the competition metric by training on `log1p(SalePrice)`.
- Benchmarks multiple models before selecting the best performer.
- Produces a ready-to-upload submission file in one command.

## Competition Link

- [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

## Project Structure

```text
residential-property-valuation/
├── pipeline.py
├── requirements.txt
├── .gitignore
├── data/
│   ├── data_description.txt
│   ├── sample_submission.csv
│   ├── test.csv
│   └── train.csv
└── outputs/
   └── submission.csv
```

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run with existing competition files in `data/`:

```bash
python pipeline.py --skip-download
```

Run with Kaggle API download enabled:

```bash
python pipeline.py
```

## Modeling Pipeline

1. **Target transformation**
  Train on `log1p(SalePrice)` so model evaluation aligns with RMSLE-style leaderboard behavior.

2. **Preprocessing**
  Numeric columns use median imputation plus scaling.
  Categorical columns use mode imputation plus one-hot encoding.

3. **Model benchmark**
  The pipeline evaluates:
  `Ridge`, `ElasticNet`, `GradientBoostingRegressor`, `RandomForestRegressor`

4. **Model selection**
  K-fold cross-validation selects the lowest RMSLE-scale error.

5. **Submission generation**
  Predictions are transformed back with `expm1` and saved to `outputs/submission.csv`.

## Best Run

| Metric | Result |
| --- | --- |
| Training rows | 1460 |
| Test rows | 1459 |
| CV folds used for winning run | 3 |
| Best CV RMSLE | 0.12437 |
| Public leaderboard score | 0.12254 |
| Winning model | Gradient Boosting |

## Kaggle Submission Command

```bash
kaggle competitions submit \
  -c house-prices-advanced-regression-techniques \
  -f outputs/submission.csv \
  -m "Gradient boosting baseline with preprocessing pipeline"
```

## Portfolio Value

- Demonstrates clean ML workflow design, not just notebook experimentation.
- Shows metric-aware modeling choices for leaderboard competitions.
- Proves ability to move from dataset acquisition to validated external submission.
- Gives a strong base for a second project focused on ensembles, explainability, or deployment.

## Strong Next Projects

1. Build a stacked ensemble with out-of-fold meta-features and try to beat `0.12254`.
2. Add SHAP-based explainability and create a valuation insight dashboard.
3. Turn the pipeline into a Streamlit or FastAPI property valuation app.
4. Add experiment tracking and automated hyperparameter search.
