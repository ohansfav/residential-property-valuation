# Residential Property Valuation

A practical competition repository for the Kaggle **House Prices - Advanced Regression Techniques** challenge.

This project is built as portfolio-grade engineering work: reproducible, cleanly structured, and thesis-ready.

## Why This Repo Works For Your Portfolio

- Real competition grounding, not toy datasets.
- Full modeling pipeline with feature preprocessing and model benchmarking.
- Explicit competition metric handling (RMSLE-oriented training via log-target modeling).
- Submission artifact generation for direct Kaggle upload.

## Project Structure

```text
residential-property-valuation/
├── pipeline.py
├── requirements.txt
├── .gitignore
├── data/
│   ├── train.csv
│   └── test.csv
└── outputs/
    └── submission.csv
```

## Competition Link

- [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

## Setup

```bash
pip install -r requirements.txt
```

If you have Kaggle API credentials configured, run:

```bash
python pipeline.py
```

If `train.csv` and `test.csv` are already in `data/`, run:

```bash
python pipeline.py --skip-download
```

## Modeling Approach

1. Target transformation:

- Train on `log1p(SalePrice)` to align with RMSLE-style objectives.

1. Feature handling:

- Numeric features: median imputation + standardization.
- Categorical features: mode imputation + one-hot encoding.

1. Model search:

- Ridge regression
- Elastic Net
- Gradient Boosting Regressor
- Random Forest Regressor

1. Selection:

- K-fold CV using root RMSE on log target (equivalent to RMSLE scale).

1. Submission generation:

- Predictions transformed back with `expm1`.
- Output file: `outputs/submission.csv`.

## Kaggle Submission Command

```bash
kaggle competitions submit \
  -c house-prices-advanced-regression-techniques \
  -f outputs/submission.csv \
  -m "Baseline blended preprocessing + model sweep"
```

## Thesis Angles You Can Build From This

1. **Structured feature-learning thesis**

- Compare manual feature engineering vs learned embeddings for mixed tabular features.
- Evaluate interpretability trade-offs with SHAP and partial dependence.

1. **Generalization and shift thesis**

- Simulate covariate drift across neighborhoods/year-built segments.
- Quantify calibration and ranking stability under shift.

1. **AutoML reliability thesis**

- Benchmark handcrafted pipelines against AutoML frameworks.
- Study reproducibility, variance, and computational cost.

1. **Uncertainty-aware valuation thesis**

- Move from point predictions to prediction intervals.
- Evaluate interval coverage versus practical decision utility.

## Next Upgrade Path

- Add stacking/ensembling with out-of-fold meta-features.
- Add feature importance and model explainability reports.
- Add experiment tracking and model registry artifacts.
- Add CI checks and data validation contracts.
