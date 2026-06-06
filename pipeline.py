"""House Prices competition training and submission pipeline.

Competition: House Prices - Advanced Regression Techniques
URL: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

This script is built for portfolio-quality reproducibility:
- optional Kaggle API download
- explicit preprocessing pipeline
- model benchmark with RMSLE-oriented cross-validation
- automatic submission file generation
"""

from __future__ import annotations

import argparse
import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class Config:
    competition_name: str = "house-prices-advanced-regression-techniques"
    data_dir: str = "data"
    output_dir: str = "outputs"
    random_state: int = 42
    cv_folds: int = 5


class HousePriceWorkbench:
    """Training and submission workflow for Ames housing competition."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.logger = self._build_logger()
        self.project_root = Path(__file__).resolve().parent
        self.data_dir = self.project_root / self.config.data_dir
        self.output_dir = self.project_root / self.config.output_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _build_logger() -> logging.Logger:
        logger = logging.getLogger("house_price_workbench")
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        return logger

    def maybe_download_data(self, skip_download: bool) -> None:
        train_path = self.data_dir / "train.csv"
        test_path = self.data_dir / "test.csv"

        if train_path.exists() and test_path.exists():
            self.logger.info("Found local train.csv and test.csv in data directory.")
            return

        if skip_download:
            raise FileNotFoundError(
                "Missing data files. Place train.csv and test.csv in data/, or rerun without --skip-download."
            )

        self.logger.info("Attempting Kaggle API download for %s", self.config.competition_name)
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except Exception as exc:
            raise RuntimeError(
                "Kaggle package is not installed. Install requirements and retry."
            ) from exc

        zip_path = self.data_dir / f"{self.config.competition_name}.zip"
        api = KaggleApi()
        api.authenticate()
        api.competition_download_files(self.config.competition_name, path=str(self.data_dir), quiet=False)

        if not zip_path.exists():
            raise FileNotFoundError(f"Expected Kaggle zip not found at {zip_path}")

        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(self.data_dir)

        self.logger.info("Competition data downloaded and extracted.")

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
        train_df = pd.read_csv(self.data_dir / "train.csv")
        test_df = pd.read_csv(self.data_dir / "test.csv")

        y = np.log1p(train_df["SalePrice"])
        X = train_df.drop(columns=["SalePrice"])

        self.logger.info("Loaded train shape: %s", X.shape)
        self.logger.info("Loaded test shape: %s", test_df.shape)
        return train_df, X, y, test_df

    def build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )

        self.logger.info("Numeric features: %d | Categorical features: %d", len(numeric_cols), len(categorical_cols))

        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, numeric_cols),
                ("cat", categorical_pipe, categorical_cols),
            ],
            remainder="drop",
        )

    def candidate_models(self) -> Dict[str, object]:
        return {
            "ridge": Ridge(alpha=12.0, random_state=self.config.random_state),
            "elastic_net": ElasticNet(alpha=0.0007, l1_ratio=0.85, random_state=self.config.random_state),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=2200,
                learning_rate=0.03,
                max_depth=3,
                max_features="sqrt",
                min_samples_leaf=15,
                min_samples_split=10,
                loss="huber",
                random_state=self.config.random_state,
            ),
            "random_forest": RandomForestRegressor(
                n_estimators=900,
                max_depth=None,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=self.config.random_state,
                n_jobs=-1,
            ),
        }

    def evaluate_and_select_model(self, X: pd.DataFrame, y: pd.Series) -> Pipeline:
        preprocessor = self.build_preprocessor(X)
        kfold = KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)

        best_name = ""
        best_score = np.inf
        best_pipeline: Pipeline | None = None

        for name, model in self.candidate_models().items():
            pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
            neg_rmse = cross_val_score(
                pipe,
                X,
                y,
                scoring="neg_root_mean_squared_error",
                cv=kfold,
                n_jobs=-1,
            )
            rmsle = -neg_rmse.mean()
            self.logger.info("%s CV RMSLE: %.5f", name, rmsle)

            if rmsle < best_score:
                best_score = rmsle
                best_name = name
                best_pipeline = pipe

        assert best_pipeline is not None
        self.logger.info("Selected model: %s | CV RMSLE: %.5f", best_name, best_score)
        return best_pipeline

    def train_and_predict(self, X: pd.DataFrame, y: pd.Series, test_df: pd.DataFrame) -> pd.DataFrame:
        model = self.evaluate_and_select_model(X, y)
        model.fit(X, y)
        log_preds = model.predict(test_df)
        sale_price = np.expm1(log_preds)
        sale_price = np.where(sale_price < 0, 0, sale_price)

        submission = pd.DataFrame({"Id": test_df["Id"], "SalePrice": sale_price})
        return submission

    def save_submission(self, submission: pd.DataFrame) -> Path:
        submission_path = self.output_dir / "submission.csv"
        submission.to_csv(submission_path, index=False)
        self.logger.info("Saved submission file: %s", submission_path)
        return submission_path

    def run(self, skip_download: bool) -> Path:
        self.maybe_download_data(skip_download=skip_download)
        _, X, y, test_df = self.load_data()
        submission = self.train_and_predict(X, y, test_df)
        return self.save_submission(submission)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="House Prices competition pipeline")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip Kaggle API download and use existing local data files.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of CV folds for model selection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(cv_folds=args.cv_folds)
    app = HousePriceWorkbench(config)
    submission_path = app.run(skip_download=args.skip_download)
    print(f"Submission ready: {submission_path}")


if __name__ == "__main__":
    main()
