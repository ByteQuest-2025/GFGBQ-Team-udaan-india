"""Emergency admissions forecasting for hospital load planning.

This module builds a forecasting model for emergency admissions using a
regression-based time series approach over engineered features.

Goals
-----
- Interpretability: use a linear model with human-understandable
  features (lags, rolling means, seasonal flags, calendar flags,
  environmental and health trend variables).
- Reliability: avoid data leakage by using only past information to
  predict future demand.
- Practicality: simple train/validation split, clear evaluation
  metrics (MAE, RMSE), and visual diagnostics.

Dependencies
------------
- pandas
- numpy
- scikit-learn
- matplotlib

These can be installed via pip if not already present, e.g.:
    pip install pandas numpy scikit-learn matplotlib

This module assumes that `hospital_data_pipeline` and
`hospital_feature_engineering` are available in the same project.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib

# Use a non-interactive backend so this module can run in servers/threads.
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from hospital_data_pipeline import example_build_master_from_default_files
from hospital_feature_engineering import (
    ColumnScaler,
    FeatureConfig,
    engineer_features,
)


# ----------------------------- Config Wrapper ----------------------------


@dataclass
class ForecastingConfig:
    """Configuration for admissions forecasting.

    Parameters
    ----------
    base_dir:
        Directory containing the raw CSVs used to build the master
        dataframe.
    test_horizon_days:
        Number of days at the end of the historical period reserved for
        evaluation (e.g., 7).
    forecast_horizon_days:
        How many days ahead each prediction targets. For next-day
        forecasting, use 1. For simplicity and interpretability, this
        implementation uses 1-day-ahead forecasting and evaluates on
        the final `test_horizon_days` days.
    feature_config:
        FeatureConfig describing how to compute features and targets
        from the master dataframe.
    """

    base_dir: Path
    test_horizon_days: int = 7
    forecast_horizon_days: int = 1
    feature_config: Optional[FeatureConfig] = None


# ----------------------------- Core Pipeline -----------------------------


def build_master_dataframe(base_dir: Path) -> pd.DataFrame:
    """Build master dataframe using shared pipeline.

    This wraps `example_build_master_from_default_files` to make it
    explicit in this forecasting module.
    """

    master_df, _ = example_build_master_from_default_files(base_dir)
    return master_df


def prepare_features_and_targets(
    master_df: pd.DataFrame,
    forecast_horizon_days: int,
    feature_config: Optional[FeatureConfig] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, FeatureConfig, ColumnScaler]:
    """Prepare X, y_admissions, and y_icu_demand for forecasting.

    This function delegates feature engineering to the shared
    `engineer_features` function, ensuring consistent transformations
    across training and evaluation.
    """

    if feature_config is None:
        # Provide a conservative default; these names should be updated
        # to match the actual master_df columns in your environment.
        feature_config = FeatureConfig()

    # Ensure the forecast horizon in the feature config matches our
    # forecasting objective.
    feature_config.forecast_horizon_days = int(forecast_horizon_days)

    X, y_adm, y_icu, scaler = engineer_features(
        master_df=master_df,
        config=feature_config,
        scaler=None,
        fit_scaler=True,
        target_mode="admissions",
    )

    return X, y_adm, y_icu, feature_config, scaler


def temporal_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_horizon_days: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform a simple temporal train/test split.

    - Training set: all observations except the final `test_horizon_days`.
    - Test set: final `test_horizon_days` observations.
    """

    if test_horizon_days <= 0 or test_horizon_days >= len(X):
        raise ValueError(
            "test_horizon_days must be > 0 and smaller than the number of observations."
        )

    split_idx = len(X) - int(test_horizon_days)

    X_train = X.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx].copy()
    y_test = y.iloc[split_idx:].copy()

    return X_train, X_test, y_train, y_test


def train_linear_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Train a simple linear regression model.

    Linear regression is chosen for interpretability: coefficients can
    be directly inspected to understand how each feature influences the
    forecast.
    """

    pipeline = Pipeline(
        steps=[
            # Constant imputation keeps all columns (median would drop all-NaN columns).
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True),
            ),
            ("model", LinearRegression()),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_forecast(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute MAE and RMSE for forecast evaluation."""

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    return {"MAE": float(mae), "RMSE": float(rmse)}


def plot_actual_vs_predicted(
    y_true: pd.Series,
    y_pred: np.ndarray,
    title: str = "Actual vs Predicted Emergency Admissions",
) -> None:
    """Plot actual vs predicted values over time.

    This function assumes that y_true has a meaningful index (e.g.,
    dates). It keeps the visualization simple and interpretable.
    """

    fig, ax = plt.subplots(figsize=(10, 5))

    # Use the same index for both series
    x_axis = y_true.index

    ax.plot(x_axis, y_true.values, label="Actual", marker="o")
    ax.plot(x_axis, y_pred, label="Predicted", marker="x")

    ax.set_xlabel("Time")
    ax.set_ylabel("Emergency Admissions")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    plt.close(fig)


def summarize_linear_model(
    model: Pipeline,
    feature_names: pd.Index,
    top_n: int = 20,
) -> pd.DataFrame:
    """Return a dataframe summarizing linear model coefficients.

    This makes it easier to inspect which features drive the forecast,
    supporting interpretability and clinical review.
    """

    # Pipeline stores the regression model under the "model" step
    lr = model.named_steps.get("model")
    if lr is None or not hasattr(lr, "coef_"):
        raise ValueError("Expected a fitted LinearRegression under pipeline step 'model'.")
    coefs = lr.coef_.ravel()
    summary_df = pd.DataFrame(
        {"feature": feature_names, "coefficient": coefs}
    ).sort_values(by="coefficient", key=lambda s: s.abs(), ascending=False)

    if top_n is not None and top_n > 0:
        summary_df = summary_df.head(top_n)

    return summary_df.reset_index(drop=True)


# ------------------------ High-Level Orchestration ----------------------


def run_admissions_forecasting_pipeline(
    config: ForecastingConfig,
) -> Dict[str, object]:
    """Run end-to-end pipeline for emergency admissions forecasting.

    Steps
    -----
    1. Build master dataframe from raw CSVs.
    2. Engineer features and targets (admissions, ICU demand).
    3. Perform temporal train/test split.
    4. Train a linear regression model on historical admissions.
    5. Evaluate MAE and RMSE on the held-out horizon.
    6. Plot actual vs predicted admissions for the test period.
    7. Summarize feature importances via linear coefficients.

    Returns
    -------
    results:
        Dictionary containing trained objects, metrics, and artefacts:
        - "model": trained LinearRegression model
        - "X_train", "X_test", "y_train", "y_test"
        - "metrics": {"MAE": ..., "RMSE": ...}
        - "coef_summary": dataframe of top features by coefficient
        - "feature_config": FeatureConfig used
        - "scaler": ColumnScaler for normalization
    """

    # 1. Build master dataframe
    master_df = build_master_dataframe(config.base_dir)

    # 2. Engineer features and targets
    X, y_adm, y_icu, feature_cfg, scaler = prepare_features_and_targets(
        master_df=master_df,
        forecast_horizon_days=config.forecast_horizon_days,
        feature_config=config.feature_config,
    )

    # 3. Temporal train/test split (use admissions as the primary target)
    X_train, X_test, y_train, y_test = temporal_train_test_split(
        X=X,
        y=y_adm,
        test_horizon_days=config.test_horizon_days,
    )

    # 4. Train linear regression model
    model = train_linear_model(X_train, y_train)

    # 5. Evaluate on the held-out test set
    y_pred = model.predict(X_test)
    metrics = evaluate_forecast(y_true=y_test, y_pred=y_pred)

    # 6. Plot actual vs predicted for visual inspection
    plot_actual_vs_predicted(y_true=y_test, y_pred=y_pred)

    # 7. Summarize coefficients for interpretability
    coef_summary = summarize_linear_model(model, feature_names=X_train.columns)

    results: Dict[str, object] = {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "metrics": metrics,
        "coef_summary": coef_summary,
        "feature_config": feature_cfg,
        "scaler": scaler,
    }

    return results


if __name__ == "__main__":
    # Example usage of the forecasting pipeline.
    base_dir = Path(".")

    # Configure forecasting
    fc = ForecastingConfig(
        base_dir=base_dir,
        test_horizon_days=7,
        forecast_horizon_days=1,
        feature_config=None,  # use defaults; adjust in real deployment
    )

    results = run_admissions_forecasting_pipeline(fc)

    print("Evaluation metrics (next-day forecast on last 7 days):")
    for name, value in results["metrics"].items():
        print(f"  {name}: {value:.3f}")

    print("\nTop linear model coefficients (by absolute magnitude):")
    print(results["coef_summary"])