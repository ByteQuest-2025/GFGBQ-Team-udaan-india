"""Staff workload and burnout risk prediction.

This module operates on the same hospital data pipeline used for
admissions and ICU forecasting. It focuses on staff-related workload
metrics and simple, interpretable risk stratification.

Objectives
----------
- Compute workload per staff member based on staff availability and
  patient load.
- Classify burnout / staffing stress risk into LOW / MEDIUM / HIGH
  using transparent thresholds derived from historical data.
- Predict next-day staffing stress level with a simple regression
  model (linear) for interpretability.
- Provide human-readable staffing recommendations that clinical and
  operations leaders can review.

Assumptions
-----------
- Master dataframe is built using `hospital_data_pipeline`.
- `hospital_feature_engineering` is configured so that
  `staff_to_patient_ratio` is available as a feature.
- Higher workload per staff (more patients per staff member) implies
  higher burnout/stress risk.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from hospital_data_pipeline import example_build_master_from_default_files
from hospital_feature_engineering import (
    ColumnScaler,
    FeatureConfig,
    engineer_features,
)
from hospital_forecasting import temporal_train_test_split


# ------------------------------ Config ----------------------------------


@dataclass
class StaffRiskConfig:
    """Configuration for staff workload and burnout risk modeling.

    Parameters
    ----------
    base_dir:
        Directory containing the raw CSVs used to build the master
        dataframe.
    forecast_horizon_days:
        How many days ahead to predict staffing stress (typically 1
        for next-day planning).
    test_horizon_days:
        Number of days at the end of the historical period reserved
        for evaluation (e.g., 7).
    feature_config:
        FeatureConfig describing how to compute features and targets
        from the master dataframe. It must be configured so that
        `staff_to_patient_ratio` is computed (i.e., staff_count_cols
        and patient_count_col are set appropriately).
    low_quantile:
        Quantile of workload per staff below which risk is considered
        LOW.
    high_quantile:
        Quantile of workload per staff above which risk is considered
        HIGH. Values between low and high are MEDIUM.
    """

    base_dir: Path
    forecast_horizon_days: int = 1
    test_horizon_days: int = 7
    feature_config: Optional[FeatureConfig] = None
    low_quantile: float = 0.33
    high_quantile: float = 0.66


# ----------------------- Core Derived Quantities ------------------------


def build_master_dataframe(base_dir: Path) -> pd.DataFrame:
    """Build master dataframe from CSVs using the shared pipeline."""

    master_df, _ = example_build_master_from_default_files(base_dir)
    return master_df


def compute_workload_features(
    master_df: pd.DataFrame,
    forecast_horizon_days: int,
    feature_config: Optional[FeatureConfig] = None,
) -> Tuple[pd.DataFrame, pd.Series, FeatureConfig, ColumnScaler]:
    """Engineer features and compute workload per staff.

    Workload per staff is defined as:
        patients_per_staff = patients / staff_total
    which is the inverse of `staff_to_patient_ratio` (staff / patients).

    Returns
    -------
    X:
        Feature matrix from `engineer_features` (numeric columns).
    workload_per_staff:
        Series aligned with X.index containing patients per staff.
    cfg:
        FeatureConfig actually used.
    scaler:
        ColumnScaler with fitted normalization parameters.
    """

    if feature_config is None:
        feature_config = FeatureConfig()

    # Align forecast horizon with staffing stress objective
    feature_config.forecast_horizon_days = int(forecast_horizon_days)

    X, _y_adm, _y_icu, scaler = engineer_features(
        master_df=master_df,
        config=feature_config,
        scaler=None,
        fit_scaler=True,
        target_mode="none",
    )

    if "staff_to_patient_ratio" not in X.columns:
        raise ValueError(
            "Feature 'staff_to_patient_ratio' not found in engineered features. "
            "Ensure FeatureConfig.staff_count_cols and FeatureConfig.patient_count_col "
            "are correctly set so that staff-to-patient ratio can be computed."
        )

    ratio = X["staff_to_patient_ratio"].astype("float64")
    # Avoid division by zero or extremely small values
    ratio_safe = ratio.replace({0.0: np.nan})
    workload_per_staff = 1.0 / ratio_safe

    return X, workload_per_staff, feature_config, scaler


# -------------------------- Risk Stratification -------------------------


def derive_risk_thresholds(
    workload_series: pd.Series,
    train_index: pd.Index,
    low_quantile: float,
    high_quantile: float,
) -> Dict[str, float]:
    """Derive LOW / MEDIUM / HIGH thresholds from training data.

    Thresholds are based on quantiles of workload per staff within the
    training period only to avoid peeking into the future.
    """

    train_values = workload_series.loc[train_index].dropna()
    if train_values.empty:
        raise ValueError("No valid workload values found in training period.")

    low_thr = float(train_values.quantile(low_quantile))
    high_thr = float(train_values.quantile(high_quantile))

    if low_thr > high_thr:
        # In rare pathological cases, quantiles may invert due to
        # extremely skewed data. Fall back to sorted unique values.
        sorted_vals = np.sort(train_values.unique())
        if len(sorted_vals) >= 3:
            low_thr = float(sorted_vals[len(sorted_vals) // 3])
            high_thr = float(sorted_vals[2 * len(sorted_vals) // 3])

    return {"low": low_thr, "high": high_thr}


def classify_risk(workload_value: float, thresholds: Dict[str, float]) -> str:
    """Classify workload into LOW / MEDIUM / HIGH risk levels."""

    if np.isnan(workload_value):
        return "UNKNOWN"

    low_thr = thresholds["low"]
    high_thr = thresholds["high"]

    if workload_value <= low_thr:
        return "LOW"
    if workload_value >= high_thr:
        return "HIGH"
    return "MEDIUM"


def recommend_staffing(risk_level: str) -> str:
    """Return a human-readable staffing recommendation for a risk level."""

    if risk_level == "LOW":
        return (
            "Workload is within comfortable range. Maintain baseline staffing, "
            "and consider allowing planned leave or training days."
        )
    if risk_level == "MEDIUM":
        return (
            "Workload is moderate. Maintain baseline staffing, limit non-essential "
            "leave, and ensure float/backup staff are available."
        )
    if risk_level == "HIGH":
        return (
            "High workload expected. Increase staffing for the next shift, postpone "
            "non-urgent activities, and consider deploying cross-cover or agency staff."
        )
    return (
        "Risk level is unknown. Review data quality and staffing plans manually "
        "before making scheduling decisions."
    )


# ------------------------- Modeling Next-Day Stress ----------------------


def prepare_stress_targets(
    X: pd.DataFrame,
    workload_per_staff: pd.Series,
    forecast_horizon_days: int,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Prepare features and targets for next-day stress prediction.

    For a horizon of H days (typically 1), the target is
        workload_per_staff(t + H)
    and features are taken from time t.

    Returns
    -------
    X_all:
        Feature matrix aligned to target for training/evaluation
        (last H rows dropped due to unknown targets).
    y_all:
        Target workload per staff at t + H.
    X_future:
        Features for the last available day used to predict the next
        unseen day in production.
    """

    H = int(forecast_horizon_days)
    if H <= 0:
        raise ValueError("forecast_horizon_days must be positive.")

    # Targets shifted forward by H days
    y_all = workload_per_staff.shift(-H)

    # Drop last H rows from X since their targets are NaN
    X_all = X.iloc[:-H].copy()
    y_all = y_all.iloc[:-H]

    # Features for the most recent day (used for next-day prediction)
    X_future = X.iloc[[-1]].copy()

    # Align indices
    valid_mask = y_all.notna()
    X_all = X_all.loc[valid_mask]
    y_all = y_all.loc[valid_mask]

    return X_all, y_all, X_future


def train_stress_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Train an interpretable linear model for staffing stress.

    Linear regression is used so that coefficients can be inspected to
    understand which features drive workload per staff.
    """

    pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True),
            ),
            ("model", LinearRegression()),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_stress_model(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute MAE and RMSE for staffing stress forecast."""

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"MAE": float(mae), "RMSE": float(rmse)}


# ------------------------ High-Level Orchestration ----------------------


def run_staff_risk_pipeline(config: StaffRiskConfig) -> Dict[str, object]:
    """Run end-to-end staff workload and burnout risk assessment.

    Steps
    -----
    1. Build master dataframe from raw CSVs.
    2. Engineer features and derive workload per staff.
    3. Prepare next-day workload targets and perform temporal
       train/test split.
    4. Train a simple linear regression stress model.
    5. Evaluate MAE & RMSE on held-out days.
    6. Derive LOW/MEDIUM/HIGH thresholds from training data.
    7. Classify current and next-day risk levels and generate staffing
       recommendations.

    Returns
    -------
    results:
        Dictionary with:
        - "stress_model": trained LinearRegression
        - "X_train", "X_test", "y_train", "y_test"
        - "metrics": {"MAE": ..., "RMSE": ...}
        - "thresholds": {"low": ..., "high": ...}
        - "current_workload_per_staff": float
        - "current_risk_level": str
        - "next_day_pred_workload_per_staff": float
        - "next_day_risk_level": str
        - "next_day_recommendation": str
        - "feature_config": FeatureConfig used
        - "scaler": ColumnScaler used in feature engineering
    """

    # 1. Build master dataframe
    master_df = build_master_dataframe(config.base_dir)

    # 2. Engineer features and workload per staff
    X, workload_series, feature_cfg, scaler = compute_workload_features(
        master_df=master_df,
        forecast_horizon_days=config.forecast_horizon_days,
        feature_config=config.feature_config,
    )

    # 3. Prepare targets and temporal split
    X_all, y_all, X_future = prepare_stress_targets(
        X=X,
        workload_per_staff=workload_series,
        forecast_horizon_days=config.forecast_horizon_days,
    )

    X_train, X_test, y_train, y_test = temporal_train_test_split(
        X=X_all,
        y=y_all,
        test_horizon_days=config.test_horizon_days,
    )

    # 4. Train linear regression model
    stress_model = train_stress_model(X_train=X_train, y_train=y_train)

    # 5. Evaluate on test horizon
    y_pred_test = stress_model.predict(X_test)
    metrics = evaluate_stress_model(y_true=y_test, y_pred=y_pred_test)

    # 6. Derive thresholds from training data
    thresholds = derive_risk_thresholds(
        workload_series=workload_series,
        train_index=X_train.index,
        low_quantile=config.low_quantile,
        high_quantile=config.high_quantile,
    )

    # Current day workload and risk (last observed day)
    current_workload = float(workload_series.iloc[-1])
    current_risk = classify_risk(current_workload, thresholds)

    # 7. Predict next-day workload and classify risk
    next_day_pred_workload = float(stress_model.predict(X_future)[0])
    next_day_risk = classify_risk(next_day_pred_workload, thresholds)
    next_day_reco = recommend_staffing(next_day_risk)

    results: Dict[str, object] = {
        "stress_model": stress_model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "metrics": metrics,
        "thresholds": thresholds,
        "current_workload_per_staff": current_workload,
        "current_risk_level": current_risk,
        "next_day_pred_workload_per_staff": next_day_pred_workload,
        "next_day_risk_level": next_day_risk,
        "next_day_recommendation": next_day_reco,
        "feature_config": feature_cfg,
        "scaler": scaler,
    }

    return results


if __name__ == "__main__":
    # Example usage of the staff risk pipeline.
    base_dir = Path(".")

    cfg = StaffRiskConfig(
        base_dir=base_dir,
        forecast_horizon_days=1,
        test_horizon_days=7,
        feature_config=None,  # configure staff_count_cols/patient_count_col in real use
        low_quantile=0.33,
        high_quantile=0.66,
    )

    results = run_staff_risk_pipeline(cfg)

    print("Staff stress model metrics (next-day on last 7 days):")
    for name, value in results["metrics"].items():
        print(f"  {name}: {value:.3f}")

    print("\nCurrent workload per staff:", results["current_workload_per_staff"])
    print("Current risk level:", results["current_risk_level"])

    print("\nNext-day predicted workload per staff:", results["next_day_pred_workload_per_staff"])
    print("Next-day risk level:", results["next_day_risk_level"])
    print("Staffing recommendation:")
    print(results["next_day_recommendation"])
