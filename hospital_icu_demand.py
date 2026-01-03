"""ICU bed demand prediction using RandomForest.

This module trains an ICU demand model using historical ICU occupancy,
engineered temporal features, calendar effects, and environmental /
health trend features.

Design choices
--------------
- Inputs:
  - ICU occupancy history (via engineered features and ICU occupancy pct).
  - Admissions history (lags / rolling means act as a proxy for
    predicted admissions in deployment).
  - Calendar features (weekend, holiday, seasons).
  - Environmental & health trend variables (and their normalized
    versions).
- Model: RandomForestRegressor (non-parametric, robust, moderately
  interpretable via feature importances).
- Temporal split: last N days as test horizon to mimic forward-looking
  forecasting.
- Target: next-day ICU beds occupied, with utilization percentage
  computed relative to capacity.

Note on "predicted admissions"
------------------------------
For simplicity and to avoid leakage, this module does not directly
inject next-day admission forecasts as a feature. Instead, it relies on
admission lags and rolling statistics, which are available at decision
time and closely related to any admissions forecast. If you already
have explicit next-day admission predictions from another model, you
can add them as an extra feature column before training.
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
from sklearn.ensemble import RandomForestRegressor
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


# ----------------------------- Config Object -----------------------------


@dataclass
class ICUDemandConfig:
    """Configuration for ICU demand prediction.

    Parameters
    ----------
    base_dir:
        Directory containing the raw CSVs used to build the master
        dataframe.
    test_horizon_days:
        Number of days at the end of the historical period reserved for
        evaluation (e.g., 7).
    forecast_horizon_days:
        How many days ahead each prediction targets (typically 1 for
        next-day ICU demand).
    feature_config:
        FeatureConfig describing how to compute features and targets
        from the master dataframe. If None, a default will be used and
        should be adapted to real column names.
    rf_params:
        Optional dictionary of RandomForestRegressor hyperparameters
        (e.g., n_estimators, max_depth) to control the model.
    """

    base_dir: Path
    test_horizon_days: int = 7
    forecast_horizon_days: int = 1
    feature_config: Optional[FeatureConfig] = None
    rf_params: Optional[Dict[str, object]] = None


# -------------------------- Data Preparation ----------------------------


def build_master_dataframe(base_dir: Path) -> pd.DataFrame:
    """Build master dataframe using the shared pipeline."""

    master_df, _ = example_build_master_from_default_files(base_dir)
    return master_df


def prepare_icu_features_and_target(
    master_df: pd.DataFrame,
    forecast_horizon_days: int,
    feature_config: Optional[FeatureConfig] = None,
) -> Tuple[pd.DataFrame, pd.Series, FeatureConfig, ColumnScaler]:
    """Engineer features and extract ICU demand target.

    This function reuses the shared `engineer_features` function but
    focuses on the ICU demand target (next-day beds occupied).
    """

    if feature_config is None:
        feature_config = FeatureConfig()

    # Align forecast horizon with ICU objective
    feature_config.forecast_horizon_days = int(forecast_horizon_days)

    X, _y_adm, y_icu, scaler = engineer_features(
        master_df=master_df,
        config=feature_config,
        scaler=None,
        fit_scaler=True,
        target_mode="icu",
    )

    return X, y_icu, feature_config, scaler


# ------------------------------ Modeling --------------------------------


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    rf_params: Optional[Dict[str, object]] = None,
) -> Pipeline:
    """Train a RandomForestRegressor for ICU demand.

    A modest-sized forest is used by default for stability and partial
    interpretability via feature importances.
    """

    if rf_params is None:
        rf_params = {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_leaf": 5,
            "random_state": 42,
            "n_jobs": -1,
        }

    pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True),
            ),
            ("model", RandomForestRegressor(**rf_params)),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_icu_forecast(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute MAE and RMSE for ICU demand forecast."""

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    return {"MAE": float(mae), "RMSE": float(rmse)}


# --------------------------- Visualizations -----------------------------


def plot_feature_importances(
    model: Pipeline,
    feature_names: pd.Index,
    top_n: int = 20,
    title: str = "Random Forest Feature Importances (ICU Demand)",
) -> pd.DataFrame:
    """Plot top-N feature importances and return them as a dataframe."""

    rf = model.named_steps.get("model")
    if rf is None or not hasattr(rf, "feature_importances_"):
        raise ValueError("Expected a fitted RandomForestRegressor under pipeline step 'model'.")
    importances = rf.feature_importances_
    df_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
    df_imp = df_imp.sort_values(by="importance", ascending=False)

    if top_n is not None and top_n > 0:
        df_imp_plot = df_imp.head(top_n)
    else:
        df_imp_plot = df_imp

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df_imp_plot["feature"][::-1], df_imp_plot["importance"][::-1])
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    plt.close(fig)

    return df_imp.reset_index(drop=True)


# ------------------------ High-Level Orchestration ----------------------


def run_icu_demand_pipeline(config: ICUDemandConfig) -> Dict[str, object]:
    """Run end-to-end ICU bed demand prediction pipeline.

    Steps
    -----
    1. Build master dataframe from raw CSVs.
    2. Engineer features and extract ICU demand target.
    3. Temporal train/test split.
    4. Train RandomForestRegressor on ICU demand.
    5. Evaluate MAE & RMSE on held-out horizon.
    6. Plot feature importances.
    7. Compute ICU utilization percentage for the test period and
       next-day prediction.

    Returns
    -------
    results:
        Dictionary containing model artefacts and evaluation outputs:
        - "rf_model": trained RandomForestRegressor
        - "X_train", "X_test", "y_train", "y_test"
        - "metrics": {"MAE": ..., "RMSE": ...}
        - "feature_importances": dataframe sorted by importance
        - "icu_utilization_pct_test": Series of utilization for test set
        - "next_day_icu_beds": float, predicted beds for final test day
        - "next_day_icu_utilization_pct": float, utilization percentage
        - "feature_config": FeatureConfig used
        - "scaler": ColumnScaler used for normalization
    """

    # 1. Build master dataframe
    master_df = build_master_dataframe(config.base_dir)

    # 2. Engineer features and ICU target
    X, y_icu, feature_cfg, scaler = prepare_icu_features_and_target(
        master_df=master_df,
        forecast_horizon_days=config.forecast_horizon_days,
        feature_config=config.feature_config,
    )

    # 3. Temporal train/test split
    X_train, X_test, y_train, y_test = temporal_train_test_split(
        X=X,
        y=y_icu,
        test_horizon_days=config.test_horizon_days,
    )

    # 4. Train RandomForestRegressor
    rf_model = train_random_forest(
        X_train=X_train,
        y_train=y_train,
        rf_params=config.rf_params,
    )

    # 5. Evaluate
    y_pred = rf_model.predict(X_test)
    metrics = evaluate_icu_forecast(y_true=y_test, y_pred=y_pred)

    # 6. Feature importances plot
    feature_importances = plot_feature_importances(
        model=rf_model,
        feature_names=X_train.columns,
    )

    # 7. ICU utilization percentages
    # Use capacity from features if available; otherwise utilization
    # cannot be computed.
    capacity_col = feature_cfg.icu_capacity_col
    icu_utilization_pct_test = None
    next_day_icu_beds = float(y_pred[-1]) if len(y_pred) > 0 else float("nan")
    next_day_icu_utilization_pct = float("nan")

    if capacity_col in X_test.columns:
        capacity_series = pd.to_numeric(X_test[capacity_col], errors="coerce").astype("float64")
        with np.errstate(divide="ignore", invalid="ignore"):
            icu_utilization_pct_test = (
                y_pred / capacity_series.values * 100.0
            )
        icu_utilization_pct_test = pd.Series(
            icu_utilization_pct_test,
            index=X_test.index,
            name="icu_utilization_pct_pred",
        )

        # Next-day utilization based on final test instance
        last_capacity = float(capacity_series.iloc[-1]) if len(capacity_series) else float("nan")
        if not np.isfinite(last_capacity) or last_capacity <= 0:
            # Fallback to the most recent non-missing capacity in train+test
            if capacity_col in X_train.columns:
                all_cap = pd.concat(
                    [
                        pd.to_numeric(X_train[capacity_col], errors="coerce"),
                        pd.to_numeric(X_test[capacity_col], errors="coerce"),
                    ]
                )
                all_cap = all_cap.dropna()
                if not all_cap.empty:
                    last_capacity = float(all_cap.iloc[-1])

        if np.isfinite(last_capacity) and last_capacity > 0:
            next_day_icu_utilization_pct = next_day_icu_beds / last_capacity * 100.0

    results: Dict[str, object] = {
        "rf_model": rf_model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "metrics": metrics,
        "feature_importances": feature_importances,
        "icu_utilization_pct_test": icu_utilization_pct_test,
        "next_day_icu_beds": next_day_icu_beds,
        "next_day_icu_utilization_pct": next_day_icu_utilization_pct,
        "feature_config": feature_cfg,
        "scaler": scaler,
    }

    return results


if __name__ == "__main__":
    # Example usage of the ICU demand pipeline.
    base_dir = Path(".")

    cfg = ICUDemandConfig(
        base_dir=base_dir,
        test_horizon_days=7,
        forecast_horizon_days=1,
        feature_config=None,  # use defaults; customize in production
        rf_params=None,       # override for tuning if needed
    )

    results = run_icu_demand_pipeline(cfg)

    print("ICU demand forecast metrics (next-day on last 7 days):")
    for name, value in results["metrics"].items():
        print(f"  {name}: {value:.3f}")

    print("\nTop ICU feature importances:")
    print(results["feature_importances"].head(10))

    print("\nNext-day predicted ICU beds:", results["next_day_icu_beds"])
    print(
        "Next-day predicted ICU utilization (%):",
        f"{results['next_day_icu_utilization_pct']:.2f}",
    )
