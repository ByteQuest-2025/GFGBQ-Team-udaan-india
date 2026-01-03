"""Pipeline controller for hospital forecasting.

This module defines a single "run" entrypoint used by interactive apps
(Streamlit) and services.

Design goals
- Button-triggered execution (no work at import time)
- Defensive NaN handling
- Clear, JSON-friendly output structure
- Small, production-style surface area
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd

from alerts import AlertThresholds, generate_alerts
from hospital_data_pipeline import example_build_master_from_default_files
from hospital_decision_engine import AlertEngineConfig, generate_alert
from hospital_feature_engineering import FeatureConfig
from hospital_forecasting import ForecastingConfig, run_admissions_forecasting_pipeline
from hospital_icu_demand import ICUDemandConfig, run_icu_demand_pipeline
from hospital_staff_risk import StaffRiskConfig, run_staff_risk_pipeline
from main import build_shared_feature_config, infer_context_flags


class DataQualityError(RuntimeError):
    """Raised when input data fails quality checks for reliable forecasts."""


@dataclass(frozen=True)
class PipelineConfig:
    base_dir: Path = Path(".")
    test_horizon_days: int = 7
    forecast_horizon_days: int = 1
    alert_thresholds: AlertThresholds = AlertThresholds()


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
    except Exception:
        return float(default)
    return f if np.isfinite(f) else float(default)


def _finite_int(value: Any) -> Optional[int]:
    try:
        f = float(value)
    except Exception:
        return None
    if not np.isfinite(f):
        return None
    return int(round(f))


def _last_numeric(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().any():
        return float(s.dropna().iloc[-1])
    return float("nan")


def _to_builtin(obj: Any) -> Any:
    """Convert numpy/pandas objects into JSON-friendly Python builtins."""

    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()

    if isinstance(obj, (Path,)):
        return str(obj)

    if isinstance(obj, (list, tuple)):
        return [_to_builtin(x) for x in obj]

    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}

    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        try:
            return _to_builtin(obj.to_dict())
        except Exception:
            pass

    return str(obj)


def _validate_master_schema(master_df: pd.DataFrame, feature_cfg: FeatureConfig) -> None:
    required_cols = [
        feature_cfg.date_col,
        feature_cfg.admissions_col,
        feature_cfg.icu_occupied_col,
        feature_cfg.icu_capacity_col,
    ]
    missing = [c for c in required_cols if c not in master_df.columns]
    if missing:
        raise ValueError(f"Master dataframe is missing required columns: {missing}")

    numeric_cols = [
        feature_cfg.admissions_col,
        feature_cfg.icu_occupied_col,
        feature_cfg.icu_capacity_col,
    ]
    non_numeric = []
    for col in numeric_cols:
        try:
            pd.to_numeric(master_df[col], errors="raise")
        except Exception:
            non_numeric.append(col)
    if non_numeric:
        raise ValueError(
            "Master dataframe has non-numeric values in required numeric columns: "
            f"{non_numeric}. Please check upstream CSV schemas."
        )


def _check_data_quality(master_df: pd.DataFrame, feature_cfg: FeatureConfig) -> None:
    key_cols = [
        feature_cfg.admissions_col,
        feature_cfg.icu_occupied_col,
        feature_cfg.icu_capacity_col,
    ]

    warn_threshold = 0.3
    fail_threshold = 0.98

    hard_fail_reasons = []
    for col in key_cols:
        if col not in master_df.columns:
            continue
        s = master_df[col]
        frac_missing = float(s.isna().mean())
        non_zero = int((s.fillna(0) != 0).sum())

        if frac_missing >= fail_threshold or non_zero == 0:
            hard_fail_reasons.append(f"{col}: {frac_missing:.0%} missing, non_zero={non_zero}")

        # Note: warn_threshold is intentionally not surfaced here; callers
        # can log warnings from monitoring/metrics if desired.

    if hard_fail_reasons:
        joined = "; ".join(hard_fail_reasons)
        raise DataQualityError(
            "Data quality check failed for critical inputs; "
            f"series is almost entirely missing or empty. Details: {joined}"
        )


def _build_labels_from_index(index: Any, n: int) -> list[str]:
    try:
        if index is not None and hasattr(index, "to_list") and len(index) == n:
            labels: list[str] = []
            for v in index.to_list():
                if isinstance(v, pd.Timestamp):
                    labels.append(v.strftime("%a"))
                else:
                    labels.append(str(v))
            return labels
    except Exception:
        pass
    return [f"D{i}" for i in range(1, n + 1)]


def run_pipeline(config: PipelineConfig | Mapping[str, Any]) -> Dict[str, Any]:
    """Run the full hospital forecasting pipeline.

    Parameters
    ----------
    config:
        Either a `PipelineConfig` or a dict-like object containing:
        - base_dir (str|Path)
        - test_horizon_days (int)
        - forecast_horizon_days (int)

    Returns
    -------
    Dict[str, Any]
        JSON-friendly structure with a stable top-level summary:
        - predicted_admissions
        - predicted_icu_beds
        - icu_utilization
        - alert_level

        And an additional `details` object with richer artefacts.
    """

    if isinstance(config, PipelineConfig):
        cfg = config
    else:
        base_dir_raw = config.get("base_dir", ".")
        cfg = PipelineConfig(
            base_dir=Path(base_dir_raw).resolve() if base_dir_raw is not None else Path(".").resolve(),
            test_horizon_days=int(config.get("test_horizon_days", config.get("eval_days", 7))),
            forecast_horizon_days=int(config.get("forecast_horizon_days", config.get("forecast_days", 1))),
        )

    base_dir = cfg.base_dir
    test_horizon = int(cfg.test_horizon_days)
    forecast_horizon = int(cfg.forecast_horizon_days)

    # 1) Load + build master dataframe
    master_df, dataset_summaries = example_build_master_from_default_files(base_dir)
    feature_cfg: FeatureConfig = build_shared_feature_config()

    # Fail fast if the master dataframe does not meet expectations.
    _validate_master_schema(master_df, feature_cfg)
    _check_data_quality(master_df, feature_cfg)

    # 2) Admissions forecasting
    adm_cfg = ForecastingConfig(
        base_dir=base_dir,
        test_horizon_days=test_horizon,
        forecast_horizon_days=forecast_horizon,
        feature_config=feature_cfg,
    )
    adm_results = run_admissions_forecasting_pipeline(adm_cfg)

    predicted_admissions: float = float("nan")
    admissions_series: Dict[str, Any] = {"labels": [], "actual": [], "predicted": []}
    admissions_model_type: Optional[str] = None
    try:
        y_test = adm_results["y_test"]
        X_test = adm_results["X_test"]
        model = adm_results["model"]
        admissions_model_type = str(type(model).__name__)
        y_pred = model.predict(X_test)
        if len(y_pred) > 0 and np.isfinite(float(y_pred[-1])):
            predicted_admissions = float(y_pred[-1])

        labels = _build_labels_from_index(getattr(y_test, "index", None), n=len(y_test))
        admissions_series = {
            "labels": labels,
            "actual": [float(x) for x in getattr(y_test, "values", [])],
            "predicted": [float(x) for x in list(y_pred)],
        }
    except Exception:
        pass

    # Fallback to last observed admissions
    if not np.isfinite(predicted_admissions):
        admissions_col = feature_cfg.admissions_col
        if admissions_col in master_df.columns:
            sorted_master = (
                master_df.sort_values(feature_cfg.date_col)
                if feature_cfg.date_col in master_df.columns
                else master_df
            )
            predicted_admissions = _last_numeric(sorted_master[admissions_col])

    # 3) ICU demand pipeline
    icu_cfg = ICUDemandConfig(
        base_dir=base_dir,
        test_horizon_days=test_horizon,
        forecast_horizon_days=forecast_horizon,
        feature_config=feature_cfg,
        rf_params=None,
    )
    icu_results = run_icu_demand_pipeline(icu_cfg)

    predicted_icu_beds = _finite_float(icu_results.get("next_day_icu_beds"), default=float("nan"))

    icu_model_type: Optional[str] = None
    try:
        if "model" in icu_results:
            icu_model_type = str(type(icu_results.get("model")).__name__)
    except Exception:
        icu_model_type = None

    # ICU current occupied (last known occupied beds)
    current_icu_occupied = float("nan")
    if feature_cfg.icu_occupied_col in master_df.columns:
        sorted_master = (
            master_df.sort_values(feature_cfg.date_col)
            if feature_cfg.date_col in master_df.columns
            else master_df
        )
        current_icu_occupied = _last_numeric(sorted_master[feature_cfg.icu_occupied_col])

    # ICU capacity: last known capacity (or 0)
    icu_capacity_next_day = 0.0
    if feature_cfg.icu_capacity_col in master_df.columns:
        sorted_master = (
            master_df.sort_values(feature_cfg.date_col)
            if feature_cfg.date_col in master_df.columns
            else master_df
        )
        cap_val = _last_numeric(sorted_master[feature_cfg.icu_capacity_col])
        if np.isfinite(cap_val):
            icu_capacity_next_day = float(cap_val)

    # Safe utilization percent
    if icu_capacity_next_day <= 0 or not np.isfinite(predicted_icu_beds):
        icu_utilization_pct = 0.0
    else:
        icu_utilization_pct = round((predicted_icu_beds / icu_capacity_next_day) * 100.0, 1)
        if not np.isfinite(icu_utilization_pct):
            icu_utilization_pct = 0.0

    # 4) Staff risk (optional; failure should not block pipeline)
    staff_risk_level = "UNKNOWN"
    staff_results: Optional[Dict[str, Any]] = None
    staff_error: Optional[str] = None
    try:
        staff_cfg = StaffRiskConfig(
            base_dir=base_dir,
            forecast_horizon_days=forecast_horizon,
            test_horizon_days=test_horizon,
            feature_config=feature_cfg,
        )
        staff_results = run_staff_risk_pipeline(staff_cfg)
        staff_risk_level = str(staff_results.get("next_day_risk_level", "UNKNOWN"))
    except Exception as exc:
        staff_error = str(exc)
        staff_results = None
        staff_risk_level = "UNKNOWN"

    # 5) Alerting (simple deterministic ICU utilization level)
    alert_level = generate_alerts(icu_utilization_pct, thresholds=cfg.alert_thresholds)

    # (Optional) richer decision-engine alert payload for explainability.
    context_flags = infer_context_flags(master_df, feature_cfg)
    decision_engine_alert = generate_alert(
        predicted_admissions=_finite_float(predicted_admissions, default=float("nan")),
        predicted_icu_demand=_finite_float(predicted_icu_beds, default=float("nan")),
        icu_capacity=float(icu_capacity_next_day),
        staff_risk_level=staff_risk_level,  # type: ignore[arg-type]
        bed_availability=_finite_float(context_flags.get("bed_availability"), default=0.0),
        high_respiratory_trend=bool(context_flags.get("high_respiratory_trend")),
        config=AlertEngineConfig(),
        include_timestamp=True,
        is_weekend=context_flags.get("is_weekend"),
        is_low_temperature=context_flags.get("is_low_temperature"),
        reduced_staff_availability=(staff_risk_level == "HIGH"),
    )

    # 6) Clear output structure (summary + details)
    date_range: Dict[str, Any] = {"start": None, "end": None}
    try:
        if feature_cfg.date_col in master_df.columns:
            date_range = {
                "start": _to_builtin(pd.to_datetime(master_df[feature_cfg.date_col], errors="coerce").min()),
                "end": _to_builtin(pd.to_datetime(master_df[feature_cfg.date_col], errors="coerce").max()),
            }
    except Exception:
        date_range = {"start": None, "end": None}

    result: Dict[str, Any] = {
        "predicted_admissions": _finite_int(predicted_admissions),
        "predicted_icu_beds": _finite_int(predicted_icu_beds),
        "icu_utilization": float(icu_utilization_pct),
        "alert_level": alert_level,
        "details": {
            "inputs": {
                "base_dir": str(base_dir),
                "test_horizon_days": test_horizon,
                "forecast_horizon_days": forecast_horizon,
                "datasets": dataset_summaries,
            },
            "data_manifest": {
                "base_dir": str(base_dir),
                "num_rows": int(master_df.shape[0]),
                "num_columns": int(master_df.shape[1]),
                "date_range": date_range,
            },
            "metrics": {
                "admissions": _to_builtin(adm_results.get("metrics")),
                "icu": _to_builtin(icu_results.get("metrics")),
                "staff": _to_builtin(staff_results.get("metrics")) if isinstance(staff_results, dict) else None,
            },
            "admissions": {
                "series": _to_builtin(admissions_series),
                "model_type": admissions_model_type,
            },
            "icu": {
                "feature_importances": _to_builtin(icu_results.get("feature_importances", [])),
                "model_type": icu_model_type,
                "current_occupied": _finite_float(current_icu_occupied, default=float("nan")),
            },
            "staff": {
                "risk_level": staff_risk_level,
                "error": staff_error,
                "results": _to_builtin(staff_results) if staff_results is not None else None,
            },
            "capacity": {
                "icu_capacity_next_day": float(icu_capacity_next_day),
            },
            "context": _to_builtin(context_flags),
            "decision_engine_alert": _to_builtin(decision_engine_alert),
        },
    }

    return result
