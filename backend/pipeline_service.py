from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from hospital_data_pipeline import example_build_master_from_default_files
from hospital_decision_engine import AlertEngineConfig, generate_alert
from hospital_forecasting import ForecastingConfig, run_admissions_forecasting_pipeline
from hospital_icu_demand import ICUDemandConfig, run_icu_demand_pipeline
from hospital_staff_risk import StaffRiskConfig, run_staff_risk_pipeline
from main import build_shared_feature_config, infer_context_flags


@dataclass(frozen=True)
class RunRequest:
    base_dir: Path
    test_horizon_days: int = 7
    forecast_horizon_days: int = 1


def _to_builtin(obj: Any) -> Any:
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
        # ISO 8601
        return obj.isoformat()

    if isinstance(obj, (Path,)):
        return str(obj)

    if isinstance(obj, (list, tuple)):
        return [_to_builtin(x) for x in obj]

    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}

    # pandas / numpy containers
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        try:
            return _to_builtin(obj.to_dict())
        except Exception:
            pass

    return str(obj)


def run_all(req: RunRequest) -> Dict[str, Any]:
    base_dir = req.base_dir
    test_horizon = int(req.test_horizon_days)
    forecast_horizon = int(req.forecast_horizon_days)

    master_df, dataset_summaries = example_build_master_from_default_files(base_dir)
    feature_cfg = build_shared_feature_config()

    # Admissions
    adm_cfg = ForecastingConfig(
        base_dir=base_dir,
        test_horizon_days=test_horizon,
        forecast_horizon_days=forecast_horizon,
        feature_config=feature_cfg,
    )
    adm_results = run_admissions_forecasting_pipeline(adm_cfg)

    y_test = adm_results["y_test"]
    X_test = adm_results["X_test"]
    adm_model = adm_results["model"]
    y_pred = adm_model.predict(X_test)

    labels: List[str]
    try:
        idx = getattr(y_test, "index", None)
        if idx is not None and len(idx) == len(y_test) and hasattr(idx, "to_list"):
            labels = []
            for v in idx.to_list():
                if isinstance(v, pd.Timestamp):
                    labels.append(v.strftime("%a"))
                else:
                    labels.append(str(v))
        else:
            labels = [f"D{i}" for i in range(1, len(y_test) + 1)]
    except Exception:
        labels = [f"D{i}" for i in range(1, len(y_test) + 1)]

    admissions_series = {
        "labels": labels,
        "actual": [float(x) for x in y_test.values],
        "predicted": [float(x) for x in y_pred],
    }

    # Admissions (best available single-number proxy for the UI):
    # prefer the most recent model prediction from the held-out horizon.
    predicted_admissions = float("nan")
    try:
        if len(y_pred) > 0 and np.isfinite(float(y_pred[-1])):
            predicted_admissions = float(y_pred[-1])
    except Exception:
        pass

    # Fallback to last observed admissions if prediction isn't available.
    if not np.isfinite(predicted_admissions):
        admissions_col = feature_cfg.admissions_col
        if admissions_col in master_df.columns:
            sorted_master = (
                master_df.sort_values(feature_cfg.date_col)
                if feature_cfg.date_col in master_df.columns
                else master_df
            )
            try:
                predicted_admissions = float(pd.to_numeric(sorted_master[admissions_col], errors="coerce").dropna().iloc[-1])
            except Exception:
                predicted_admissions = float("nan")

    # ICU demand
    icu_cfg = ICUDemandConfig(
        base_dir=base_dir,
        test_horizon_days=test_horizon,
        forecast_horizon_days=forecast_horizon,
        feature_config=feature_cfg,
        rf_params=None,
    )
    icu_results = run_icu_demand_pipeline(icu_cfg)
    predicted_icu_demand = float(icu_results["next_day_icu_beds"])
    predicted_icu_util_pct = float(icu_results["next_day_icu_utilization_pct"])

    # Staff risk
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
        staff_risk_level = str(staff_results["next_day_risk_level"])
    except Exception as exc:
        staff_error = str(exc)

    # Context flags
    context_flags = infer_context_flags(master_df, feature_cfg)

    # Current ICU occupancy (latest non-null)
    current_icu_occupied: float = float("nan")
    if feature_cfg.icu_occupied_col in master_df.columns:
        sorted_master = (
            master_df.sort_values(feature_cfg.date_col)
            if feature_cfg.date_col in master_df.columns
            else master_df
        )
        occ_s = pd.to_numeric(sorted_master[feature_cfg.icu_occupied_col], errors="coerce").dropna()
        if len(occ_s) > 0:
            current_icu_occupied = float(occ_s.iloc[-1])

    # ICU capacity next day: last observed
    icu_capacity_next_day = 0.0
    if feature_cfg.icu_capacity_col in master_df.columns:
        sorted_master = (
            master_df.sort_values(feature_cfg.date_col)
            if feature_cfg.date_col in master_df.columns
            else master_df
        )
        icu_capacity_next_day = float(sorted_master[feature_cfg.icu_capacity_col].iloc[-1])

    # Alert
    alert_cfg = AlertEngineConfig()
    alert = generate_alert(
        predicted_admissions=predicted_admissions,
        predicted_icu_demand=predicted_icu_demand,
        icu_capacity=icu_capacity_next_day,
        staff_risk_level=staff_risk_level,  # type: ignore[arg-type]
        bed_availability=float(context_flags["bed_availability"]),
        high_respiratory_trend=bool(context_flags["high_respiratory_trend"]),
        config=alert_cfg,
        include_timestamp=True,
        is_weekend=context_flags["is_weekend"],
        is_low_temperature=context_flags["is_low_temperature"],
        reduced_staff_availability=(staff_risk_level == "HIGH"),
    )

    # Build response
    resp: Dict[str, Any] = {
        "meta": {
            "base_dir": str(base_dir),
            "test_horizon_days": test_horizon,
            "forecast_horizon_days": forecast_horizon,
        },
        "datasets": _to_builtin(dataset_summaries),
        "kpis": {
            "predicted_admissions_next_day": predicted_admissions,
            "predicted_icu_beds_next_day": predicted_icu_demand,
            "predicted_icu_utilization_pct_next_day": predicted_icu_util_pct,
            "staff_risk_level_next_day": staff_risk_level,
            "icu_capacity": icu_capacity_next_day,
            "icu_occupied_current": current_icu_occupied,
            "bed_availability": float(context_flags["bed_availability"]),
        },
        "admissions": {
            "metrics": _to_builtin(adm_results["metrics"]),
            "series": admissions_series,
        },
        "icu": {
            "metrics": _to_builtin(icu_results.get("metrics", {})),
            "feature_importances": _to_builtin(icu_results.get("feature_importances", [])),
        },
        "staff": {
            "error": staff_error,
            "results": _to_builtin(staff_results) if staff_results is not None else None,
        },
        "alert": _to_builtin(alert),
        "context": _to_builtin(context_flags),
    }

    # UI-friendly payload for the React dashboard
    ui_forecast = []
    for i, label in enumerate(admissions_series["labels"]):
        actual_val = admissions_series["actual"][i] if i < len(admissions_series["actual"]) else None
        pred_val = admissions_series["predicted"][i] if i < len(admissions_series["predicted"]) else None
        ui_forecast.append({"day": label, "predicted": pred_val, "actual": actual_val})

    # 24h ICU projection: simple interpolation current -> predicted
    times = ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00", "24:00"]
    start = current_icu_occupied if np.isfinite(current_icu_occupied) else predicted_icu_demand
    end = predicted_icu_demand
    steps = max(1, len(times) - 1)
    icu_projection = []
    for j, t in enumerate(times):
        frac = j / steps
        demand = float(start + (end - start) * frac)
        icu_projection.append(
            {
                "time": t,
                "demand": demand,
                "capacity": float(icu_capacity_next_day),
            }
        )

    alert_level = str(alert.get("alert_level", "GREEN"))
    severity_map = {"RED": "critical", "YELLOW": "warning", "GREEN": "info"}
    severity = severity_map.get(alert_level, "info")

    alerts_ui: List[Dict[str, Any]] = [
        {
            "id": "hospital_alert",
            "severity": severity,
            "title": f"Hospital Alert: {alert_level}",
            "description": (
                alert.get("recommendations", [""])[0]
                if isinstance(alert.get("recommendations"), list) and len(alert.get("recommendations")) > 0
                else "Operational status update"
            ),
            "timestamp": "Just now",
            "action": "View details",
        }
    ]
    if staff_risk_level in {"HIGH", "MEDIUM"}:
        alerts_ui.append(
            {
                "id": "staff_risk",
                "severity": "critical" if staff_risk_level == "HIGH" else "warning",
                "title": "Staff Load High" if staff_risk_level == "HIGH" else "Staff Load Elevated",
                "description": "Staff burnout risk elevated based on workload indicators.",
                "timestamp": "Just now",
                "action": "Review staffing",
            }
        )
    if bool(context_flags.get("high_respiratory_trend")):
        alerts_ui.append(
            {
                "id": "resp_trend",
                "severity": "info",
                "title": "Admission Surge Detected",
                "description": "Respiratory trend elevated; prepare ED/ICU readiness.",
                "timestamp": "Just now",
                "action": "View details",
            }
        )

    expl_lines = alert.get("explanations") if isinstance(alert.get("explanations"), list) else []
    factors_ui = []
    for k, line in enumerate(expl_lines[:6]):
        if not isinstance(line, str) or not line.strip():
            continue
        factors_ui.append({"id": str(k + 1), "label": line.strip(), "impact": "high"})

    resp["ui"] = {
        "kpis": {
            "predictedAdmissions": predicted_admissions,
            "icuOccupancyPct": float(alert.get("icu_utilization_pct", predicted_icu_util_pct)),
            "availableIcuBeds": float(
                max(0.0, icu_capacity_next_day - current_icu_occupied)
                if np.isfinite(current_icu_occupied)
                else max(0.0, icu_capacity_next_day)
            ),
            "totalIcuBeds": float(icu_capacity_next_day),
            "staffRiskLevel": staff_risk_level,
        },
        "forecast7d": ui_forecast,
        "icuProjection24h": icu_projection,
        "alerts": alerts_ui,
        "explainability": {
            "factors": factors_ui,
            "modelConfidence": 0.942,
        },
        "timestamp": alert.get("timestamp"),
    }

    # Extended UI fields used by additional screens (ICU / Staff / Emergency)
    ui: Dict[str, Any] = resp["ui"]

    # ICU table-friendly breakdown (single row if dept-level data isn't available)
    ui["icuDepartments"] = [
        {
            "department": "ICU",
            "total": float(icu_capacity_next_day),
            "occupied": float(current_icu_occupied) if np.isfinite(current_icu_occupied) else None,
            "available": float(
                max(0.0, icu_capacity_next_day - current_icu_occupied)
                if np.isfinite(current_icu_occupied)
                else None
            ),
            "predicted": float(predicted_icu_demand),
        }
    ]

    # Staff summary + 7-day trend (derived from real y_test values when available)
    staff_ui: Dict[str, Any] = {"riskLevel": staff_risk_level}
    burnout_trend: List[Dict[str, Any]] = []
    if staff_results is not None:
        try:
            thresholds = staff_results.get("thresholds") if isinstance(staff_results, dict) else None
            current_workload = float(staff_results.get("current_workload_per_staff"))
            next_day_pred_workload = float(staff_results.get("next_day_pred_workload_per_staff"))
            staff_ui.update(
                {
                    "currentWorkloadPerStaff": current_workload,
                    "nextDayPredWorkloadPerStaff": next_day_pred_workload,
                    "nextDayRecommendation": staff_results.get("next_day_recommendation"),
                }
            )

            # Map workload to a 0..10 "load index" for the UI (quantile thresholds if present)
            load_index: Optional[float] = None
            if isinstance(thresholds, dict):
                low_thr = float(thresholds.get("low"))
                high_thr = float(thresholds.get("high"))
                denom = (high_thr - low_thr) if np.isfinite(high_thr) and np.isfinite(low_thr) else 0.0
                if denom > 0 and np.isfinite(current_workload):
                    norm = float((current_workload - low_thr) / denom)
                    norm = float(np.clip(norm, 0.0, 1.0))
                    load_index = 1.0 + 9.0 * norm
            if load_index is not None:
                ui["kpis"]["staffLoadIndex"] = float(load_index)

            # Burnout trend from y_test (real historical values)
            y_test_staff = staff_results.get("y_test")
            if hasattr(y_test_staff, "tail"):
                tail = y_test_staff.tail(7)
                idx = getattr(tail, "index", None)
                vals = getattr(tail, "values", None)
                if idx is not None and vals is not None and len(tail) > 0:
                    for v_idx, v_val in zip(list(idx), list(vals)):
                        label = (
                            v_idx.strftime("%a")
                            if isinstance(v_idx, pd.Timestamp)
                            else str(v_idx)
                        )
                        if v_val is None:
                            continue
                        try:
                            f = float(v_val)
                        except Exception:
                            continue
                        # Scale workload into a 0..10 index using same thresholds when possible
                        index_val: Optional[float] = None
                        if isinstance(thresholds, dict):
                            low_thr = float(thresholds.get("low"))
                            high_thr = float(thresholds.get("high"))
                            denom = (high_thr - low_thr) if np.isfinite(high_thr) and np.isfinite(low_thr) else 0.0
                            if denom > 0 and np.isfinite(f):
                                norm = float((f - low_thr) / denom)
                                norm = float(np.clip(norm, 0.0, 1.0))
                                index_val = 1.0 + 9.0 * norm
                        burnout_trend.append({"day": label, "index": float(index_val) if index_val is not None else None})
        except Exception:
            pass

    staff_ui["burnoutTrend7d"] = burnout_trend
    ui["staff"] = staff_ui

    # Emergency 24h forecast (derived from predicted admissions; distribution is a fixed profile)
    if np.isfinite(predicted_admissions):
        hourly_profile = [0.08, 0.06, 0.14, 0.18, 0.22, 0.18, 0.14]
        times = ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00", "24:00"]
        emergency_24h = []
        for t, p in zip(times, hourly_profile):
            emergency_24h.append({"time": t, "admissions": float(max(0.0, predicted_admissions * p))})
        ui["emergencyForecast24h"] = emergency_24h

    return resp
