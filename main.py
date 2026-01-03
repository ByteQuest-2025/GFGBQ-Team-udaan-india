"""End-to-end hospital forecasting and alert pipeline.

This script wires together the modular components in this project to
run a full decision workflow:

1. Load and validate datasets into a master dataframe.
2. Perform feature engineering for admissions, ICU, environment, and staff.
3. Train/evaluate prediction models (admissions, ICU demand, staff risk).
4. Generate an operational alert for the next day.
5. Print human-readable summaries and a JSON alert payload.

Usage
-----
    python main.py --base-dir . --test-horizon 7

Notes
-----
- This script assumes that the CSV files described in
  `hospital_data_pipeline` are present under `--base-dir`.
- Column names used in the FeatureConfig below are illustrative and
  should be aligned with your actual master dataframe schema.
- No synthetic data is generated; all computations assume real data
  from your hospital systems.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import numpy as np
    import pandas as pd
except ModuleNotFoundError as exc:
    missing = getattr(exc, "name", "a required dependency")
    raise SystemExit(
        "Missing dependency: "
        f"{missing}.\n\n"
        "You're likely running this script with the wrong Python interpreter.\n"
        "Use the workspace virtual environment Python instead:\n\n"
        "  C:/Users/dasne/Desktop/udaanindia/.venv/Scripts/python.exe main.py --base-dir . --test-horizon 7\n"
    ) from exc

from hospital_data_pipeline import example_build_master_from_default_files
from hospital_decision_engine import AlertEngineConfig, generate_alert
from hospital_feature_engineering import FeatureConfig
from hospital_forecasting import ForecastingConfig, run_admissions_forecasting_pipeline
from hospital_icu_demand import ICUDemandConfig, run_icu_demand_pipeline
from hospital_staff_risk import StaffRiskConfig, run_staff_risk_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hospital forecasting and alert pipeline")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=".",
        help="Base directory containing hospital CSV datasets.",
    )
    parser.add_argument(
        "--test-horizon",
        type=int,
        default=7,
        help="Number of days reserved for evaluation at the end of the series.",
    )
    parser.add_argument(
        "--forecast-horizon",
        type=int,
        default=1,
        help="Forecast horizon in days for next-day style predictions.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to write the final alert JSON payload.",
    )
    return parser.parse_args()


def build_shared_feature_config() -> FeatureConfig:
    """Create a shared FeatureConfig used across models.

    Adjust the column names here to match your master dataframe. The
    defaults reflect the prefixed naming produced by
    `hospital_data_pipeline.build_master_dataframe` when using default
    dataset names.
    """

    return FeatureConfig(
        date_col="date",
        admissions_col="emergency__admissions",  # update to your admissions column
        icu_occupied_col="icu__beds_occupied",   # update to your ICU occupied column
        icu_capacity_col="icu__beds_capacity",   # update to your ICU capacity column
        staff_count_cols=[
            "staff__doctors_on_duty",
            "staff__nurses_on_duty",
        ],
        patient_count_col="emergency__admissions",  # or current census column
        weather_cols=[
            "env_health__temperature_c",
            "env_health__aqi",
        ],
        health_trend_cols=[
            "env_health__flu_index",
        ],
        holiday_col="calendar__is_holiday",
        weekend_col="calendar__is_weekend",
        forecast_horizon_days=1,
    )


def infer_context_flags(master_df: pd.DataFrame, feature_cfg: FeatureConfig) -> Dict[str, Any]:
    """Infer contextual boolean flags from the most recent day.

    - is_weekend: from calendar features if available.
    - is_low_temperature: from environmental temperature if available.
    - high_respiratory_trend: from any respiratory/flu-like index.
    - bed_availability: derived from ICU capacity - occupied if possible.
    """

    df = master_df.copy()
    if feature_cfg.date_col in df.columns:
        df = df.sort_values(feature_cfg.date_col)
    last = df.iloc[-1]

    # Weekend flag
    is_weekend: Optional[bool] = None
    if feature_cfg.weekend_col and feature_cfg.weekend_col in df.columns:
        is_weekend = bool(last[feature_cfg.weekend_col] == 1)

    # Low temperature flag (e.g., < 18°C)
    is_low_temperature: Optional[bool] = None
    temp_col_candidates = [c for c in df.columns if "temperature" in c.lower()]
    if temp_col_candidates:
        tcol = temp_col_candidates[0]
        temp_val = float(last[tcol]) if not pd.isna(last[tcol]) else None
        if temp_val is not None:
            is_low_temperature = temp_val < 18.0

    # Respiratory / flu trend: look for a column containing "flu" or "resp"
    high_respiratory_trend = False
    resp_cols = [
        c
        for c in df.columns
        if ("flu" in c.lower() or "resp" in c.lower()) and pd.api.types.is_numeric_dtype(df[c])
    ]
    if resp_cols:
        rcol = resp_cols[0]
        series = df[rcol].astype("float64")
        median_val = float(series.median()) if not series.empty else 0.0
        last_val = float(last[rcol]) if not pd.isna(last[rcol]) else median_val
        high_respiratory_trend = last_val > median_val

    # Bed availability: ICU capacity - occupied.
    # Use the most recent row where both values are present to avoid NaNs.
    bed_availability = 0.0
    if feature_cfg.icu_capacity_col in df.columns and feature_cfg.icu_occupied_col in df.columns:
        cap_series = pd.to_numeric(df[feature_cfg.icu_capacity_col], errors="coerce")
        occ_series = pd.to_numeric(df[feature_cfg.icu_occupied_col], errors="coerce")
        valid = cap_series.notna() & occ_series.notna()
        if valid.any():
            cap = float(cap_series.loc[valid].iloc[-1])
            occ = float(occ_series.loc[valid].iloc[-1])
            bed_availability = max(cap - occ, 0.0)

    return {
        "is_weekend": is_weekend,
        "is_low_temperature": is_low_temperature,
        "high_respiratory_trend": high_respiratory_trend,
        "bed_availability": bed_availability,
    }


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()

    print("=== Step 1: Load datasets and build master dataframe ===")
    master_df, dataset_summaries = example_build_master_from_default_files(base_dir)
    print("Master dataframe shape:", master_df.shape)

    shared_feature_cfg = build_shared_feature_config()

    print("\n=== Step 2: Admissions forecasting model (for evaluation) ===")
    fc = ForecastingConfig(
        base_dir=base_dir,
        test_horizon_days=args.test_horizon,
        forecast_horizon_days=args.forecast_horizon,
        feature_config=shared_feature_cfg,
    )
    adm_results = run_admissions_forecasting_pipeline(fc)
    print("Admissions forecast metrics:", adm_results["metrics"])

    # For simplicity and transparency, approximate next-day admissions by
    # the most recent observed admissions. In a deployment scenario, you
    # can instead use the admissions model to generate a true forecast
    # for the next day and pass it here.
    admissions_col = shared_feature_cfg.admissions_col
    last_admissions_val = float("nan")
    if admissions_col in master_df.columns:
        master_df_sorted = (
            master_df.sort_values(shared_feature_cfg.date_col)
            if shared_feature_cfg.date_col in master_df.columns
            else master_df
        )
        s = pd.to_numeric(master_df_sorted[admissions_col], errors="coerce")
        if s.notna().any():
            last_admissions_val = float(s.dropna().iloc[-1])

    print("Approximate next-day admissions (using last observed):", last_admissions_val)

    print("\n=== Step 3: ICU demand prediction model ===")
    icu_cfg = ICUDemandConfig(
        base_dir=base_dir,
        test_horizon_days=args.test_horizon,
        forecast_horizon_days=args.forecast_horizon,
        feature_config=shared_feature_cfg,
        rf_params=None,
    )
    icu_results = run_icu_demand_pipeline(icu_cfg)
    print("ICU demand forecast metrics:", icu_results["metrics"])
    next_day_icu_beds = float(icu_results["next_day_icu_beds"])
    next_day_icu_util_pct = float(icu_results["next_day_icu_utilization_pct"])
    print("Next-day predicted ICU beds:", next_day_icu_beds)
    print("Next-day predicted ICU utilization (%):", f"{next_day_icu_util_pct:.2f}")

    print("\n=== Step 4: Staff workload and burnout risk model ===")
    staff_risk_level = "UNKNOWN"
    try:
        staff_cfg = StaffRiskConfig(
            base_dir=base_dir,
            forecast_horizon_days=args.forecast_horizon,
            test_horizon_days=args.test_horizon,
            feature_config=shared_feature_cfg,
        )
        staff_results = run_staff_risk_pipeline(staff_cfg)
        staff_risk_level = str(staff_results["next_day_risk_level"])
        print("Staff stress forecast metrics:", staff_results["metrics"])
        print("Next-day staff risk level:", staff_risk_level)
    except Exception as exc:  # pragma: no cover - defensive
        print("Staff risk pipeline failed; defaulting staff_risk_level=UNKNOWN")
        print("Reason:", exc)

    print("\n=== Step 5: Decision engine and alert generation ===")
    context_flags = infer_context_flags(master_df, shared_feature_cfg)

    # For ICU capacity, assume same capacity as the last observed day
    icu_capacity_next_day = 0.0
    if shared_feature_cfg.icu_capacity_col in master_df.columns:
        master_df_sorted = (
            master_df.sort_values(shared_feature_cfg.date_col)
            if shared_feature_cfg.date_col in master_df.columns
            else master_df
        )
        cap_s = pd.to_numeric(master_df_sorted[shared_feature_cfg.icu_capacity_col], errors="coerce")
        if cap_s.notna().any():
            icu_capacity_next_day = float(cap_s.dropna().iloc[-1])

    alert_cfg = AlertEngineConfig()
    alert_response = generate_alert(
        predicted_admissions=last_admissions_val,
        predicted_icu_demand=next_day_icu_beds,
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

    print("\nFinal alert level:", alert_response["alert_level"])
    print("ICU utilization (%):", f"{alert_response['icu_utilization_pct']:.2f}")

    print("\nRecommendations:")
    for line in alert_response["recommendations"]:
        print("-", line)

    print("\nExplanations:")
    for line in alert_response["explanations"]:
        print("-", line)

    alert_json = json.dumps(alert_response, indent=2)

    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.write_text(alert_json, encoding="utf-8")
        print(f"\nAlert JSON written to: {output_path}")

    print("\nRaw JSON payload:")
    print(alert_json)


if __name__ == "__main__":
    main()
