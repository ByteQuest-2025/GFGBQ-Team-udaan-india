"""Streamlit dashboard for hospital forecasting and alerts.

This app reuses the existing modular pipeline to:
- Load real hospital datasets.
- Run forecasting models for admissions and ICU demand.
- Estimate staff workload risk.
- Generate an alert level and explanations.
- Visualize results in an operational dashboard.

Run with:
    streamlit run dashboard_app.py

Requirements (install via pip if needed):
    streamlit pandas numpy scikit-learn matplotlib
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

from hospital_data_pipeline import example_build_master_from_default_files
from hospital_decision_engine import AlertEngineConfig, generate_alert
from hospital_feature_engineering import FeatureConfig
from hospital_forecasting import ForecastingConfig, run_admissions_forecasting_pipeline
from hospital_icu_demand import ICUDemandConfig, run_icu_demand_pipeline
from hospital_staff_risk import StaffRiskConfig, run_staff_risk_pipeline
from main import build_shared_feature_config, infer_context_flags


st.set_page_config(
    page_title="Hospital Operations Dashboard",
    layout="wide",
)


def run_pipelines(
    base_dir: Path,
    test_horizon: int,
    forecast_horizon: int,
) -> Dict[str, Any]:
    """Execute core pipelines and return a bundle of outputs.

    This function mirrors the logic in main.py but returns artefacts for
    use in interactive visualization instead of printing to stdout.
    """

    master_df, dataset_summaries = example_build_master_from_default_files(base_dir)
    feature_cfg: FeatureConfig = build_shared_feature_config()

    # Admissions forecasting (used mainly for metrics in this dashboard)
    adm_cfg = ForecastingConfig(
        base_dir=base_dir,
        test_horizon_days=test_horizon,
        forecast_horizon_days=forecast_horizon,
        feature_config=feature_cfg,
    )
    adm_results = run_admissions_forecasting_pipeline(adm_cfg)

    # Approximate next-day admissions using last observed value; in a
    # production deployment you can plug in a dedicated admissions
    # forecast at this point.
    admissions_col = feature_cfg.admissions_col
    if admissions_col in master_df.columns:
        if feature_cfg.date_col in master_df.columns:
            master_sorted = master_df.sort_values(feature_cfg.date_col)
        else:
            master_sorted = master_df
        next_day_admissions = float(master_sorted[admissions_col].iloc[-1])
    else:
        next_day_admissions = float("nan")

    # ICU demand forecast
    icu_cfg = ICUDemandConfig(
        base_dir=base_dir,
        test_horizon_days=test_horizon,
        forecast_horizon_days=forecast_horizon,
        feature_config=feature_cfg,
        rf_params=None,
    )
    icu_results = run_icu_demand_pipeline(icu_cfg)
    next_day_icu_beds = float(icu_results["next_day_icu_beds"])
    next_day_icu_util_pct = float(icu_results["next_day_icu_utilization_pct"])

    # Staff risk (may fail gracefully if features are unavailable)
    staff_risk_level = "UNKNOWN"
    staff_results: Optional[Dict[str, Any]] = None
    try:
        staff_cfg = StaffRiskConfig(
            base_dir=base_dir,
            forecast_horizon_days=forecast_horizon,
            test_horizon_days=test_horizon,
            feature_config=feature_cfg,
        )
        staff_results = run_staff_risk_pipeline(staff_cfg)
        staff_risk_level = str(staff_results["next_day_risk_level"])
    except Exception as exc:  # pragma: no cover - defensive
        staff_results = None
        st.warning(
            f"Staff risk pipeline failed; defaulting risk level to UNKNOWN. Reason: {exc}",
            icon="⚠️",
        )

    # Context flags (weekend, temperature, respiratory trend, beds)
    context_flags = infer_context_flags(master_df, feature_cfg)

    # ICU capacity for next day: assume same as last observed day
    icu_capacity_next_day = 0.0
    if feature_cfg.icu_capacity_col in master_df.columns:
        if feature_cfg.date_col in master_df.columns:
            master_sorted = master_df.sort_values(feature_cfg.date_col)
        else:
            master_sorted = master_df
        icu_capacity_next_day = float(
            master_sorted[feature_cfg.icu_capacity_col].iloc[-1]
        )

    # Generate alert
    alert_cfg = AlertEngineConfig()
    alert_response = generate_alert(
        predicted_admissions=next_day_admissions,
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

    outputs: Dict[str, Any] = {
        "master_df": master_df,
        "feature_cfg": feature_cfg,
        "adm_results": adm_results,
        "icu_results": icu_results,
        "staff_results": staff_results,
        "next_day_admissions": next_day_admissions,
        "next_day_icu_beds": next_day_icu_beds,
        "next_day_icu_util_pct": next_day_icu_util_pct,
        "staff_risk_level": staff_risk_level,
        "context_flags": context_flags,
        "alert_response": alert_response,
    }

    return outputs


def render_dashboard(outputs: Dict[str, Any]) -> None:
    """Render Streamlit components from pipeline outputs."""

    adm_results = outputs["adm_results"]
    icu_results = outputs["icu_results"]
    staff_results = outputs["staff_results"]
    next_day_admissions = outputs["next_day_admissions"]
    next_day_icu_beds = outputs["next_day_icu_beds"]
    next_day_icu_util_pct = outputs["next_day_icu_util_pct"]
    staff_risk_level = outputs["staff_risk_level"]
    alert_response = outputs["alert_response"]

    st.markdown("## Operational Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Next-day admissions (approx)",
            value=f"{next_day_admissions:.0f}" if not np.isnan(next_day_admissions) else "N/A",
        )

    with col2:
        st.metric(
            label="Next-day ICU beds (predicted)",
            value=f"{next_day_icu_beds:.0f}",
        )

    with col3:
        st.metric(
            label="Next-day ICU utilization (%)",
            value=f"{next_day_icu_util_pct:.1f}",
        )

    # Admission forecast (7-day evaluation window)
    st.markdown("---")
    st.markdown("### 7-Day Admission Forecast Window (Evaluation)")

    y_test = adm_results["y_test"]
    X_test = adm_results["X_test"]
    model = adm_results["model"]
    y_pred = model.predict(X_test)

    # Build a simple dataframe for plotting (index treated as relative days)
    forecast_df = pd.DataFrame(
        {
            "day": range(1, len(y_test) + 1),
            "actual_admissions": y_test.values,
            "predicted_admissions": y_pred,
        }
    )
    forecast_df = forecast_df.set_index("day")

    st.line_chart(forecast_df)

    # ICU occupancy gauge (color-coded)
    st.markdown("---")
    st.markdown("### ICU Occupancy Status")

    icu_col1, icu_col2 = st.columns([1, 2])

    icu_util = alert_response["icu_utilization_pct"]
    alert_level = alert_response["alert_level"]

    emoji_map = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}
    emoji = emoji_map.get(alert_level, "⚪")

    with icu_col1:
        st.markdown(f"#### Alert: {emoji} {alert_level}")
        st.metric("ICU utilization (%)", f"{icu_util:.1f}")

    with icu_col2:
        st.progress(min(max(int(icu_util), 0), 100))

    # Staff workload status
    st.markdown("---")
    st.markdown("### Staff Workload & Burnout Risk")

    staff_emoji_map = {"LOW": "✅", "MEDIUM": "⚠️", "HIGH": "🔥", "UNKNOWN": "❓"}
    staff_emoji = staff_emoji_map.get(staff_risk_level, "❓")

    st.markdown(f"**Next-day staff risk:** {staff_emoji} {staff_risk_level}")

    if staff_results is not None:
        st.write(
            "Predicted workload per staff (next day):",
            f"{staff_results['next_day_pred_workload_per_staff']:.2f}",
        )
        st.write("Current workload per staff:", f"{staff_results['current_workload_per_staff']:.2f}")

    # Alert recommendations and explanations
    st.markdown("---")
    st.markdown("### Alert Recommendations and Explanations")

    rec_tab, expl_tab, json_tab = st.tabs(["Recommendations", "Explanations", "Raw JSON"])

    with rec_tab:
        for line in alert_response["recommendations"]:
            st.markdown(f"- {line}")

    with expl_tab:
        for line in alert_response["explanations"]:
            st.markdown(f"- {line}")

    with json_tab:
        st.json(alert_response)


def main() -> None:
    st.title("Hospital Operations Forecasting Dashboard")

    st.sidebar.header("Configuration")
    base_dir_str = st.sidebar.text_input(
        "Base directory",
        value=".",
        autocomplete="off",
    )
    base_dir = Path(base_dir_str).resolve()
    test_horizon = st.sidebar.number_input(
        "Evaluation horizon (days)", min_value=3, max_value=30, value=7, step=1
    )
    forecast_horizon = st.sidebar.number_input(
        "Forecast horizon (days)", min_value=1, max_value=7, value=1, step=1
    )

    run_button = st.sidebar.button("Run pipeline")

    if run_button:
        with st.spinner("Running pipelines on real hospital data..."):
            try:
                outputs = run_pipelines(
                    base_dir=base_dir,
                    test_horizon=int(test_horizon),
                    forecast_horizon=int(forecast_horizon),
                )
                render_dashboard(outputs)
            except Exception as exc:  # pragma: no cover - defensive
                st.error(f"Pipeline execution failed: {exc}")
    else:
        st.info("Set configuration in the sidebar and click 'Run pipeline' to view results.")


if __name__ == "__main__":
    main()
