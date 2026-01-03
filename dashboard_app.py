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

from pipeline import run_pipeline


st.set_page_config(
    page_title="Hospital Operations Dashboard",
    layout="wide",
)


def run_pipelines(
    base_dir: Path,
    test_horizon: int,
    forecast_horizon: int,
) -> Dict[str, Any]:
    """Execute pipelines via the centralized pipeline controller.

    The controller returns a stable summary + nested details. For this
    dashboard we also run admissions forecasting to retain the existing
    evaluation chart.
    """

    result = run_pipeline(
        {
            "base_dir": base_dir,
            "test_horizon_days": int(test_horizon),
            "forecast_horizon_days": int(forecast_horizon),
        }
    )

    details = result.get("details", {}) if isinstance(result, dict) else {}
    decision_engine_alert = details.get("decision_engine_alert", {})

    admissions_series = {}
    if isinstance(details, dict):
        admissions = details.get("admissions")
        if isinstance(admissions, dict) and isinstance(admissions.get("series"), dict):
            admissions_series = admissions.get("series")

    outputs: Dict[str, Any] = {
        "admissions_series": admissions_series,
        "next_day_admissions": float(result.get("predicted_admissions"))
        if result.get("predicted_admissions") is not None
        else float("nan"),
        "next_day_icu_beds": float(result.get("predicted_icu_beds"))
        if result.get("predicted_icu_beds") is not None
        else 0.0,
        "next_day_icu_util_pct": float(result.get("icu_utilization", 0.0)),
        "staff_risk_level": str(details.get("staff", {}).get("risk_level", "UNKNOWN")),
        "alert_response": decision_engine_alert
        if isinstance(decision_engine_alert, dict)
        else {"alert_level": str(result.get("alert_level", "GREEN")), "icu_utilization_pct": float(result.get("icu_utilization", 0.0)), "recommendations": [], "explanations": []},
    }

    return outputs


def render_dashboard(outputs: Dict[str, Any]) -> None:
    """Render Streamlit components from pipeline outputs."""

    admissions_series = outputs.get("admissions_series") if isinstance(outputs, dict) else None
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

    labels = []
    actual = []
    predicted = []
    if isinstance(admissions_series, dict):
        labels = admissions_series.get("labels") if isinstance(admissions_series.get("labels"), list) else []
        actual = admissions_series.get("actual") if isinstance(admissions_series.get("actual"), list) else []
        predicted = admissions_series.get("predicted") if isinstance(admissions_series.get("predicted"), list) else []

    if labels and (actual or predicted):
        n = min(len(labels), len(actual) if actual else len(labels), len(predicted) if predicted else len(labels))
        plot_df = pd.DataFrame(
            {
                "day": labels[:n],
                "actual_admissions": actual[:n] if actual else [None] * n,
                "predicted_admissions": predicted[:n] if predicted else [None] * n,
            }
        ).set_index("day")
        st.line_chart(plot_df)
    else:
        st.info("Admissions evaluation series unavailable for this run.")

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
