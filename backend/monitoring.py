from __future__ import annotations

"""Prometheus metrics for the hospital backend.

This module defines process-wide metrics that can be scraped by a
Prometheus server via the /metrics endpoint.
"""

from typing import Any, Dict

from prometheus_client import Counter, Gauge


PIPELINE_RUNS_TOTAL = Counter(
    "hospital_pipeline_runs_total",
    "Total number of pipeline runs",
    ["status"],
)

PIPELINE_LAST_RUN_TIMESTAMP = Gauge(
    "hospital_pipeline_last_run_timestamp",
    "Unix timestamp of the last completed pipeline run",
)

PIPELINE_ADMISSIONS_MAE = Gauge(
    "hospital_admissions_mae",
    "Last recorded MAE for admissions forecast",
)

PIPELINE_ICU_MAE = Gauge(
    "hospital_icu_mae",
    "Last recorded MAE for ICU demand forecast",
)

PIPELINE_STAFF_MAE = Gauge(
    "hospital_staff_mae",
    "Last recorded MAE for staff stress model",
)


def update_from_monitoring_payload(monitoring: Dict[str, Any]) -> None:
    """Update Prometheus metrics from a monitoring payload.

    The payload is expected to be the `monitoring` dict produced by
    `pipeline_service.run_all`.
    """

    PIPELINE_RUNS_TOTAL.labels(status="success").inc()

    ts = monitoring.get("run_completed_at")
    # Prometheus Gauges expect numeric values; we store a best-effort
    # Unix timestamp when available.
    if isinstance(ts, (int, float)):
        PIPELINE_LAST_RUN_TIMESTAMP.set(float(ts))

    admissions = monitoring.get("admissions_metrics") or {}
    icu = monitoring.get("icu_metrics") or {}
    staff = monitoring.get("staff_metrics") or {}

    mae_keys = ["MAE", "mae"]

    for key in mae_keys:
        if key in admissions:
            try:
                PIPELINE_ADMISSIONS_MAE.set(float(admissions[key]))
            except Exception:
                pass
            break

    for key in mae_keys:
        if key in icu:
            try:
                PIPELINE_ICU_MAE.set(float(icu[key]))
            except Exception:
                pass
            break

    if isinstance(staff, dict):
        for key in mae_keys:
            if key in staff:
                try:
                    PIPELINE_STAFF_MAE.set(float(staff[key]))
                except Exception:
                    pass
                break
