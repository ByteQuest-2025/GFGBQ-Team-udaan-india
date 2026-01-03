from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import logging

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from backend.config import get_settings
from backend.monitoring import update_from_monitoring_payload
from backend.history_store import store_monitoring_snapshot, get_recent_runs
from backend.pipeline_service import RunRequest, run_all, DataQualityError


class RunBody(BaseModel):
    base_dir: str = Field(default=".")
    test_horizon_days: int = Field(default=7, ge=3, le=90)
    forecast_horizon_days: int = Field(default=1, ge=1, le=30)


class WhatIfBody(BaseModel):
    admission_surge_pct: float = Field(default=0.0, ge=-50.0, le=100.0)
    temperature_c: float = Field(default=15.0, ge=-50.0, le=60.0)
    staff_availability_pct: float = Field(default=100.0, ge=0.0, le=100.0)


logger = logging.getLogger("hospital_backend")


settings = get_settings()

app = FastAPI(title=settings.app_name, version="0.1.0")

# CORS for local Vite dev + optional overrides via env
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_last_result: Optional[Dict[str, Any]] = None


def _get_latest_dashboard() -> Dict[str, Any]:
    global _last_result
    if _last_result is None:
        req = RunRequest(base_dir=Path(".").resolve(), test_horizon_days=7, forecast_horizon_days=1)
        _last_result = run_all(req)
    return _last_result


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    return (
        "<h2>Hospital Operations Backend</h2>"
        "<ul>"
        "<li><a href='/docs'>API docs</a></li>"
        "<li><a href='/api/ui/dashboard'>UI dashboard JSON</a></li>"
        "<li><a href='/health'>Health</a></li>"
        "</ul>"
    )


@app.get("/favicon.ico")
def favicon() -> Response:
    # Avoid noisy 404s in browser devtools.
    return Response(status_code=204)


@app.get("/health")
def health() -> Dict[str, Any]:
    """Lightweight liveness check.

    Returns basic status plus environment so platforms can verify the
    service is up. For deeper checks, see `/health/ready`.
    """

    return {"status": "ok", "environment": settings.environment}


@app.get("/health/ready")
def health_ready() -> Dict[str, Any]:
    """Readiness check with a lightweight pipeline sanity check.

    In a production setting this can be extended to validate access to
    required datasets, model artifacts, or downstream services without
    executing the full forecasting pipeline on every request.
    """

    ok = True
    details: Dict[str, Any] = {"pipeline_cached": _last_result is not None}
    try:
        # Run a minimal dashboard fetch which will lazily trigger the
        # default pipeline run on first call if needed.
        _ = _get_latest_dashboard()
    except Exception as exc:  # pragma: no cover - defensive in prod
        ok = False
        details["error"] = str(exc)

    return {"status": "ok" if ok else "error", **details}


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:  # pragma: no cover - safety net
    """Catch-all handler to avoid leaking internal errors in responses.

    Detailed stack traces should be emitted to logs in production
    instead of being returned to clients.
    """

    logger.exception("Unhandled exception during request", extra={"path": str(request.url)})

    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "path": str(request.url),
        },
    )


@app.exception_handler(DataQualityError)
async def data_quality_exception_handler(request: Request, exc: DataQualityError) -> JSONResponse:
    """Return a clear, structured response when data quality gates fail."""

    logger.warning("Data quality gate blocked forecast", extra={"path": str(request.url), "error": str(exc)})

    return JSONResponse(
        status_code=503,
        content={
            "detail": str(exc),
            "code": "data_quality_error",
        },
    )


@app.post("/api/run")
def api_run(body: RunBody) -> Dict[str, Any]:
    global _last_result
    req = RunRequest(
        base_dir=Path(body.base_dir).resolve(),
        test_horizon_days=body.test_horizon_days,
        forecast_horizon_days=body.forecast_horizon_days,
    )
    _last_result = run_all(req)

    monitoring = _last_result.get("monitoring", {}) if isinstance(_last_result, dict) else {}
    if isinstance(monitoring, dict):
        try:
            update_from_monitoring_payload(monitoring)
        except Exception:  # pragma: no cover - metrics should not break the API
            logger.exception("Failed to update Prometheus metrics from monitoring payload")

        try:
            store_monitoring_snapshot(monitoring)
        except Exception:  # pragma: no cover - persistence is best-effort
            logger.exception("Failed to persist monitoring snapshot to SQLite")
    logger.info(
        "pipeline_run_completed",
        extra={
            "monitoring": monitoring,
        },
    )

    return _last_result


@app.get("/api/dashboard")
def api_dashboard() -> Dict[str, Any]:
    # Returns the last run result if available; otherwise runs with defaults.
    return _get_latest_dashboard()


@app.get("/metrics")
def metrics() -> Response:
    """Expose Prometheus metrics for scraping.

    This uses the default process-wide registry from `prometheus_client`.
    """

    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.get("/api/monitoring/last-run")
def api_monitoring_last_run() -> Dict[str, Any]:
    """Expose monitoring snapshot from the most recent pipeline run.

    This is a thin wrapper around the `monitoring` section produced by
    `pipeline_service.run_all` and is intended for dashboards or external
    monitoring systems.
    """

    data = _get_latest_dashboard()
    monitoring = data.get("monitoring") if isinstance(data, dict) else None
    return {"monitoring": monitoring}


@app.get("/api/monitoring/history")
def api_monitoring_history(limit: int = 50) -> Dict[str, Any]:
    """Return recent monitoring runs from the SQLite history store."""

    runs = get_recent_runs(limit=limit)
    return {"runs": runs}


@app.get("/api/ui/dashboard")
def api_ui_dashboard() -> Dict[str, Any]:
    data = api_dashboard()
    ui = data.get("ui")
    # Fallback: if older cache doesn't have UI payload, return the raw payload.
    return ui if isinstance(ui, dict) else data


@app.post("/api/ui/whatif")
def api_ui_whatif(body: WhatIfBody) -> Dict[str, Any]:
    ui = api_ui_dashboard()

    kpis = ui.get("kpis") if isinstance(ui, dict) else None
    kpis = kpis if isinstance(kpis, dict) else {}

    baseline_admissions = float(kpis.get("predictedAdmissions") or 248.0)
    baseline_icu_pct = float(kpis.get("icuOccupancyPct") or 87.5)
    baseline_staff_load = float(kpis.get("staffLoadIndex") or 6.8)

    admission_surge = float(body.admission_surge_pct)
    temperature_c = float(body.temperature_c)
    staff_availability = float(body.staff_availability_pct)

    # Match the existing UI logic, but drive baseline from real pipeline KPIs.
    admission_impact = (admission_surge / 100.0) * baseline_admissions
    temp_impact = (15.0 - temperature_c) * 3.0  # cold weather increases admissions
    projected_admissions = int(round(baseline_admissions + admission_impact + temp_impact))

    icu_impact = (admission_surge / 100.0) * 10.0 + (15.0 - temperature_c) * 0.5
    projected_icu_pct = float(min(100.0, baseline_icu_pct + icu_impact))

    staff_impact = (100.0 - staff_availability) / 100.0
    projected_staff_load = float(baseline_staff_load + staff_impact * 3.0)

    return {
        "baseline": {
            "admissions": int(round(baseline_admissions)),
            "icuOccupancyPct": baseline_icu_pct,
            "staffLoadIndex": baseline_staff_load,
        },
        "projections": {
            "admissions": projected_admissions,
            "icuOccupancyPct": projected_icu_pct,
            "staffLoadIndex": projected_staff_load,
        },
    }


@app.get("/api/alert")
def api_alert() -> Dict[str, Any]:
    data = api_dashboard()
    return {"alert": data.get("alert"), "kpis": data.get("kpis")}


@app.get("/api/admissions")
def api_admissions() -> Dict[str, Any]:
    data = api_dashboard()
    return {"admissions": data.get("admissions"), "kpis": data.get("kpis")}


@app.get("/api/icu")
def api_icu() -> Dict[str, Any]:
    data = api_dashboard()
    return {"icu": data.get("icu"), "kpis": data.get("kpis"), "alert": data.get("alert")}


@app.get("/api/staff")
def api_staff() -> Dict[str, Any]:
    data = api_dashboard()
    return {"staff": data.get("staff"), "kpis": data.get("kpis")}
