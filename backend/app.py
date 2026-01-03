from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.pipeline_service import RunRequest, run_all


class RunBody(BaseModel):
    base_dir: str = Field(default=".")
    test_horizon_days: int = Field(default=7, ge=3, le=90)
    forecast_horizon_days: int = Field(default=1, ge=1, le=30)


class WhatIfBody(BaseModel):
    admission_surge_pct: float = Field(default=0.0, ge=-50.0, le=100.0)
    temperature_c: float = Field(default=15.0, ge=-50.0, le=60.0)
    staff_availability_pct: float = Field(default=100.0, ge=0.0, le=100.0)


app = FastAPI(title="Hospital Operations Backend", version="0.1.0")

# CORS for local Vite dev + optional overrides via env
cors_origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000",
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in cors_origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"] ,
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
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/run")
def api_run(body: RunBody) -> Dict[str, Any]:
    global _last_result
    req = RunRequest(
        base_dir=Path(body.base_dir).resolve(),
        test_horizon_days=body.test_horizon_days,
        forecast_horizon_days=body.forecast_horizon_days,
    )
    _last_result = run_all(req)
    return _last_result


@app.get("/api/dashboard")
def api_dashboard() -> Dict[str, Any]:
    # Returns the last run result if available; otherwise runs with defaults.
    return _get_latest_dashboard()


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
