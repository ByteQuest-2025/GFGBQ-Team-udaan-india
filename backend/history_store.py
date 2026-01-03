from __future__ import annotations

"""SQLite-backed storage for monitoring snapshots.

This is intended for lightweight historical analysis and trend charts.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import sqlite3

from backend.config import get_settings


_INITIALIZED = False


def _db_path() -> Path:
    settings = get_settings()
    return Path(settings.monitoring_db_path).resolve()


def _ensure_initialized() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return

    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS monitoring_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_started_at TEXT,
                run_completed_at TEXT,
                test_horizon_days INTEGER,
                forecast_horizon_days INTEGER,
                alert_level TEXT,
                staff_risk_level TEXT,
                admissions_mae REAL,
                icu_mae REAL,
                staff_mae REAL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()

    _INITIALIZED = True


def _extract_mae(metrics: Any) -> Optional[float]:
    if not isinstance(metrics, dict):
        return None
    for key in ("MAE", "mae"):
        if key in metrics:
            try:
                return float(metrics[key])
            except Exception:
                return None
    return None


def store_monitoring_snapshot(monitoring: Dict[str, Any]) -> None:
    """Persist a single monitoring snapshot to SQLite.

    The input should be the `monitoring` dict from `pipeline_service.run_all`.
    """

    _ensure_initialized()
    path = _db_path()
    conn = sqlite3.connect(path)
    try:
        admissions = monitoring.get("admissions_metrics") or {}
        icu = monitoring.get("icu_metrics") or {}
        staff = monitoring.get("staff_metrics") or {}

        conn.execute(
            """
            INSERT INTO monitoring_runs (
                run_started_at,
                run_completed_at,
                test_horizon_days,
                forecast_horizon_days,
                alert_level,
                staff_risk_level,
                admissions_mae,
                icu_mae,
                staff_mae
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                monitoring.get("run_started_at"),
                monitoring.get("run_completed_at"),
                int(monitoring.get("test_horizon_days") or 0),
                int(monitoring.get("forecast_horizon_days") or 0),
                str(monitoring.get("alert_level") or ""),
                str(monitoring.get("staff_risk_level") or ""),
                _extract_mae(admissions),
                _extract_mae(icu),
                _extract_mae(staff),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_recent_runs(limit: int = 50) -> List[Dict[str, Any]]:
    """Return the most recent monitoring runs, newest first."""

    _ensure_initialized()
    path = _db_path()
    conn = sqlite3.connect(path)
    try:
        cur = conn.execute(
            """
            SELECT
                id,
                run_started_at,
                run_completed_at,
                test_horizon_days,
                forecast_horizon_days,
                alert_level,
                staff_risk_level,
                admissions_mae,
                icu_mae,
                staff_mae
            FROM monitoring_runs
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    results: List[Dict[str, Any]] = []
    for r in rows:
        (
            run_id,
            run_started_at,
            run_completed_at,
            test_horizon_days,
            forecast_horizon_days,
            alert_level,
            staff_risk_level,
            admissions_mae,
            icu_mae,
            staff_mae,
        ) = r
        results.append(
            {
                "id": run_id,
                "run_started_at": run_started_at,
                "run_completed_at": run_completed_at,
                "test_horizon_days": test_horizon_days,
                "forecast_horizon_days": forecast_horizon_days,
                "alert_level": alert_level,
                "staff_risk_level": staff_risk_level,
                "admissions_mae": admissions_mae,
                "icu_mae": icu_mae,
                "staff_mae": staff_mae,
            }
        )

    return results
