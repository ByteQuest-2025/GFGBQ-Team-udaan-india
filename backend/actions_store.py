from __future__ import annotations

"""SQLite-backed storage for operational actions triggered from the UI.

This provides a simple audit trail for things like staffing requests,
overflow activations, and notification events.
"""

from pathlib import Path
from typing import Any, Dict, List
import json
import sqlite3
from datetime import datetime

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
            CREATE TABLE IF NOT EXISTS actions_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                action_type TEXT NOT NULL,
                source TEXT NOT NULL,
                payload TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()

    _INITIALIZED = True


def log_action(action_type: str, source: str, payload: Dict[str, Any] | None = None) -> int:
    """Persist a single UI-triggered action and return its ID."""

    _ensure_initialized()
    path = _db_path()
    conn = sqlite3.connect(path)
    try:
        created_at = datetime.utcnow().isoformat() + "Z"
        payload_json = json.dumps(payload or {}, ensure_ascii=False)
        cur = conn.execute(
            """
            INSERT INTO actions_log (created_at, action_type, source, payload)
            VALUES (?, ?, ?, ?)
            """,
            (created_at, action_type, source, payload_json),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def get_recent_actions(limit: int = 50) -> List[Dict[str, Any]]:
    """Return recent actions, newest first."""

    _ensure_initialized()
    path = _db_path()
    conn = sqlite3.connect(path)
    try:
        cur = conn.execute(
            """
            SELECT id, created_at, action_type, source, payload
            FROM actions_log
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    results: List[Dict[str, Any]] = []
    for row in rows:
        row_id, created_at, action_type, source, payload_json = row
        try:
            payload = json.loads(payload_json) if payload_json else {}
        except Exception:
            payload = {"_raw": payload_json}
        results.append(
            {
                "id": row_id,
                "created_at": created_at,
                "action_type": action_type,
                "source": source,
                "payload": payload,
            }
        )

    return results
