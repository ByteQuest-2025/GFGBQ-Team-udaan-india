"""Alert utilities for the hospital forecasting pipeline.

This module intentionally keeps alert logic small and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AlertThresholds:
    """Thresholds for ICU utilization-based alerting.

    Values are percentages in the range 0..100.
    """

    yellow_pct: float = 70.0
    red_pct: float = 85.0


def generate_alerts(icu_utilization_pct: float, thresholds: AlertThresholds | None = None) -> str:
    """Return GREEN/YELLOW/RED based on ICU utilization percentage."""

    t = thresholds or AlertThresholds()

    try:
        util = float(icu_utilization_pct)
    except Exception:
        util = 0.0

    if util >= t.red_pct:
        return "RED"
    if util >= t.yellow_pct:
        return "YELLOW"
    return "GREEN"
