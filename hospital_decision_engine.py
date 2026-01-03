"""Hospital decision and alert engine.

This module takes as input the outputs of upstream forecasting models
and operational metrics and produces:
- A high-level alert level (GREEN / YELLOW / RED).
- Actionable, human-readable recommendations.
- A structured JSON-style dictionary for downstream APIs/dashboards.

Inputs (per evaluation time point)
----------------------------------
- predicted_admissions: float
    Forecasted emergency admissions for the next day.
- predicted_icu_demand: float
    Forecasted ICU beds that will be occupied next day.
- icu_capacity: float
    Total ICU bed capacity (or effective staffed beds) for next day.
- staff_risk_level: str
    Output from staff risk module: one of {"LOW", "MEDIUM", "HIGH", "UNKNOWN"}.
- bed_availability: float
    Number of currently available hospital beds (or beds expected to
    be free next day).
- high_respiratory_trend: bool
    Flag indicating high respiratory disease trend (e.g., from
    environmental/health trend analytics).

Rules (simplified, configurable via AlertEngineConfig)
-----------------------------------------------------
- ICU > 85% → RED ALERT
- ICU 70–85% → YELLOW ALERT
- ICU < 70% → GREEN ALERT (unless escalated by other factors)
- Staff burnout HIGH → recommend reserve/backup staff
- High respiratory trend → recommend increased ICU readiness
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Literal, Optional, TypedDict

from hospital_explainability import ExplanationConfig, build_alert_explanations


AlertLevel = Literal["GREEN", "YELLOW", "RED"]
StaffRiskLevel = Literal["LOW", "MEDIUM", "HIGH", "UNKNOWN"]


class AlertResponse(TypedDict, total=False):
    """JSON-serializable response schema for alerts."""

    timestamp: str
    alert_level: AlertLevel
    icu_utilization_pct: float
    predicted_admissions: float
    predicted_icu_demand: float
    icu_capacity: float
    bed_availability: float
    staff_risk_level: StaffRiskLevel
    high_respiratory_trend: bool
    recommendations: List[str]
    explanations: List[str]


@dataclass
class AlertEngineConfig:
    """Configurable thresholds and behavior for the alert engine."""

    icu_yellow_min_pct: float = 70.0
    icu_red_min_pct: float = 85.0
    min_safe_beds: float = 5.0
    escalate_yellow_if_staff_high: bool = True
    escalate_yellow_if_respiratory_high: bool = True


# -------------------------- Core Computations ---------------------------


def compute_icu_utilization_pct(
    predicted_icu_demand: float,
    icu_capacity: float,
) -> float:
    """Compute ICU utilization as a percentage.

    Returns 0 when capacity is non-positive to avoid division errors.
    """

    if icu_capacity <= 0:
        return 0.0
    return float(predicted_icu_demand) / float(icu_capacity) * 100.0


def determine_base_alert_level(
    icu_utilization_pct: float,
    config: AlertEngineConfig,
) -> AlertLevel:
    """Determine base alert level from ICU utilization only."""

    if icu_utilization_pct >= config.icu_red_min_pct:
        return "RED"
    if icu_utilization_pct >= config.icu_yellow_min_pct:
        return "YELLOW"
    return "GREEN"


def refine_alert_level(
    base_level: AlertLevel,
    icu_utilization_pct: float,
    staff_risk_level: StaffRiskLevel,
    high_respiratory_trend: bool,
    bed_availability: float,
    config: AlertEngineConfig,
) -> AlertLevel:
    """Refine alert level considering staff, beds, and respiratory trend.

    Simple rule set for interpretability:
    - Upgrade to RED if ICU is YELLOW, staff HIGH, and beds are low.
    - Upgrade GREEN → YELLOW if respiratory trend is high and
      utilization is near YELLOW threshold.
    """

    level = base_level

    if (
        level == "YELLOW"
        and staff_risk_level == "HIGH"
        and bed_availability <= config.min_safe_beds
    ):
        return "RED"

    if (
        level == "GREEN"
        and high_respiratory_trend
        and icu_utilization_pct >= config.icu_yellow_min_pct * 0.9
    ):
        return "YELLOW"

    return level


def build_recommendations(
    alert_level: AlertLevel,
    icu_utilization_pct: float,
    predicted_admissions: float,
    predicted_icu_demand: float,
    staff_risk_level: StaffRiskLevel,
    bed_availability: float,
    high_respiratory_trend: bool,
    config: Optional[AlertEngineConfig] = None,
) -> List[str]:
    """Generate human-readable recommendations based on rules."""

    if config is None:
        config = AlertEngineConfig()

    recs: List[str] = []

    if alert_level == "RED":
        recs.append(
            "RED ALERT: ICU load is critically high. Prioritize critical cases, "
            "activate surge ICU protocols, and postpone elective procedures."
        )
    elif alert_level == "YELLOW":
        recs.append(
            "YELLOW ALERT: ICU utilization elevated. Prepare surge capacity, "
            "review elective schedule, and closely monitor admissions."
        )
    else:
        recs.append(
            "GREEN ALERT: ICU load within safe range. Maintain baseline operations "
            "but continue routine monitoring of trends."
        )

    if staff_risk_level == "HIGH":
        recs.append(
            "Staff burnout risk HIGH: schedule reserve/backup staff, restrict "
            "non-essential leave, and provide relief coverage where possible."
        )
    elif staff_risk_level == "MEDIUM":
        recs.append(
            "Staff burnout risk MEDIUM: ensure float staff are available and "
            "monitor overtime hours and sickness rates."
        )

    if bed_availability <= config.min_safe_beds:
        recs.append(
            "Bed availability is low: accelerate discharge planning, optimize "
            "step-down bed usage, and coordinate with nearby facilities."
        )

    if high_respiratory_trend:
        recs.append(
            "High respiratory trend detected: increase ICU readiness for "
            "respiratory cases, confirm ventilator availability, and review "
            "oxygen supply and staffing in respiratory wards."
        )

    recs.append(
        f"Forecast summary: {predicted_admissions:.0f} admissions and "
        f"{predicted_icu_demand:.0f} ICU beds expected; ICU utilization "
        f"around {icu_utilization_pct:.1f}% with {bed_availability:.0f} beds free."
    )

    return recs


# --------------------------- Public Interface ---------------------------


def generate_alert(
    *,
    predicted_admissions: float,
    predicted_icu_demand: float,
    icu_capacity: float,
    staff_risk_level: StaffRiskLevel,
    bed_availability: float,
    high_respiratory_trend: bool,
    config: Optional[AlertEngineConfig] = None,
    include_timestamp: bool = True,
    # Optional contextual flags for richer explanations
    is_weekend: Optional[bool] = None,
    is_low_temperature: Optional[bool] = None,
    reduced_staff_availability: Optional[bool] = None,
) -> AlertResponse:
    """Compute alert level, recommendations, and JSON-style response.

    This function is stateless and purely rule-based, making it easy to
    reason about and suitable for clinical validation.
    """

    if config is None:
        config = AlertEngineConfig()

    icu_util_pct = compute_icu_utilization_pct(
        predicted_icu_demand=predicted_icu_demand,
        icu_capacity=icu_capacity,
    )

    base_level = determine_base_alert_level(icu_util_pct, config)
    final_level = refine_alert_level(
        base_level=base_level,
        icu_utilization_pct=icu_util_pct,
        staff_risk_level=staff_risk_level,
        high_respiratory_trend=high_respiratory_trend,
        bed_availability=bed_availability,
        config=config,
    )

    recs = build_recommendations(
        alert_level=final_level,
        icu_utilization_pct=icu_util_pct,
        predicted_admissions=predicted_admissions,
        predicted_icu_demand=predicted_icu_demand,
        staff_risk_level=staff_risk_level,
        bed_availability=bed_availability,
        high_respiratory_trend=high_respiratory_trend,
        config=config,
    )

    # Build human-readable explanations using rule traces and
    # high-level signals. Thresholds are passed explicitly to keep the
    # logic transparent and configurable.
    expl_config = ExplanationConfig(
        icu_yellow_min_pct=config.icu_yellow_min_pct,
        icu_red_min_pct=config.icu_red_min_pct,
        low_bed_threshold=config.min_safe_beds,
    )

    explanations = build_alert_explanations(
        alert_level=final_level,
        icu_utilization_pct=icu_util_pct,
        staff_risk_level=staff_risk_level,
        high_respiratory_trend=high_respiratory_trend,
        bed_availability=bed_availability,
        is_weekend=is_weekend,
        is_low_temperature=is_low_temperature,
        reduced_staff_availability=reduced_staff_availability,
        config=expl_config,
    )

    response: AlertResponse = {
        "alert_level": final_level,
        "icu_utilization_pct": float(icu_util_pct),
        "predicted_admissions": float(predicted_admissions),
        "predicted_icu_demand": float(predicted_icu_demand),
        "icu_capacity": float(icu_capacity),
        "bed_availability": float(bed_availability),
        "staff_risk_level": staff_risk_level,
        "high_respiratory_trend": bool(high_respiratory_trend),
        "recommendations": recs,
        "explanations": explanations,
    }

    if include_timestamp:
        response["timestamp"] = datetime.utcnow().isoformat() + "Z"

    return response


if __name__ == "__main__":
    # Minimal CLI-style example wiring this decision engine.
    # In practice, plug in real predictions from your forecast models.

    cfg = AlertEngineConfig()

    example_response = generate_alert(
        predicted_admissions=120.0,
        predicted_icu_demand=26.0,
        icu_capacity=30.0,
        staff_risk_level="HIGH",
        bed_availability=8.0,
        high_respiratory_trend=True,
        config=cfg,
        include_timestamp=True,
    )

    import json

    print(json.dumps(example_response, indent=2))
