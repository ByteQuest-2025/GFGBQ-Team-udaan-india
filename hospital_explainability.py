"""Explainability utilities for hospital alerts and forecasts.

This module focuses on generating human-readable explanations for
alerts produced by the hospital decision engine. Rather than using
opaque black-box explanations, it relies on:

- Transparent rule traces (why a rule fired).
- High-level feature signals already available to the decision layer
  (e.g., respiratory trends, weekend/holiday flags, staff risk).

The goal is to provide concise, clinician-friendly narratives such as:

    "High ICU demand predicted due to:
     - Rising respiratory cases
     - Low temperature
     - Weekend effect
     - Reduced staff availability"

The functions here are intentionally lightweight and do not depend on
SHAP or other heavy XAI libraries, keeping the explanations easy to
validate with clinical stakeholders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional


AlertLevel = Literal["GREEN", "YELLOW", "RED"]
StaffRiskLevel = Literal["LOW", "MEDIUM", "HIGH", "UNKNOWN"]


@dataclass
class ExplanationConfig:
    """Configuration for mapping signals to explanation phrases.

    Thresholds should align with those used in the alert engine, but
    they are passed in explicitly from the caller to avoid tight
    coupling between modules.
    """

    icu_yellow_min_pct: float = 70.0
    icu_red_min_pct: float = 85.0
    low_bed_threshold: float = 5.0


def build_alert_explanations(
    *,
    alert_level: AlertLevel,
    icu_utilization_pct: float,
    staff_risk_level: StaffRiskLevel,
    high_respiratory_trend: bool,
    bed_availability: float,
    is_weekend: Optional[bool] = None,
    is_low_temperature: Optional[bool] = None,
    reduced_staff_availability: Optional[bool] = None,
    config: Optional[ExplanationConfig] = None,
) -> List[str]:
    """Generate a list of explanation sentences for an alert.

    Parameters
    ----------
    alert_level:
        Final alert level from the decision engine.
    icu_utilization_pct:
        Forecasted ICU utilization percentage for the decision period.
    staff_risk_level:
        Staff workload/burnout risk level.
    high_respiratory_trend:
        Whether respiratory/related health trends are currently high.
    bed_availability:
        Number of beds expected to be free.
    is_weekend:
        Optional flag indicating if the decision period falls on a
        weekend; if None, no weekend-related explanation is added.
    is_low_temperature:
        Optional flag derived from environmental data indicating low
        ambient temperature; if None, temperature is not mentioned.
    reduced_staff_availability:
        Optional explicit flag for reduced staff availability (e.g.,
        high leave rates); complements staff_risk_level.
    config:
        Optional ExplanationConfig; if None, defaults are used.
    """

    if config is None:
        config = ExplanationConfig()

    reasons: List[str] = []

    # ICU load contributions
    if icu_utilization_pct >= config.icu_red_min_pct:
        reasons.append(
            "ICU occupancy is projected to exceed the critical safety threshold "
            f"of about {config.icu_red_min_pct:.0f}% (forecast ~{icu_utilization_pct:.1f}%)."
        )
    elif icu_utilization_pct >= config.icu_yellow_min_pct:
        reasons.append(
            "ICU occupancy is forecast to be in the elevated range "
            f"around {icu_utilization_pct:.1f}%, above the usual comfort zone."
        )

    # Respiratory / seasonal drivers
    if high_respiratory_trend:
        reasons.append(
            "Recent respiratory or infection-related indicators are rising, which "
            "historically increases ICU admissions."
        )

    if is_low_temperature is True:
        reasons.append(
            "Low ambient temperature is expected, which is associated with higher "
            "rates of respiratory and cardiac presentations."
        )

    # Calendar / weekend effects
    if is_weekend is True:
        reasons.append(
            "The forecast period falls on a weekend, when emergency demand and "
            "admission patterns typically differ from weekdays."
        )

    # Staffing contributions
    if staff_risk_level == "HIGH" or reduced_staff_availability:
        reasons.append(
            "Staffing indicators suggest high workload and reduced available "
            "coverage, increasing the risk of burnout and bottlenecks."
        )
    elif staff_risk_level == "MEDIUM":
        reasons.append(
            "Staff workload is moderate and may limit flexibility if demand "
            "rises further."
        )

    # Bed capacity contributions
    if bed_availability <= config.low_bed_threshold:
        reasons.append(
            "Overall bed availability is low, leaving limited buffer to absorb "
            "unexpected surges in admissions."
        )

    # Fallback if nothing specific triggered
    if not reasons:
        if alert_level == "GREEN":
            reasons.append(
                "Key indicators (ICU load, respiratory trends, staffing, beds) are "
                "all within typical ranges for this period."
            )
        else:
            reasons.append(
                "Alert level is elevated based on combined patterns in ICU load, "
                "demand forecasts, and staffing, even though no single driver "
                "dominates."
            )

    # Compose with a leading summary sentence
    if alert_level == "RED":
        header = "High ICU demand and operational stress are expected due to:"  # noqa: E501
    elif alert_level == "YELLOW":
        header = "Elevated ICU demand and staffing pressure are expected due to:"
    else:
        header = "Conditions are currently stable, but should be monitored because:"

    return [header] + reasons
