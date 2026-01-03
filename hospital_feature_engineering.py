"""Feature engineering utilities for hospital load prediction.

This module operates on the master dataframe produced by
`hospital_data_pipeline.build_master_dataframe` and constructs a
feature matrix suitable for ML models predicting hospital load.

Focus areas:
- Explainable transformations (lags, rolling means, ratios, flags).
- Clear, configurable column mappings (no hard-coded hospital schema).
- Avoiding obvious temporal leakage by using past information only.

Assumptions
-----------
- Input is a pandas DataFrame with a daily `date` column and numeric
  columns for admissions, ICU utilization, staff counts, weather, and
  health trends. Column names are provided via configuration.
- All data are real hospital data (no synthetic generation here).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd


# ---------------------------- Config Objects ----------------------------


@dataclass
class FeatureConfig:
    """Configuration describing how to engineer features.

    Parameters
    ----------
    date_col:
        Name of the date column (daily granularity).
    admissions_col:
        Column representing daily emergency admissions / total admissions
        used to create lag and rolling features.
    icu_occupied_col:
        Column representing number of ICU beds occupied.
    icu_capacity_col:
        Column representing total ICU bed capacity on that day.
    staff_count_cols:
        Columns representing available staff counts (e.g., doctors,
        nurses, residents) to be summed for staff-to-patient ratio.
    patient_count_col:
        Column representing the denominator for staff-to-patient ratio
        (often the same as admissions_col or current census).
    weather_cols:
        Columns representing environmental / weather variables to
        normalize (e.g., temperature, AQI, humidity).
    health_trend_cols:
        Columns representing community health trend variables to
        normalize (e.g., flu_index, pollution_related_visits).
    holiday_col:
        Optional column indicating public holidays (bool or 0/1). If
        None, holidays will not be inferred and will default to 0.
    weekend_col:
        Optional column indicating weekend days (bool or 0/1). If None,
        this will be derived from the date column.
    forecast_horizon_days:
        How many days ahead to predict. Targets will be shifted
        negatively so that features on day t predict outcomes on
        day t + forecast_horizon_days. Set to 0 to predict same-day.
    lags_admissions:
        Lags (in days) for which to create admission-based features.
    rolling_windows_admissions:
        Window sizes (in days) for rolling mean of admissions.
    """

    date_col: str = "date"
    admissions_col: str = "emergency__admissions"
    icu_occupied_col: str = "icu__beds_occupied"
    icu_capacity_col: str = "icu__beds_capacity"
    staff_count_cols: List[str] = field(default_factory=list)
    patient_count_col: Optional[str] = None
    weather_cols: List[str] = field(default_factory=list)
    health_trend_cols: List[str] = field(default_factory=list)
    holiday_col: Optional[str] = None
    weekend_col: Optional[str] = None
    forecast_horizon_days: int = 0
    lags_admissions: List[int] = field(default_factory=lambda: [1, 7])
    rolling_windows_admissions: List[int] = field(default_factory=lambda: [3, 7])
    admissions_target_col: Optional[str] = None
    icu_demand_target_col: Optional[str] = None


@dataclass
class ColumnScaler:
    """Simple column-wise standardizer (z-score) for selected columns.

    This is intentionally lightweight to avoid external dependencies and
    to keep the transformation explainable.
    """

    means: Dict[str, float] = field(default_factory=dict)
    stds: Dict[str, float] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame, columns: Sequence[str]) -> None:
        """Estimate mean and std for each specified column."""

        for col in columns:
            if col not in df.columns:
                continue
            series = df[col].astype("float64")
            mean = float(series.mean())
            std = float(series.std(ddof=0))  # population std for stability
            self.means[col] = mean
            # Avoid zero std to prevent division errors
            self.stds[col] = std if std > 0 else 1.0

    def transform(self, df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
        """Apply z-score normalization and append `<col>_z` columns.

        Original columns are left unchanged for interpretability.
        """

        out_df = df.copy()
        for col in columns:
            if col not in out_df.columns:
                continue
            if col not in self.means or col not in self.stds:
                # If not fitted for this column, skip transformation.
                continue
            mean = self.means[col]
            std = self.stds[col]
            out_df[f"{col}_z"] = (out_df[col].astype("float64") - mean) / std
        return out_df


# ------------------------- Feature Transformations ----------------------


def _add_admissions_lags(
    df: pd.DataFrame, config: FeatureConfig
) -> pd.DataFrame:
    """Add lag features for admissions.

    Example new columns for lags [1, 7]:
    - `admissions_col` = emergency__admissions
    - `emergency__admissions_lag_1`
    - `emergency__admissions_lag_7`
    """

    working = df.copy()
    src_col = config.admissions_col

    if src_col not in working.columns:
        return working

    for lag in config.lags_admissions:
        col_name = f"{src_col}_lag_{lag}d"
        working[col_name] = working[src_col].shift(lag)

    return working


def _add_admissions_rolling_means(
    df: pd.DataFrame, config: FeatureConfig
) -> pd.DataFrame:
    """Add rolling mean features over admissions.

    Windows are specified in days. Rolling uses past values including
    the current day, aligned with the row index.
    """

    working = df.copy()
    src_col = config.admissions_col

    if src_col not in working.columns:
        return working

    # Ensure data are sorted chronologically
    working = working.sort_values(config.date_col)

    for window in config.rolling_windows_admissions:
        col_name = f"{src_col}_rolling_mean_{window}d"
        working[col_name] = (
            working[src_col]
            .astype("float64")
            .rolling(window=window, min_periods=1)
            .mean()
        )

    return working


def _add_icu_occupancy(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Add ICU occupancy percentage feature.

    - `icu_occupied_col` / `icu_capacity_col` * 100
    - Result stored in `icu_occupancy_pct`.
    """

    working = df.copy()

    occ = config.icu_occupied_col
    cap = config.icu_capacity_col

    if occ not in working.columns or cap not in working.columns:
        return working

    denom = working[cap].replace({0: pd.NA}).astype("float64")
    working["icu_occupancy_pct"] = (
        working[occ].astype("float64") / denom
    ) * 100.0

    return working


def _add_staff_to_patient_ratio(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Add staff-to-patient ratio feature.

    - Sum specified staff columns.
    - Divide by patient_count_col (admissions or census).
    - Result stored in `staff_to_patient_ratio`.
    """

    working = df.copy()

    if not config.staff_count_cols or config.patient_count_col is None:
        return working

    # Only keep staff columns that actually exist
    valid_staff_cols = [c for c in config.staff_count_cols if c in working.columns]
    if not valid_staff_cols or config.patient_count_col not in working.columns:
        return working

    staff_total = working[valid_staff_cols].astype("float64").sum(axis=1)
    denom = working[config.patient_count_col].replace({0: pd.NA}).astype("float64")

    working["staff_to_patient_ratio"] = staff_total / denom

    return working


def _add_seasonal_flags(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Add seasonal flags for winter, monsoon, and summer.

    Season mapping is tailored for India-like climates:
    - Winter:  December–February (12, 1, 2)
    - Summer:  March–May (3, 4, 5)
    - Monsoon: June–September (6, 7, 8, 9)
    Months 10–11 are 0 for all three flags.
    """

    working = df.copy()

    if config.date_col not in working.columns:
        return working

    date_series = pd.to_datetime(working[config.date_col])
    month = date_series.dt.month

    working["is_winter"] = month.isin([12, 1, 2]).astype("int8")
    working["is_summer"] = month.isin([3, 4, 5]).astype("int8")
    working["is_monsoon"] = month.isin([6, 7, 8, 9]).astype("int8")

    return working


def _add_calendar_flags(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Encode holidays and weekends.

    - Weekend is derived from the date column if weekend_col is not
      provided.
    - Holiday uses an explicit column when available; otherwise it
      defaults to 0 (non-holiday) to avoid guessing.
    """

    working = df.copy()

    # Weekend flag
    if config.weekend_col and config.weekend_col in working.columns:
        weekend_raw = pd.to_numeric(working[config.weekend_col], errors="coerce")
        working["is_weekend"] = weekend_raw.fillna(0).astype("int8")
    elif config.date_col in working.columns:
        date_series = pd.to_datetime(working[config.date_col])
        # Saturday (5) and Sunday (6)
        working["is_weekend"] = date_series.dt.weekday.isin([5, 6]).astype("int8")

    # Holiday flag
    if config.holiday_col and config.holiday_col in working.columns:
        holiday_raw = pd.to_numeric(working[config.holiday_col], errors="coerce")
        working["is_holiday"] = holiday_raw.fillna(0).astype("int8")
    else:
        # If no explicit holiday information, default to 0.
        working["is_holiday"] = 0

    return working


def _normalize_groups(
    df: pd.DataFrame,
    config: FeatureConfig,
    scaler: Optional[ColumnScaler] = None,
    fit_scaler: bool = False,
) -> Tuple[pd.DataFrame, ColumnScaler]:
    """Normalize weather and health trend features using z-scores.

    - Uses ColumnScaler to store means/stds for reproducibility.
    - Appends `<col>_z` columns; original columns remain unchanged.
    - When `fit_scaler` is True, fits on the provided dataframe; this
      should typically be called on the training split only.
    """

    working = df.copy()

    if scaler is None:
        scaler = ColumnScaler()

    cols_to_scale: List[str] = []
    cols_to_scale.extend(config.weather_cols)
    cols_to_scale.extend(config.health_trend_cols)

    # Keep only existing columns
    cols_to_scale = [c for c in cols_to_scale if c in working.columns]

    if fit_scaler and cols_to_scale:
        scaler.fit(working, cols_to_scale)

    if cols_to_scale:
        working = scaler.transform(working, cols_to_scale)

    return working, scaler


# -------------------------- Public API Function -------------------------


def engineer_features(
    master_df: pd.DataFrame,
    config: FeatureConfig,
    scaler: Optional[ColumnScaler] = None,
    fit_scaler: bool = False,
    target_mode: str = "both",
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, ColumnScaler]:
    """Create engineered feature matrix and targets for ML.

    Parameters
    ----------
    master_df:
        Merged dataframe from `hospital_data_pipeline.build_master_dataframe`.
    config:
        FeatureConfig describing column mappings and parameters.
    scaler:
        Optional pre-fitted ColumnScaler (useful when transforming
        validation/test sets without refitting).
    fit_scaler:
        If True, fit scaler on `master_df` before transforming. In a
        real ML workflow, this should only be True on the training set.

    Parameters
    ----------
    target_mode:
        Controls how rows are filtered for missing targets:
        - "both": require both admissions and ICU targets to be present (default)
        - "admissions": require admissions target only
        - "icu": require ICU target only
        - "none": do not require either target (useful for feature-only workflows)

    Returns
    -------
    X:
        Final feature matrix (numeric dataframe) ready for ML.
    y_admissions:
        Target series for admissions (e.g., emergency load).
    y_icu_demand:
        Target series for ICU demand.
    scaler:
        ColumnScaler containing fitted means/stds for reproducibility.
    """

    if config.admissions_target_col is None:
        config.admissions_target_col = config.admissions_col
    if config.icu_demand_target_col is None:
        config.icu_demand_target_col = config.icu_occupied_col
    if config.patient_count_col is None:
        config.patient_count_col = config.admissions_col

    df = master_df.copy()

    # Sort by date to ensure temporal consistency
    if config.date_col in df.columns:
        df = df.sort_values(config.date_col).reset_index(drop=True)

    # Core engineered features
    df = _add_admissions_lags(df, config)
    df = _add_admissions_rolling_means(df, config)
    df = _add_icu_occupancy(df, config)
    df = _add_staff_to_patient_ratio(df, config)
    df = _add_seasonal_flags(df, config)
    df = _add_calendar_flags(df, config)

    # Normalize weather and health trend features
    df, scaler = _normalize_groups(df, config, scaler=scaler, fit_scaler=fit_scaler)

    # Build targets (optionally shifted forward for forecasting)
    y_adm = df[config.admissions_target_col].astype("float64")
    y_icu = df[config.icu_demand_target_col].astype("float64")

    if config.forecast_horizon_days > 0:
        shift = -int(config.forecast_horizon_days)
        y_adm = y_adm.shift(shift)
        y_icu = y_icu.shift(shift)

    # Align features and targets: select rows based on required targets.
    mode = str(target_mode).lower().strip()
    if mode == "both":
        valid_mask = y_adm.notna() & y_icu.notna()
    elif mode == "admissions":
        valid_mask = y_adm.notna()
    elif mode == "icu":
        valid_mask = y_icu.notna()
    elif mode == "none":
        valid_mask = pd.Series(True, index=df.index)
    else:
        raise ValueError(
            "target_mode must be one of {'both','admissions','icu','none'}, "
            f"got: {target_mode!r}"
        )

    df_valid = df.loc[valid_mask].copy()
    y_adm_valid = y_adm.loc[valid_mask]
    y_icu_valid = y_icu.loc[valid_mask]

    # Drop non-feature columns and keep numeric types only
    drop_cols: List[str] = []
    if config.date_col in df_valid.columns:
        drop_cols.append(config.date_col)

    # Keep target columns out of X to avoid trivial leakage
    for target_col in {config.admissions_target_col, config.icu_demand_target_col}:
        if target_col in df_valid.columns:
            drop_cols.append(target_col)

    X = df_valid.drop(columns=drop_cols, errors="ignore")
    X = X.select_dtypes(include=["number"]).copy()

    return X, y_adm_valid, y_icu_valid, scaler


if __name__ == "__main__":
    # Minimal example showing how to wire this module with
    # `hospital_data_pipeline` in a script. In practice, you would
    # perform train/validation/test splits and only fit the scaler on
    # the training data.
    from pathlib import Path

    from hospital_data_pipeline import example_build_master_from_default_files

    base_dir = Path(".")
    master_df, _ = example_build_master_from_default_files(base_dir)

    # Example configuration — adjust column names to match your schema.
    cfg = FeatureConfig(
        date_col="date",
        admissions_col="emergency__admissions",  # update if different
        icu_occupied_col="icu__beds_occupied",   # update if different
        icu_capacity_col="icu__beds_capacity",   # update if different
        staff_count_cols=[
            "staff__doctors_on_duty",
            "staff__nurses_on_duty",
        ],
        patient_count_col="emergency__admissions",  # or census
        weather_cols=[
            "env_health__temperature_c",
            "env_health__aqi",
        ],
        health_trend_cols=[
            "env_health__flu_index",
        ],
        holiday_col="calendar__is_holiday",
        weekend_col="calendar__is_weekend",
        forecast_horizon_days=1,
    )

    X, y_adm, y_icu, fitted_scaler = engineer_features(
        master_df=master_df,
        config=cfg,
        scaler=None,
        fit_scaler=True,
    )

    print("Feature matrix shape:", X.shape)
    print("Admissions target length:", len(y_adm))
    print("ICU demand target length:", len(y_icu))
