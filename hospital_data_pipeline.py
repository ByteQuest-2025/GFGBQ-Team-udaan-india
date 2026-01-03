"""Hospital data loading and validation pipeline.

This module loads and validates multiple hospital-related CSV datasets,
parses date/time columns, checks data quality, and aligns everything on a
common daily date key to produce a master dataframe ready for ML.

Datasets expected (filenames can be customized by the caller):
- emergency_admissions.csv
- icu_bed_utilization.csv
- staff_availability.csv
- calendar_seasonal.csv
- environmental_health_trends.csv

The code is written to be:
- Modular and reusable
- Explainable (clear transformations, no hidden feature engineering)
- Robust to minor schema differences (e.g., different date column names)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd


# ----------------------------- Data Classes -----------------------------

@dataclass
class DatasetConfig:
    """Configuration for a single dataset.

    Attributes
    ----------
    name:
        Logical name of the dataset (used for prefixes in merged dataframe).
    path:
        Filesystem path to the CSV file.
    date_columns:
        Optional list of column names that should be parsed as dates. If not
        provided, the pipeline will attempt to infer date-like columns.
    preferred_date_column:
        Optional name of the primary date column to use as the alignment key.
        If not provided, the pipeline will try to infer a single date column.
    """

    name: str
    path: Path
    date_columns: Optional[List[str]] = None
    preferred_date_column: Optional[str] = None


# ---------------------- Auto-detection + Canonicalization ----------------------


def _list_csv_files(base_dir: Path) -> List[Path]:
    return sorted([p for p in base_dir.glob("*.csv") if p.is_file()])


def _choose_by_keywords(files: List[Path], keywords: List[str]) -> Optional[Path]:
    """Choose the best-matching CSV by filename keywords.

    This is intentionally simple and transparent: score by how many keywords
    appear in the lowercase filename.
    """

    if not files:
        return None

    scored: List[Tuple[int, Path]] = []
    for p in files:
        name = p.name.lower()
        # Tokenize filename to avoid accidental matches like 'er' in 'services'
        tokens = [t for t in ''.join(ch if ch.isalnum() else ' ' for ch in name).split() if t]
        token_set = set(tokens)

        score = 0
        for kw in keywords:
            kw_l = kw.lower()
            if len(kw_l) <= 2:
                # Short keywords must match as a token (e.g., 'er')
                if kw_l in token_set:
                    score += 1
            else:
                if kw_l in name or kw_l in token_set:
                    score += 1
        scored.append((score, p))

    scored.sort(key=lambda t: (t[0], t[1].name), reverse=True)
    best_score, best_path = scored[0]
    if best_score <= 0:
        return None
    return best_path


def _read_csv_flexible(path: Path) -> pd.DataFrame:
    """Read CSV with mild delimiter flexibility.

    Some of the provided datasets use ';' as delimiter (e.g., hosp.csv). We
    try default ',' first and fall back to ';' if it looks like a single
    column file.
    """

    df = pd.read_csv(path)
    if df.shape[1] == 1:
        df2 = pd.read_csv(path, sep=";", engine="python")
        if df2.shape[1] > 1:
            return df2
    return df


def _week1_monday(reference_date: pd.Timestamp) -> pd.Timestamp:
    ref = pd.to_datetime(reference_date).normalize()
    return ref - pd.Timedelta(days=int(ref.weekday()))


def _expand_week_to_daily(df_week: pd.DataFrame, week_start_col: str = "week_start") -> pd.DataFrame:
    """Expand a week-level dataframe into daily rows (Mon..Sun).

    Each week-level row becomes 7 daily rows with the same values.
    """

    rows: List[pd.DataFrame] = []
    for _, r in df_week.iterrows():
        start = pd.to_datetime(r[week_start_col]).normalize()
        dates = pd.date_range(start=start, periods=7, freq="D")
        rep = pd.DataFrame({"date": dates})
        for col in df_week.columns:
            if col == week_start_col:
                continue
            rep[col] = r[col]
        rows.append(rep)
    if not rows:
        return pd.DataFrame(columns=["date"])
    return pd.concat(rows, ignore_index=True)


def _canonical_emergency_from_er_data(path: Path) -> pd.DataFrame:
    """Build daily emergency admissions from a patient-level ER dataset."""

    df = _read_csv_flexible(path)

    # Common variants encountered in provided files
    date_col_candidates = [
        "Patient Admission Date",
        "Visit Date",
        "arrival_date",
        "admission_date",
        "date",
    ]
    date_col = next((c for c in date_col_candidates if c in df.columns), None)
    if date_col is None:
        raise ValueError(
            f"Unable to infer an admissions date column in {path.name}. "
            f"Columns: {list(df.columns)[:30]}"
        )

    # Many ER exports use dd-mm-YYYY; dayfirst=True avoids ambiguous parsing.
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    df = df.dropna(subset=[date_col]).copy()
    df["date"] = df[date_col].dt.normalize()

    out = df.groupby("date").size().rename("admissions").reset_index()

    # Optional: include mean wait time if present
    wait_candidates = ["Patient Waittime", "Total Wait Time (min)"]
    wait_col = next((c for c in wait_candidates if c in df.columns), None)
    if wait_col is not None:
        df[wait_col] = pd.to_numeric(df[wait_col], errors="coerce")
        wait_mean = df.groupby("date")[wait_col].mean().rename("avg_wait_minutes")
        out = out.merge(wait_mean.reset_index(), on="date", how="left")

    return out.sort_values("date").reset_index(drop=True)


def _canonical_icu_from_patients_and_services(
    patients_path: Optional[Path],
    services_weekly_path: Optional[Path],
    week1_monday: Optional[pd.Timestamp],
) -> pd.DataFrame:
    """Build daily ICU occupancy and capacity.

    - Occupancy: computed from `patients.csv` ICU stays (arrival_date..departure_date).
    - Capacity: taken from `services_weekly.csv` where service == 'ICU' and
      expanded to daily. If weekly data is unavailable, capacity is left as NA.
    """

    daily_occupancy: Optional[pd.DataFrame] = None
    if patients_path is not None and patients_path.exists():
        p = _read_csv_flexible(patients_path)
        if "service" in p.columns and "arrival_date" in p.columns and "departure_date" in p.columns:
            p = p[p["service"].astype(str).str.lower() == "icu"].copy()
            p["arrival_date"] = pd.to_datetime(p["arrival_date"], errors="coerce")
            p["departure_date"] = pd.to_datetime(p["departure_date"], errors="coerce")
            p = p.dropna(subset=["arrival_date", "departure_date"]).copy()
            if not p.empty:
                start = p["arrival_date"].min().normalize()
                end = p["departure_date"].max().normalize()
                all_days = pd.date_range(start=start, end=end, freq="D")
                occ = []
                # Straightforward loop for clarity (datasets are typically moderate size)
                for day in all_days:
                    active = (p["arrival_date"].dt.normalize() <= day) & (p["departure_date"].dt.normalize() > day)
                    occ.append(int(active.sum()))
                daily_occupancy = pd.DataFrame({"date": all_days, "beds_occupied": occ})

    # Prefer anchoring weekly capacity expansion to the ICU occupancy period
    # (if available) so that capacity aligns with occupancy dates.
    capacity_anchor_monday = week1_monday
    if daily_occupancy is not None and not daily_occupancy.empty:
        capacity_anchor_monday = _week1_monday(daily_occupancy["date"].min())

    daily_capacity: Optional[pd.DataFrame] = None
    if services_weekly_path is not None and services_weekly_path.exists() and capacity_anchor_monday is not None:
        s = _read_csv_flexible(services_weekly_path)
        if "service" in s.columns and "week" in s.columns and "available_beds" in s.columns:
            icu = s[s["service"].astype(str).str.lower() == "icu"].copy()
            icu["week"] = pd.to_numeric(icu["week"], errors="coerce")
            icu = icu.dropna(subset=["week"]).copy()
            icu["week_start"] = capacity_anchor_monday + pd.to_timedelta((icu["week"].astype(int) - 1) * 7, unit="D")
            icu["beds_capacity"] = pd.to_numeric(icu["available_beds"], errors="coerce")
            icu_week = icu[["week_start", "beds_capacity"]].groupby("week_start", as_index=False).mean()
            daily_capacity = _expand_week_to_daily(icu_week, week_start_col="week_start")

    if daily_occupancy is None and daily_capacity is None:
        return pd.DataFrame(columns=["date", "beds_occupied", "beds_capacity"])
    if daily_occupancy is None:
        return daily_capacity.sort_values("date").reset_index(drop=True)
    if daily_capacity is None:
        return daily_occupancy.sort_values("date").reset_index(drop=True)

    out = daily_occupancy.merge(daily_capacity, on="date", how="outer").sort_values("date")
    return out.reset_index(drop=True)


def _canonical_staff_from_schedule(
    staff_schedule_path: Optional[Path],
    week1_monday: Optional[pd.Timestamp],
) -> pd.DataFrame:
    """Build daily staff availability counts (doctors/nurses) from weekly schedules."""

    if staff_schedule_path is None or not staff_schedule_path.exists() or week1_monday is None:
        return pd.DataFrame(columns=["date", "doctors_on_duty", "nurses_on_duty"])

    ss = _read_csv_flexible(staff_schedule_path)
    needed = {"week", "role", "service", "present"}
    if not needed.issubset(set(ss.columns)):
        return pd.DataFrame(columns=["date", "doctors_on_duty", "nurses_on_duty"])

    ss = ss.copy()
    ss["week"] = pd.to_numeric(ss["week"], errors="coerce")
    ss["present"] = pd.to_numeric(ss["present"], errors="coerce").fillna(0)
    ss = ss.dropna(subset=["week"]).copy()

    # Focus on emergency staffing; adjust here if you want ICU staffing instead.
    ss = ss[ss["service"].astype(str).str.lower() == "emergency"].copy()
    ss["week_start"] = week1_monday + pd.to_timedelta((ss["week"].astype(int) - 1) * 7, unit="D")

    # Aggregate weekly counts by role (count of present staff)
    ss["role"] = ss["role"].astype(str).str.lower()
    doctors = ss[ss["role"].eq("doctor")].groupby("week_start")["present"].sum().rename("doctors_on_duty")
    nurses = ss[ss["role"].eq("nurse")].groupby("week_start")["present"].sum().rename("nurses_on_duty")
    weekly = pd.concat([doctors, nurses], axis=1).fillna(0).reset_index()

    daily = _expand_week_to_daily(weekly, week_start_col="week_start")
    return daily.sort_values("date").reset_index(drop=True)


def _canonical_calendar_from_holidays(
    holidays_path: Optional[Path],
    date_index: pd.DatetimeIndex,
    preferred_country_codes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Build calendar features: weekend + holiday flags.

    If a holiday dataset exists, we filter by countryRegionCode. If no preferred
    codes are found, holidays default to 0 (to avoid guessing).
    """

    if preferred_country_codes is None:
        preferred_country_codes = ["IN", "IND", "US", "GB"]

    cal = pd.DataFrame({"date": pd.to_datetime(date_index).normalize()})
    cal["is_weekend"] = cal["date"].dt.weekday.isin([5, 6]).astype("int8")
    cal["is_holiday"] = 0

    if holidays_path is None or not holidays_path.exists():
        return cal

    h = _read_csv_flexible(holidays_path)
    if "date" not in h.columns:
        return cal

    h["date"] = pd.to_datetime(h["date"], errors="coerce").dt.normalize()
    h = h.dropna(subset=["date"]).copy()

    if "countryRegionCode" in h.columns:
        available_codes = set(h["countryRegionCode"].astype(str).unique())
        chosen = next((c for c in preferred_country_codes if c in available_codes), None)
        if chosen is not None:
            h = h[h["countryRegionCode"].astype(str) == chosen].copy()
        else:
            # No suitable country code found; keep holidays disabled (transparent default)
            return cal

    holiday_dates = set(h["date"].unique())
    cal["is_holiday"] = cal["date"].isin(holiday_dates).astype("int8")
    return cal


def _canonical_env_health_from_influenza(
    seasonal_influenza_path: Optional[Path],
    influenza_weekly_path: Optional[Path],
) -> pd.DataFrame:
    """Build a daily health-trend signal from influenza datasets.

    Output column `flu_index` is a simple, explainable proxy:
    - If Seasonal influenza is available: use `Percent_Positive` for Total_Influenza,
      keyed by `weekending`.
    - Else if influenza_weekly is available: use `ALL_INF` keyed by `EDATE`.
    The weekly signal is forward-filled to daily.
    """

    weekly: Optional[pd.DataFrame] = None

    if seasonal_influenza_path is not None and seasonal_influenza_path.exists():
        s = _read_csv_flexible(seasonal_influenza_path)
        if {"weekending", "Respiratory_Virus"}.issubset(set(s.columns)):
            s = s.copy()
            s["weekending"] = pd.to_datetime(s["weekending"], errors="coerce")
            s = s.dropna(subset=["weekending"]).copy()
            # Prefer Total_Influenza if present
            sv = s[s["Respiratory_Virus"].astype(str).str.lower().str.contains("total")].copy()
            if sv.empty:
                sv = s
            value_col = "Percent_Positive" if "Percent_Positive" in sv.columns else None
            if value_col is not None:
                sv[value_col] = pd.to_numeric(sv[value_col], errors="coerce")
                weekly = (
                    sv.groupby(sv["weekending"].dt.normalize())[value_col]
                    .mean()
                    .rename("flu_index")
                    .reset_index()
                    .rename(columns={"weekending": "date"})
                )

    if weekly is None and influenza_weekly_path is not None and influenza_weekly_path.exists():
        w = _read_csv_flexible(influenza_weekly_path)
        if "EDATE" in w.columns and "ALL_INF" in w.columns:
            w = w.copy()
            w["EDATE"] = pd.to_datetime(w["EDATE"], errors="coerce")
            w = w.dropna(subset=["EDATE"]).copy()
            w["ALL_INF"] = pd.to_numeric(w["ALL_INF"], errors="coerce")
            weekly = (
                w.groupby(w["EDATE"].dt.normalize())["ALL_INF"]
                .mean()
                .rename("flu_index")
                .reset_index()
                .rename(columns={"EDATE": "date"})
            )

    if weekly is None or weekly.empty:
        return pd.DataFrame(columns=["date", "flu_index"])

    weekly = weekly.sort_values("date").reset_index(drop=True)
    # Forward-fill weekly signal to daily
    date_range = pd.date_range(start=weekly["date"].min(), end=weekly["date"].max(), freq="D")
    daily = pd.DataFrame({"date": date_range})
    daily = daily.merge(weekly, on="date", how="left").sort_values("date")
    daily["flu_index"] = daily["flu_index"].ffill()
    return daily.reset_index(drop=True)


# -------------------------- Helper Functions ---------------------------


def _infer_date_columns(df: pd.DataFrame) -> List[str]:
    """Infer likely date/time columns based on column names and dtypes.

    This function is conservative: it only returns columns that either
    already have a datetime dtype or whose names clearly indicate a
    temporal meaning.
    """

    # Columns already parsed as datetime
    datetime_cols = [
        c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])
    ]

    # Name-based heuristics for potential date columns
    name_based_candidates: List[str] = []
    date_keywords = ["date", "day", "time", "timestamp"]
    for col in df.columns:
        lower = str(col).lower()
        if any(kw in lower for kw in date_keywords):
            name_based_candidates.append(col)

    # Combine and de-duplicate while preserving order
    seen = set()
    result: List[str] = []
    for col in list(datetime_cols) + name_based_candidates:
        if col not in seen:
            seen.add(col)
            result.append(col)

    return result


def _parse_dates(df: pd.DataFrame, date_cols: Optional[List[str]]) -> pd.DataFrame:
    """Ensure specified or inferred columns are converted to datetime.

    Parameters
    ----------
    df:
        Raw dataframe loaded from CSV.
    date_cols:
        Explicit list of date columns or None to infer.
    """

    working_df = df.copy()

    # If no explicit list is provided, infer likely date columns
    if date_cols is None:
        date_cols = _infer_date_columns(working_df)

    for col in date_cols:
        if col not in working_df.columns:
            continue
        # Use pandas to_datetime with errors="coerce" to avoid hard failures
        working_df[col] = pd.to_datetime(working_df[col], errors="coerce")

    return working_df


def _choose_primary_date_column(
    df: pd.DataFrame, preferred: Optional[str]
) -> str:
    """Select a single date column to act as the alignment key.

    Strategy
    --------
    - If `preferred` is provided and exists (after parsing), use it.
    - Otherwise, choose the first datetime64 column.
    - If none exist, raise a clear error so the user can adjust config.
    """

    if preferred is not None:
        if preferred not in df.columns:
            raise ValueError(f"Preferred date column '{preferred}' not found in columns: {list(df.columns)}")
        return preferred

    datetime_cols = [
        c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])
    ]
    if not datetime_cols:
        raise ValueError(
            "No datetime-like columns found. "
            "Please provide `date_columns` and `preferred_date_column` in DatasetConfig."
        )

    return datetime_cols[0]


def _standardize_date_column(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Standardize the primary date column to a daily `date` column.

    - Truncates timestamps to date (no time-of-day) using `.dt.normalize()`.
    - Renames the column to a canonical name `date` for merging.
    """

    working_df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(working_df[date_col]):
        working_df[date_col] = pd.to_datetime(working_df[date_col], errors="coerce")

    working_df["date"] = working_df[date_col].dt.normalize()

    if "date" != date_col:
        working_df = working_df.drop(columns=[date_col])

    return working_df


def _summarize_dataframe(name: str, df: pd.DataFrame) -> Dict[str, object]:
    """Produce basic summary statistics and missing-value profile.

    This function both returns a structured summary (for programmatic use)
    and can be used to print human-readable diagnostics in a notebook or
    script.
    """

    summary: Dict[str, object] = {
        "name": name,
        "num_rows": int(df.shape[0]),
        "num_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values_per_column": df.isna().sum().to_dict(),
        # describe(include="all") can be large; use numeric only for core stats
        "numeric_summary": df.describe(include=["number"]).to_dict(),
    }

    return summary


def load_dataset(config: DatasetConfig) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Load a single dataset according to its configuration.

    Steps
    -----
    1. Read CSV using pandas.
    2. Parse or infer date columns.
    3. Choose primary date column and standardize to daily `date`.
    4. Generate summary statistics.

    Returns
    -------
    df:
        Cleaned dataframe with a standardized `date` column.
    summary:
        Dictionary with basic statistics and missing-value information.
    """

    if not config.path.exists():
        raise FileNotFoundError(f"CSV file not found: {config.path}")

    raw_df = pd.read_csv(config.path)

    # Parse explicit or inferred date columns
    parsed_df = _parse_dates(raw_df, config.date_columns)

    # Choose and standardize primary date column
    primary_date_col = _choose_primary_date_column(parsed_df, config.preferred_date_column)
    std_df = _standardize_date_column(parsed_df, primary_date_col)

    # Build summary for diagnostics
    summary = _summarize_dataframe(config.name, std_df)

    return std_df, summary


def _prefix_non_key_columns(df: pd.DataFrame, prefix: str, key_columns: List[str]) -> pd.DataFrame:
    """Prefix non-key columns to avoid collisions during merge.

    Example: columns ["date", "beds_available", "beds_occupied"] with
    prefix "icu" and key ["date"] will become
    ["date", "icu__beds_available", "icu__beds_occupied"].
    """

    working_df = df.copy()
    non_keys = [c for c in working_df.columns if c not in key_columns]

    if non_keys:
        rename_map = {c: f"{prefix}__{c}" for c in non_keys}
        working_df = working_df.rename(columns=rename_map)

    return working_df


def build_master_dataframe(
    configs: List[DatasetConfig],
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, object]]]:
    """Load, validate, and merge multiple datasets on daily date.

    Parameters
    ----------
    configs:
        List of DatasetConfig objects describing each dataset.

    Returns
    -------
    master_df:
        Merged dataframe with one row per date (outer join across all
        datasets) and prefixed feature columns.
    summaries:
        Mapping from dataset name to its summary dictionary.
    """

    if not configs:
        raise ValueError("No dataset configurations provided.")

    cleaned_dfs: List[pd.DataFrame] = []
    summaries: Dict[str, Dict[str, object]] = {}

    for cfg in configs:
        df, summary = load_dataset(cfg)
        # Prefix non-key columns to keep dataset origin explicit
        df_prefixed = _prefix_non_key_columns(df, cfg.name, key_columns=["date"])
        cleaned_dfs.append(df_prefixed)
        summaries[cfg.name] = summary

    # Outer join across all dates to avoid dropping any day
    master_df = cleaned_dfs[0]
    for next_df in cleaned_dfs[1:]:
        master_df = master_df.merge(next_df, on="date", how="outer", sort=True)

    # Sort by date for chronological coherence
    master_df = master_df.sort_values("date").reset_index(drop=True)

    return master_df, summaries


# --------------------------- Example Usage -----------------------------


def example_build_master_from_default_files(base_dir: Path) -> Tuple[pd.DataFrame, Dict[str, Dict[str, object]]]:
    """Convenience function to build master dataframe from default CSV names.

    Adjust `preferred_date_column` or `date_columns` below if your schema
    uses different column names for dates/timestamps.
    """

    # If the canonical filenames exist, prefer them.
    canonical_paths = {
        "emergency": base_dir / "emergency_admissions.csv",
        "icu": base_dir / "icu_bed_utilization.csv",
        "staff": base_dir / "staff_availability.csv",
        "calendar": base_dir / "calendar_seasonal.csv",
        "env_health": base_dir / "environmental_health_trends.csv",
    }
    if all(p.exists() for p in canonical_paths.values()):
        configs = [
            DatasetConfig(name=k, path=v) for k, v in canonical_paths.items()
        ]
        master_df, summaries = build_master_dataframe(configs)
        return master_df, summaries

    # Otherwise, auto-detect and canonicalize from the CSVs present.
    csvs = _list_csv_files(base_dir)

    # Heuristic mapping based on filenames present in the workspace.
    emergency_path = _choose_by_keywords(csvs, ["er", "emergency", "admission"]) or _choose_by_keywords(csvs, ["hospital", "er_data"])
    patients_path = _choose_by_keywords(csvs, ["patients"])  # used for ICU occupancy
    services_path = _choose_by_keywords(csvs, ["services", "weekly"])  # used for ICU capacity
    staff_schedule_path = _choose_by_keywords(csvs, ["staff", "schedule"]) or _choose_by_keywords(csvs, ["staff"])
    holidays_path = _choose_by_keywords(csvs, ["holiday"]) or _choose_by_keywords(csvs, ["calendar"])  # optional
    seasonal_influenza_path = _choose_by_keywords(csvs, ["seasonal", "influenza"])  # optional
    influenza_weekly_path = _choose_by_keywords(csvs, ["influenza", "weekly"])  # optional

    if emergency_path is None:
        raise FileNotFoundError(
            "Could not auto-detect an emergency admissions file. "
            "Expected a CSV with keywords like 'ER', 'emergency', or 'admission' in its name."
        )

    emergency_daily = _canonical_emergency_from_er_data(emergency_path)
    ref_monday = _week1_monday(emergency_daily["date"].min())

    icu_daily = _canonical_icu_from_patients_and_services(
        patients_path=patients_path,
        services_weekly_path=services_path,
        week1_monday=ref_monday,
    )

    staff_daily = _canonical_staff_from_schedule(
        staff_schedule_path=staff_schedule_path,
        week1_monday=ref_monday,
    )

    # Calendar flags derived for the full date range of the master
    all_dates = pd.date_range(
        start=emergency_daily["date"].min(),
        end=emergency_daily["date"].max(),
        freq="D",
    )
    calendar_daily = _canonical_calendar_from_holidays(
        holidays_path=holidays_path,
        date_index=all_dates,
        preferred_country_codes=["IN", "IND", "US", "GB"],
    )

    env_health_daily = _canonical_env_health_from_influenza(
        seasonal_influenza_path=seasonal_influenza_path,
        influenza_weekly_path=influenza_weekly_path,
    )

    # Merge to master (outer to keep all days)
    master_df = emergency_daily.merge(icu_daily, on="date", how="outer")
    master_df = master_df.merge(staff_daily, on="date", how="outer")
    master_df = master_df.merge(calendar_daily, on="date", how="left")
    master_df = master_df.merge(env_health_daily, on="date", how="left")
    master_df = master_df.sort_values("date").reset_index(drop=True)

    # Prefix columns to keep dataset provenance consistent with the rest of the project.
    emergency_df = _prefix_non_key_columns(emergency_daily, "emergency", key_columns=["date"])
    icu_df = _prefix_non_key_columns(icu_daily, "icu", key_columns=["date"])
    staff_df = _prefix_non_key_columns(staff_daily, "staff", key_columns=["date"])
    calendar_df = _prefix_non_key_columns(calendar_daily, "calendar", key_columns=["date"])
    env_df = _prefix_non_key_columns(env_health_daily, "env_health", key_columns=["date"])

    master_df = emergency_df
    for next_df in [icu_df, staff_df, calendar_df, env_df]:
        master_df = master_df.merge(next_df, on="date", how="outer", sort=True)
    master_df = master_df.sort_values("date").reset_index(drop=True)

    summaries: Dict[str, Dict[str, object]] = {
        "emergency": _summarize_dataframe("emergency", emergency_df),
        "icu": _summarize_dataframe("icu", icu_df),
        "staff": _summarize_dataframe("staff", staff_df),
        "calendar": _summarize_dataframe("calendar", calendar_df),
        "env_health": _summarize_dataframe("env_health", env_df),
    }
    return master_df, summaries


if __name__ == "__main__":
    # Example CLI-style usage. In a real workflow, this section can be
    # replaced by a notebook or pipeline orchestration code.
    base_path = Path(".")

    master, dataset_summaries = example_build_master_from_default_files(base_path)

    # Print small, human-readable overview without dumping full dataframes
    print("Master dataframe shape:", master.shape)
    print("Master dataframe columns:")
    print(master.columns.tolist())

    print("\nPer-dataset summaries (rows, columns, missing counts):")
    for name, summary in dataset_summaries.items():
        print(f"\n=== {name} ===")
        print("Rows:", summary["num_rows"])
        print("Columns:", summary["num_columns"])
        print("Missing values per column:")
        for col, n_missing in summary["missing_values_per_column"].items():
            print(f"  {col}: {n_missing}")
