"""
Core processing functions for the sensor time-series labeling prototype.

These are adapted directly from the existing thesis notebook workflow.
This module is UI-agnostic (no Streamlit imports) so it can be reused,
tested, and eventually moved into src/ as-is when the full app is built.
"""
from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# Time handling
# ---------------------------------------------------------------------------

def has_split_time_columns(df: pd.DataFrame) -> bool:
    """
    Some raw exports split the timestamp across 4 columns instead of one.
    Detects that format so rebuild_split_time() is only applied when needed.
    """
    return all(c in df.columns for c in ["Unnamed: 1", "Unnamed: 2", "Unnamed: 3"])


def rebuild_split_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct a single 'Time' timestamp column from a date column plus
    separate hour/minute/second columns.
    """
    df = df.copy()
    date = pd.to_datetime(df["Time"], format="%d.%m.%Y", errors="coerce")
    df["Time"] = (
        date
        + pd.to_timedelta(df["Unnamed: 1"].astype(int), unit="h")
        + pd.to_timedelta(df["Unnamed: 2"].astype(int), unit="m")
        + pd.to_timedelta(df["Unnamed: 3"].astype(int), unit="s")
    )
    df.drop(columns=["Unnamed: 1", "Unnamed: 2", "Unnamed: 3"], inplace=True)
    return df


def sensor_reading_interval(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add reading_interval_s, elapsed_runtime_s, elapsed_runtime_m columns.
    Must be called before trim_by_elapsed_seconds().
    """
    out = df.copy()
    out["Time"] = pd.to_datetime(out["Time"])
    out["reading_interval_s"] = out["Time"].diff().dt.total_seconds().fillna(0)
    elapsed_runtime = out["Time"] - out["Time"].iloc[0]
    out["elapsed_runtime_s"] = elapsed_runtime.dt.total_seconds()
    out["elapsed_runtime_m"] = out["elapsed_runtime_s"] / 60
    return out


def trim_by_elapsed_seconds(
    df: pd.DataFrame, start_s: float, end_s: float | None
) -> pd.DataFrame:
    """
    Replacement for snip_df_at_timecode(). That function sliced by row
    index (df.loc[start:end]), which only worked when the row index
    happened to line up with elapsed seconds.

    This trims using elapsed_runtime_s instead, which is a real, meaningful
    quantity regardless of sampling rate. Requires sensor_reading_interval()
    to have been applied first. end_s == 0 (or None) means "to the end",
    matching the original function's convention.
    """
    out = df.copy()
    if end_s is None or end_s == 0:
        out = out[out["elapsed_runtime_s"] >= start_s]
    else:
        out = out[(out["elapsed_runtime_s"] >= start_s) & (out["elapsed_runtime_s"] <= end_s)]
    out.reset_index(drop=True, inplace=True)
    return out


def run_checks(df: pd.DataFrame) -> dict:
    """
    Non-printing version of basic_checklist(): returns findings so the UI
    can render them instead of relying on stdout.
    """
    missing = df.isna().sum()
    missing = {k: int(v) for k, v in missing[missing > 0].to_dict().items()}
    monotonic = bool(df["Time"].is_monotonic_increasing) if "Time" in df.columns else None
    return {"missing_values": missing, "time_monotonic": monotonic}


# ---------------------------------------------------------------------------
# Derived pressure features
# ---------------------------------------------------------------------------

def add_pressure_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def require(col: str) -> None:
        if col not in df.columns:
            raise ValueError(
                f"add_pressure_features needs column '{col}', which is not "
                f"present in this file. Columns found: {list(df.columns)}"
            )

    require("Pump outlet pressure (Mean)")

    if "TS inlet pressure (Mean)" in df.columns:
        require("TS outlet pressure (Mean)")
        df["Differential pressure (Mean)"] = (
            df["TS inlet pressure (Mean)"] - df["TS outlet pressure (Mean)"]
        )
        df["Pump to inlet DP"] = df["Pump outlet pressure (Mean)"] - df["TS inlet pressure (Mean)"]
        df["Total system DP"] = df["Pump outlet pressure (Mean)"] - df["TS outlet pressure (Mean)"]
    else:
        require("TS outlet pressure (Mean)")
        require("Differential pressure (Mean)")
        df["TS inlet pressure (Mean)"] = (
            df["TS outlet pressure (Mean)"] + df["Differential pressure (Mean)"]
        )
        df["Pump to inlet DP"] = df["Pump outlet pressure (Mean)"] - df["TS inlet pressure (Mean)"]
        df["Total system DP"] = df["Pump outlet pressure (Mean)"] - df["TS outlet pressure (Mean)"]
    return df


MASTER_COLUMN_ORDER = [
    "Time",
    "Temperature TS inlet (Mean)",
    "Temperature TS outlet (Mean)",
    "Tank temperature (Mean)",
    "Bypass temperature (Mean)",
    "Pump outlet pressure (Mean)",
    "TS inlet pressure (Mean)",
    "TS outlet pressure (Mean)",
    "Differential pressure (Mean)",
    "Pump to inlet DP",
    "Total system DP",
    "Flow rate (Mean)",
    "reading_interval_s",
    "elapsed_runtime_s",
]


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in MASTER_COLUMN_ORDER if c in df.columns]
    remaining = [c for c in df.columns if c not in cols]
    return df[cols + remaining]


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def standardize_sampling(df: pd.DataFrame, target_freq: str = "1s") -> pd.DataFrame:
    df = df.select_dtypes(include=["number"])
    df = df.groupby("elapsed_runtime_s").mean()
    df.index = pd.to_timedelta(df.index, unit="s")
    standardized_df = df.resample(target_freq).mean()
    standardized_df = standardized_df.interpolate(method="linear")
    standardized_df.index = standardized_df.index.total_seconds()
    standardized_df.index.name = "elapsed_runtime_s"
    if "reading_interval_s" in standardized_df.columns:
        standardized_df = standardized_df.drop(columns=["reading_interval_s"])
    return standardized_df.reset_index()


# ---------------------------------------------------------------------------
# Relative ratio deviation + labeling
# ---------------------------------------------------------------------------

def add_rdd(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Pressure_Ratio"] = (
        df["TS inlet pressure (Mean)"] - df["TS outlet pressure (Mean)"]
    ) / (df["Pump outlet pressure (Mean)"] - df["TS inlet pressure (Mean)"] + 1e-6)

    base_ratio_val = df["Pressure_Ratio"].iloc[:60].median()
    df["Relative_Ratio_Dev"] = (df["Pressure_Ratio"] / (base_ratio_val + 1e-6) - 1).abs()
    df["run_specific_time_s"] = df["elapsed_runtime_s"] - df["elapsed_runtime_s"].iloc[0]
    return df


def label_run(df: pd.DataFrame, one: float, two: float, three: float) -> pd.DataFrame:
    out = df.copy()
    out["label"] = 0
    if one:
        out.loc[out["run_specific_time_s"] >= one, "label"] = 1
    if two:
        out.loc[out["run_specific_time_s"] >= two, "label"] = 2
    if three:
        out.loc[out["run_specific_time_s"] >= three, "label"] = 3
    return out


# ---------------------------------------------------------------------------
# Full per-run pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    raw_df: pd.DataFrame, start_s: float, end_s: float | None, target_freq: str
) -> pd.DataFrame:
    """
    Wires the functions above together in the order used in the notebook:
    time handling -> trim -> derived features -> reorder -> resample -> RDD.
    """
    df = raw_df.copy()
    if has_split_time_columns(df):
        df = rebuild_split_time(df)
    else:
        df["Time"] = pd.to_datetime(df["Time"])

    df = sensor_reading_interval(df)
    df = trim_by_elapsed_seconds(df, start_s, end_s)
    df = add_pressure_features(df)
    df = reorder_columns(df)
    df = standardize_sampling(df, target_freq)
    df = add_rdd(df)
    return df
