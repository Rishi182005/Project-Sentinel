# utils_safe.py
import pandas as pd
import numpy as np
from scipy.stats import entropy

def safe_int(val, default=0):
    if val is None:
        return default
    if isinstance(val, bool):
        return int(val)
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, str):
        v = val.strip().lower()
        if v in ("true",):
            return 1
        if v in ("false", "", "none", "null"):
            return 0
        try:
            return int(float(v))
        except Exception:
            return default
    return default

def safe_bool(val):
    if val is None:
        return 0
    if isinstance(val, bool):
        return int(val)
    if isinstance(val, (int, float)):
        return 1 if val else 0
    if isinstance(val, str):
        return 1 if val.strip().lower() == "true" else 0
    return 0

def safe_str(val):
    if val is None:
        return ""
    if isinstance(val, (bool, int, float)):
        return ""
    return str(val).strip()

def safe_url(val):
    if val is None:
        return 0
    s = str(val).strip().lower()
    return 0 if s in ("", "none", "null", "false", "0") else 1

TIMING_COLS = [
    "mean_interval_sec",
    "std_interval_sec",
    "min_interval_sec",
    "burst_ratio",
    "night_post_ratio",
    "posting_hour_std",
    "daily_count_mean",
    "daily_count_std",
    "daily_count_cv",
    "interval_entropy",
]


def compute_timing_features(tweets):
    f = {}
    timestamps, hours = [], []

    for t in tweets:
        try:
            raw = t.get("created_at", None)
            if raw:
                ts = pd.to_datetime(str(raw), errors="coerce")
                if pd.notna(ts):
                    if ts.tzinfo is None:
                        ts = ts.tz_localize("UTC")
                    timestamps.append(ts)
                    hours.append(ts.hour)
        except Exception:
            continue

    if len(timestamps) >= 2:
        timestamps = sorted(timestamps)
        intervals = [
            abs((timestamps[i + 1] - timestamps[i]).total_seconds())
            for i in range(len(timestamps) - 1)
        ]

        f["mean_interval_sec"] = float(np.mean(intervals))
        f["std_interval_sec"] = float(np.std(intervals))
        f["min_interval_sec"] = float(np.min(intervals))
        f["burst_ratio"] = sum(1 for x in intervals if x < 60) / len(intervals)
        f["night_post_ratio"] = sum(1 for h in hours if h < 6) / max(len(hours), 1)
        f["posting_hour_std"] = float(np.std(hours))

        dc = pd.Series([t.date() for t in timestamps]).value_counts()
        f["daily_count_mean"] = float(dc.mean())
        f["daily_count_std"] = float(dc.std()) if len(dc) > 1 else 0.0
        f["daily_count_cv"] = (
            float(dc.std()) / float(dc.mean()) if dc.mean() > 0 else 0.0
        )

        hist, _ = np.histogram(intervals, bins=24)
        f["interval_entropy"] = float(entropy(hist + 1))
    else:
        for col in TIMING_COLS:
            f[col] = 0.0

    return f
