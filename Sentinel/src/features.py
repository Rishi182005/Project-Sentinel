# features.py
import pandas as pd
from .utils_safe import (
    safe_int,
    safe_bool,
    safe_str,
    safe_url,
    TIMING_COLS,
    compute_timing_features,
)

def extract_features(user):
    f = {}
    try:
        p = user.get("profile", {}) or {}

        # basic counts
        f["followers_count"] = safe_int(p.get("followers_count", 0))
        f["friends_count"] = safe_int(p.get("friends_count", 0))
        f["statuses_count"] = safe_int(p.get("statuses_count", 0))
        f["favourites_count"] = safe_int(p.get("favourites_count", 0))
        f["listed_count"] = safe_int(p.get("listed_count", 0))

        # booleans / url
        f["verified"] = safe_bool(p.get("verified", False))
        f["has_url"] = safe_url(p.get("url", None))

        # text features
        desc = safe_str(p.get("description", ""))
        sname = safe_str(p.get("screen_name", ""))
        name = safe_str(p.get("name", ""))

        f["description_length"] = len(desc)
        f["has_description"] = int(len(desc) > 0)
        f["screen_name_length"] = len(sname)
        f["name_length"] = len(name)

        # ratios
        fol = f["followers_count"]
        fri = f["friends_count"]
        f["follower_following_ratio"] = fol / (fri + 1)
        f["following_follower_ratio"] = fri / (fol + 1)
        f["engagement_rate"] = f["favourites_count"] / (f["statuses_count"] + 1)
        f["username_digit_ratio"] = sum(c.isdigit() for c in sname) / (len(sname) + 1)

        # account age
        ca = p.get("created_at", None)
        if ca:
            try:
                created = pd.to_datetime(str(ca), errors="coerce")
                now = pd.Timestamp.now(tz="UTC")
                if pd.notna(created):
                    if created.tzinfo is None:
                        created = created.tz_localize("UTC")
                    f["account_age_days"] = max(0, (now - created).days)
                else:
                    f["account_age_days"] = 0
            except Exception:
                f["account_age_days"] = 0
        else:
            f["account_age_days"] = 0

        # tweets and timing features
        tweets = user.get("tweet", []) or []
        f["tweet_count"] = len(tweets)
        f.update(compute_timing_features(tweets))

        # label
        label = user.get("label", None)
        if label in ("bot", 1, "1", 1.0):
            f["label"] = 1
        elif label in ("human", 0, "0", 0.0):
            f["label"] = 0
        else:
            f["label"] = -1

    except Exception:
        return None

    return f

FEATURE_COLS = [
    "followers_count",
    "friends_count",
    "statuses_count",
    "favourites_count",
    "listed_count",
    "verified",
    "has_url",
    "description_length",
    "has_description",
    "screen_name_length",
    "name_length",
    "follower_following_ratio",
    "following_follower_ratio",
    "engagement_rate",
    "username_digit_ratio",
    "account_age_days",
    "tweet_count",
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
