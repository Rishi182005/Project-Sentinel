# training.py
import json
import os

import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

from .sentinel_config import PATH, SAMPLE_USERS
from .features import extract_features, FEATURE_COLS
from .utils_safe import safe_str

# where to store models: Sentinel/models/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(PATH)))  # -> Sentinel
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def save_model(model, name: str = "sentinel_xgb.pkl"):
    """Save trained model to models/ folder."""
    path = os.path.join(MODELS_DIR, name)
    joblib.dump(model, path)
    print(f" Model saved to {path}")
    return path


def load_json(filename):
    path = os.path.join(PATH, filename)
    print(f" Loading {filename}...", end=" ")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"{len(data):,} records")
    return data


def build_dataframe():
    train_data = load_json("train.json")
    test_data = load_json("test.json")
    dev_data = load_json("dev.json")

    all_data = train_data + test_data + dev_data
    print(f"\n Total: {len(all_data):,} records\n")

    all_features, errors = [], 0
    print(" Extracting features...")
    for i, user in enumerate(all_data):
        if i % 5000 == 0:
            print(f" {i}/{len(all_data)} | ok={len(all_features)} err={errors}")
        feat = extract_features(user)
        if feat:
            all_features.append(feat)
        else:
            errors += 1

    print(f" Done: {len(all_features):,} ok, {errors} errors\n")

    df = pd.DataFrame(all_features)
    df = df[df["label"] != -1].reset_index(drop=True)
    return df, all_data


def train_model(df):
    X = df[FEATURE_COLS].fillna(0).astype(float)
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"...  Training XGBoost on {len(X_train):,} samples...")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    print(" Training complete!\n")
    return model, X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(
        y_test, y_pred, target_names=["Human", "Bot"], output_dict=True
    )
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn) * 100

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    step = max(1, len(fpr) // 60)
    roc_fpr = [round(float(v), 4) for v in fpr[::step]]
    roc_tpr = [round(float(v), 4) for v in tpr[::step]]
    if roc_fpr[-1] != 1.0:
        roc_fpr.append(1.0)
        roc_tpr.append(1.0)

    print(" ── Performance ────────────────────────────")
    print(classification_report(y_test, y_pred, target_names=["Human", "Bot"]))
    print(f" AUC-ROC: {auc:.4f}\n")

    return {
        "report": report,
        "auc": auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": accuracy,
        "roc_fpr": roc_fpr,
        "roc_tpr": roc_tpr,
    }


def sample_users_for_dashboard(model, all_data):
    from .features import FEATURE_COLS, extract_features

    samples = []
    shown = 0
    for user in all_data:
        if shown >= SAMPLE_USERS:
            break
        feat = extract_features(user)
        if not feat:
            continue

        import pandas as pd

        Xu = pd.DataFrame([feat])[FEATURE_COLS].fillna(0).astype(float)
        prob = float(model.predict_proba(Xu)[0][1])
        name = safe_str(user.get("profile", {}).get("screen_name", "unknown"))

        true_label = user.get("label", None)
        if true_label in ("bot", 1, "1", 1.0):
            true_str = "BOT"
        elif true_label in ("human", 0, "0", 0.0):
            true_str = "HUMAN"
        else:
            true_str = "?"

        samples.append(
            {
                "handle": name if name else "unknown",
                "botProb": round(prob * 100, 2),
                "authScore": round((1 - prob) * 100, 2),
                "verdict": "BOT" if prob > 0.5 else "HUMAN",
                "trueLabel": true_str,
                "correct": (prob > 0.5) == (true_str == "BOT"),
                "dailyCV": round(feat.get("daily_count_cv", 0), 3),
                "timingStd": round(feat.get("std_interval_sec", 0), 1),
                "burst": round(feat.get("burst_ratio", 0), 3),
                "followers": feat.get("followers_count", 0),
                "verified": bool(feat.get("verified", 0)),
                "descLen": feat.get("description_length", 0),
                "accountAge": feat.get("account_age_days", 0),
            }
        )
        shown += 1

    return samples
