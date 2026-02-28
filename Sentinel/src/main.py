# main.py (inside src)
import warnings
warnings.filterwarnings("ignore")

from .sentinel_config import PATH
from .training import (
    build_dataframe,
    train_model,
    evaluate_model,
    sample_users_for_dashboard,
    save_model,
)
from .dashboard import render_html, save_and_open_dashboard
from .features import FEATURE_COLS


def main():
    print("=" * 60)
    print(" SENTINEL — Loading TwiBot-20 Dataset")
    print("=" * 60)

    df, all_data = build_dataframe()

    total_samples = len(df)
    total_bots = int(df["label"].sum())
    total_humans = int((df["label"] == 0).sum())
    bot_pct = df["label"].mean() * 100

    print(f" Dataset: {total_samples:,} samples")
    print(f" Bots : {total_bots:,} ({bot_pct:.1f}%)")
    print(f" Humans: {total_humans:,}\n")

    model, X_train, X_test, y_train, y_test = train_model(df)
    save_model(model, "sentinel_xgb.pkl")
    eval_res = evaluate_model(model, X_test, y_test)

    importance = (
        df[FEATURE_COLS]
        .columns.to_series()
        .copy()
    )  # you can recompute with model.feature_importances_

    import pandas as pd

    importance = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(
        ascending=False
    )
    feat_names = list(importance.index[:15])
    feat_scores = [float(v) for v in importance.values[:15]]

    sample_users = sample_users_for_dashboard(model, all_data)

    html = render_html(
        accuracy=eval_res["accuracy"],
        auc=eval_res["auc"],
        report=eval_res["report"],
        tp=eval_res["tp"],
        tn=eval_res["tn"],
        fp=eval_res["fp"],
        fn=eval_res["fn"],
        total_samples=total_samples,
        total_bots=total_bots,
        total_humans=total_humans,
        bot_pct=bot_pct,
        feat_names=feat_names,
        feat_scores=feat_scores,
        sample_users=sample_users,
        roc_fpr=eval_res["roc_fpr"],
        roc_tpr=eval_res["roc_tpr"],
        test_size=len(X_test),
        train_size=len(X_train),
        n_features=len(FEATURE_COLS),
    )

    save_and_open_dashboard(html)


if __name__ == "__main__":
    main()
