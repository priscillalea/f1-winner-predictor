import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

from src.data_loader import FEATURES

RANDOM_STATE = 42


def top3_precision(true_pos: pd.Series, pred_pos: np.ndarray) -> float:
    """Fraction of the true top-3 finishers captured in the predicted top-3."""
    df_ev = pd.DataFrame({"true": true_pos.values, "pred": pred_pos})
    true_top3 = set(df_ev.nsmallest(3, "true").index)
    pred_top3 = set(df_ev.nsmallest(3, "pred").index)
    return len(true_top3 & pred_top3) / 3.0


@st.cache_resource(show_spinner="Training Random Forest & XGBoost models...")
def train_and_evaluate(_df_feat: pd.DataFrame, feature_names: tuple = tuple(FEATURES)) -> dict:
    """Train Random Forest and XGBoost with a temporal 75/25 split and return models + metrics."""
    df_clean = pd.DataFrame(_df_feat).dropna(subset=FEATURES + ["Position"]).copy()
    rounds_sorted = sorted(df_clean["Round"].unique())
    split_at = rounds_sorted[int(len(rounds_sorted) * 0.75)]

    train = df_clean[df_clean["Round"] < split_at].copy()
    test = df_clean[df_clean["Round"] >= split_at].copy()

    X_train, y_train = train[FEATURES], train["Position"]
    X_test, y_test = test[FEATURES], test["Position"]

    rf = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)

    xgb_model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    xgb_model.fit(X_train, y_train)

    models = {"Random Forest": rf, "XGBoost": xgb_model}
    eval_rows = []
    for name, model in models.items():
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)
        train_mae = mean_absolute_error(y_train, train_preds)
        test_mae = mean_absolute_error(y_test, test_preds)
        p3 = float(
            np.mean(
                [
                    top3_precision(g["Position"], model.predict(g[FEATURES]))
                    for _, g in test.groupby("Round")
                ]
            )
        )
        eval_rows.append(
            {
                "Model": name,
                "Train MAE": round(train_mae, 3),
                "Test MAE": round(test_mae, 3),
                "Top-3 Precision": round(p3, 3),
            }
        )
        test[f"pred_{name}"] = np.round(test_preds, 2)

    eval_df = pd.DataFrame(eval_rows)
    best_name = eval_df.loc[eval_df["Test MAE"].idxmin(), "Model"]

    return {
        "models": models,
        "eval_df": eval_df,
        "best_name": best_name,
        "best_model": models[best_name],
        "split_at": int(split_at),
        "train_rounds": f"Rounds 1–{split_at - 1}",
        "test_rounds": f"Rounds {split_at}–{rounds_sorted[-1]}",
        "train_size": len(train),
        "test_size": len(test),
        "test_df": test,
    }
