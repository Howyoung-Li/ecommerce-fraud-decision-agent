from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    build_feature_frame,
    feature_columns,
    split_feature_frame,
)
from .paths import DATA_ARTIFACTS_DIR, MODELING_ARTIFACTS_DIR, POLICY_ARTIFACTS_DIR


TOPK_RATES = [0.01, 0.03, 0.05, 0.10, 0.20]


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_transformer = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def build_logistic_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(scale_numeric=True)),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="lbfgs",
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_xgb_pipeline(y_train: pd.Series) -> Pipeline:
    positive = int(y_train.sum())
    negative = int((y_train == 0).sum())
    scale_pos_weight = negative / positive if positive else 1.0
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(scale_numeric=False)),
            (
                "model",
                XGBClassifier(
                    n_estimators=220,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.90,
                    colsample_bytree=0.90,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    scale_pos_weight=scale_pos_weight,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def ks_statistic(y_true: pd.Series, score: np.ndarray) -> float:
    table = pd.DataFrame({"y": y_true.to_numpy(), "score": score})
    positives = table[table["y"].eq(1)]["score"]
    negatives = table[table["y"].eq(0)]["score"]
    thresholds = np.sort(np.unique(score))
    if positives.empty or negatives.empty:
        return 0.0
    pos_cdf = np.searchsorted(np.sort(positives), thresholds, side="right") / len(positives)
    neg_cdf = np.searchsorted(np.sort(negatives), thresholds, side="right") / len(negatives)
    return float(np.max(np.abs(pos_cdf - neg_cdf)))


def evaluate_scores(y_true: pd.Series, score: np.ndarray) -> dict[str, float]:
    precision, recall, _ = precision_recall_curve(y_true, score)
    return {
        "rows": int(len(y_true)),
        "fraud_rows": int(y_true.sum()),
        "fraud_rate": float(y_true.mean()),
        "roc_auc": float(roc_auc_score(y_true, score)),
        "pr_auc": float(average_precision_score(y_true, score)),
        "ks": ks_statistic(y_true, score),
        "max_f1": float(np.max(2 * precision * recall / np.maximum(precision + recall, 1e-12))),
    }


def topk_table(scored: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    ordered = scored.sort_values(score_col, ascending=False).reset_index(drop=True)
    total_fraud = ordered[TARGET].sum()
    rows = []
    for rate in TOPK_RATES:
        k = max(1, int(round(len(ordered) * rate)))
        top = ordered.iloc[:k]
        fraud_captured = int(top[TARGET].sum())
        precision = fraud_captured / k
        recall = fraud_captured / total_fraud if total_fraud else 0.0
        rows.append(
            {
                "topk_rate": rate,
                "review_volume": k,
                "fraud_captured": fraud_captured,
                "precision_at_k": precision,
                "recall_at_k": recall,
                "baseline_fraud_rate": float(ordered[TARGET].mean()),
                "lift_vs_baseline": precision / ordered[TARGET].mean()
                if ordered[TARGET].mean()
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


def score_frame(model: Pipeline, frame: pd.DataFrame, model_name: str) -> pd.DataFrame:
    out = frame[
        [
            "user_id",
            "purchase_time",
            TARGET,
            "split",
            "signup_to_purchase_seconds",
            "purchase_value",
            "device_seen_count_hist",
            "ip_seen_count_hist",
            "source",
            "browser",
            "country",
        ]
    ].copy()
    out["model_name"] = model_name
    out["score"] = model.predict_proba(frame[feature_columns()])[:, 1]
    return out


def train_and_write_outputs() -> dict[str, Path]:
    MODELING_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    POLICY_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    frame = build_feature_frame()
    frame.to_csv(DATA_ARTIFACTS_DIR / "feature_frame.csv", index=False)
    splits = split_feature_frame(frame)

    x_train = splits.train[feature_columns()]
    y_train = splits.train[TARGET]
    models = {
        "logistic_regression": build_logistic_pipeline(),
        "xgboost": build_xgb_pipeline(y_train),
    }

    metrics: dict[str, dict[str, dict[str, float]]] = {}
    scored_frames = []
    topk_frames = []

    for model_name, model in models.items():
        model.fit(x_train, y_train)
        joblib.dump(model, MODELING_ARTIFACTS_DIR / f"{model_name}.joblib")
        metrics[model_name] = {}
        for split_name, split in [
            ("train", splits.train),
            ("valid", splits.valid),
            ("test", splits.test),
        ]:
            scored = score_frame(model, split, model_name)
            scored_frames.append(scored.assign(evaluation_split=split_name))
            metrics[model_name][split_name] = evaluate_scores(split[TARGET], scored["score"].to_numpy())
            if split_name in {"valid", "test"}:
                topk_frames.append(
                    topk_table(scored)
                    .assign(model_name=model_name, evaluation_split=split_name)
                )

    metrics_path = MODELING_ARTIFACTS_DIR / "model_metrics.json"
    scored_path = MODELING_ARTIFACTS_DIR / "scored_transactions.csv"
    topk_path = POLICY_ARTIFACTS_DIR / "topk_review_strategy.csv"

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.concat(scored_frames, ignore_index=True).to_csv(scored_path, index=False)
    pd.concat(topk_frames, ignore_index=True).to_csv(topk_path, index=False)

    return {
        "metrics": metrics_path,
        "scored_transactions": scored_path,
        "topk_strategy": topk_path,
    }

