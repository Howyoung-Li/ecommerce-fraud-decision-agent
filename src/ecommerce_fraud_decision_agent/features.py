from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data_audit import load_fraud_data, load_ip_country, map_country


NUMERIC_FEATURES = [
    "signup_to_purchase_seconds",
    "signup_to_purchase_minutes",
    "purchase_value",
    "purchase_value_log",
    "purchase_hour",
    "purchase_dayofweek",
    "is_night_purchase",
    "is_fast_purchase_le_1s",
    "is_flash_purchase_le_2min",
    "device_seen_count_hist",
    "ip_seen_count_hist",
    "country_is_unknown",
]

CATEGORICAL_FEATURES = [
    "source",
    "browser",
    "country",
]

TARGET = "class"


@dataclass(frozen=True)
class SplitFrames:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def add_asof_history_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add prior-event counts using only earlier purchase timestamps."""
    ordered = data.sort_values(["purchase_time", "user_id"]).reset_index(drop=True).copy()
    ordered["device_seen_count_hist"] = ordered.groupby("device_id").cumcount()
    ordered["ip_seen_count_hist"] = ordered.groupby("ip_address").cumcount()
    return ordered


def build_feature_frame() -> pd.DataFrame:
    data = load_fraud_data()
    data["country"] = map_country(data, load_ip_country())
    data = add_asof_history_features(data)

    data["signup_to_purchase_minutes"] = data["signup_to_purchase_seconds"] / 60
    data["purchase_value_log"] = np.log1p(data["purchase_value"])
    data["purchase_hour"] = data["purchase_time"].dt.hour
    data["purchase_dayofweek"] = data["purchase_time"].dt.dayofweek
    data["is_night_purchase"] = (
        (data["purchase_hour"] <= 5) | (data["purchase_hour"] >= 23)
    ).astype(int)
    data["is_fast_purchase_le_1s"] = data["is_fast_purchase_le_1s"].astype(int)
    data["is_flash_purchase_le_2min"] = (
        data["signup_to_purchase_seconds"] <= 120
    ).astype(int)
    data["country_is_unknown"] = data["country"].eq("UNKNOWN").astype(int)
    data["split"] = assign_time_splits(data)
    return data


def assign_time_splits(ordered: pd.DataFrame) -> pd.Series:
    n = len(ordered)
    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)
    split = pd.Series("test", index=ordered.index, dtype=object)
    split.iloc[:train_end] = "train"
    split.iloc[train_end:valid_end] = "valid"
    return split


def split_feature_frame(frame: pd.DataFrame) -> SplitFrames:
    return SplitFrames(
        train=frame[frame["split"].eq("train")].copy(),
        valid=frame[frame["split"].eq("valid")].copy(),
        test=frame[frame["split"].eq("test")].copy(),
    )


def feature_columns() -> list[str]:
    return NUMERIC_FEATURES + CATEGORICAL_FEATURES
