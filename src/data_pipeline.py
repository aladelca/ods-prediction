"""Data loading and split utilities for ODS text classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_PATH,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    DEFAULT_TRAIN_SIZE,
    DEFAULT_VAL_SIZE,
)


@dataclass
class DatasetSplit:
    X_train: pd.Series
    y_train: pd.Series
    X_val: pd.Series
    y_val: pd.Series
    X_test: pd.Series
    y_test: pd.Series


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and validate training dataset."""
    if not path.exists():
        raise FileNotFoundError(f"No existe dataset en: {path}")

    df = pd.read_excel(path)
    required = {"textos", "ODS"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {sorted(missing)}")

    df = df.dropna(subset=["textos", "ODS"]).copy()
    df["textos"] = df["textos"].astype(str)
    df["ODS"] = df["ODS"].astype(int)
    return df


def maybe_subsample(
    df: pd.DataFrame,
    sample_size: int | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """Return stratified subsample if sample_size is provided."""
    if sample_size is None or sample_size <= 0 or sample_size >= len(df):
        return df.reset_index(drop=True)

    sampled, _ = train_test_split(
        df,
        train_size=sample_size,
        stratify=df["ODS"],
        random_state=random_state,
    )
    return sampled.reset_index(drop=True)


def split_train_val_test(
    df: pd.DataFrame,
    train_size: float = DEFAULT_TRAIN_SIZE,
    val_size: float = DEFAULT_VAL_SIZE,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> DatasetSplit:
    """Create stratified train/val/test split."""
    if abs((train_size + val_size + test_size) - 1.0) > 1e-8:
        raise ValueError("train_size + val_size + test_size debe sumar 1.0")

    X = df["textos"]
    y = df["ODS"]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    val_ratio_within_train_val = val_size / (train_size + val_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_ratio_within_train_val,
        stratify=y_train_val,
        random_state=random_state,
    )

    return DatasetSplit(
        X_train=X_train.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        X_val=X_val.reset_index(drop=True),
        y_val=y_val.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
    )
