# src/clean_pipeline.py
import pandas as pd
import numpy as np
import os
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- File paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
TEST_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "test.csv"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_house_prices.csv"
PROCESSED_TEST_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_test.csv"
STATS_PATH = PROJECT_ROOT / "data" / "processed" / "imputation_stats.json"

# Ordinal quality scales used throughout this dataset (Po < Fa < TA < Gd < Ex)
QUALITY_MAP = {"No_Garage": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5,
               "NA": 0, "None": 0}
ORDINAL_QUALITY_COLS = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
    "KitchenQual", "GarageQual", "GarageCond", "FireplaceQu", "PoolQC",
]


def load_data(path=RAW_PATH):
    """Load a raw CSV (train or test)."""
    if not os.path.exists(path):
        logging.error(f"File not found: {path}")
        raise FileNotFoundError(f"Please place the CSV at {path}")
    df = pd.read_csv(path)
    logging.info(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns from {path.name}")
    return df


def remove_outliers(df):
    """
    Drop the well-known GrLivArea outliers documented by the dataset author
    (Dean De Cock): a couple of very large homes sold for suspiciously low
    prices. Left in, they distort linear-model fits significantly.
    Only applied to training data (test data must never be dropped).
    """
    if "SalePrice" not in df.columns:
        return df  # this is the test set, has no SalePrice — skip
    before = df.shape[0]
    df = df[~((df["GrLivArea"] > 4000) & (df["SalePrice"] < 300000))].copy()
    removed = before - df.shape[0]
    if removed:
        logging.info(f"Removed {removed} known outlier rows (large GrLivArea, low SalePrice).")
    return df


def drop_high_missing(df, threshold=50, cols_to_drop=None):
    """
    Drop columns where more than 'threshold' percent of values are missing.
    If cols_to_drop is provided (e.g. computed on train), reuse it so
    train and test end up with identical columns.
    """
    if cols_to_drop is None:
        missing_percent = df.isnull().mean() * 100
        cols_to_drop = missing_percent[missing_percent > threshold].index.tolist()
    cols_present = [c for c in cols_to_drop if c in df.columns]
    if cols_present:
        df = df.drop(columns=cols_present)
        logging.info(f"Dropped columns: {cols_present}")
    else:
        logging.info("No columns to drop.")
    return df, cols_to_drop


def fix_garage_columns(df):
    """NaN in garage columns means 'No Garage' — fill accordingly rather than imputing."""
    garage_text = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
    for col in garage_text:
        if col in df.columns:
            df[col] = df[col].fillna("No_Garage")
    if 'GarageYrBlt' in df.columns:
        df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)
    logging.info("Fixed garage columns (NaNs represent 'No Garage').")
    return df


def encode_ordinal_quality(df):
    """Map quality/condition text columns onto a numeric 0-5 scale instead of
    leaving them as unordered categoricals — preserves the Po<Fa<TA<Gd<Ex ordering."""
    for col in ORDINAL_QUALITY_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("NA").map(QUALITY_MAP).fillna(0).astype(int)
    logging.info(f"Ordinally encoded quality columns: "
                 f"{[c for c in ORDINAL_QUALITY_COLS if c in df.columns]}")
    return df


def fit_impute_stats(df):
    """Compute median (numeric) / mode (categorical) fill values from TRAINING data only."""
    stats = {}
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                stats[col] = {"type": "numeric", "value": float(df[col].median())}
            else:
                stats[col] = {"type": "categorical", "value": str(df[col].mode()[0])}
    return stats


def apply_impute_stats(df, stats):
    """Apply previously-fit imputation stats to any dataframe (train or test)."""
    for col, info in stats.items():
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(info["value"])
            logging.info(f"   Imputed {col} with {info['type']} value: {info['value']}")
    return df


def engineer_features(df):
    """Create engineered features. Safe to run on train or test — no target leakage."""
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalBath'] = (df['FullBath'] + 0.5 * df['HalfBath'] +
                        df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath'])
    df['TotalPorchSF'] = (df['OpenPorchSF'] + df['EnclosedPorch'] +
                           df['3SsnPorch'] + df['ScreenPorch'])
    df['Quality_Area'] = df['OverallQual'] * df['TotalSF']
    df['HouseAge'] = (df['YrSold'] - df['YearBuilt']).clip(lower=0)
    df['RemodAge'] = (df['YrSold'] - df['YearRemodAdd']).clip(lower=0)
    df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
    logging.info("Engineered 7 features (TotalSF, TotalBath, TotalPorchSF, "
                 "Quality_Area, HouseAge, RemodAge, HasGarage).")
    return df


def save_data(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logging.info(f"Saved data to {path}")


def process_train():
    """Full pipeline on training data. Fits stats and saves them for reuse on test."""
    logging.info("=== Processing TRAIN data ===")
    df = load_data(RAW_PATH)
    df = remove_outliers(df)
    df, dropped_cols = drop_high_missing(df)
    df = fix_garage_columns(df)
    df = encode_ordinal_quality(df)

    stats = fit_impute_stats(df)
    df = apply_impute_stats(df, stats)

    df = engineer_features(df)
    save_data(df, PROCESSED_PATH)

    os.makedirs(os.path.dirname(STATS_PATH), exist_ok=True)
    with open(STATS_PATH, "w") as f:
        json.dump({"dropped_cols": dropped_cols, "impute_stats": stats}, f, indent=2)
    logging.info(f"Saved imputation stats to {STATS_PATH}")

    logging.info(f"Final TRAIN shape: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def process_test():
    """
    Apply the exact same transformations to test.csv, reusing the columns
    dropped and the imputation statistics learned from train — never
    re-fitting anything on test data.
    """
    if not os.path.exists(TEST_RAW_PATH):
        logging.warning(f"No test.csv found at {TEST_RAW_PATH} — skipping test processing.")
        return None
    if not os.path.exists(STATS_PATH):
        raise RuntimeError("Run process_train() first — no saved imputation stats found.")

    logging.info("=== Processing TEST data ===")
    with open(STATS_PATH) as f:
        saved = json.load(f)

    df = load_data(TEST_RAW_PATH)
    df, _ = drop_high_missing(df, cols_to_drop=saved["dropped_cols"])
    df = fix_garage_columns(df)
    df = encode_ordinal_quality(df)
    df = apply_impute_stats(df, saved["impute_stats"])
    df = engineer_features(df)
    save_data(df, PROCESSED_TEST_PATH)

    logging.info(f"Final TEST shape: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def main():
    logging.info("Starting Data Cleaning...")
    process_train()
    process_test()
    logging.info("Data Cleaning Complete!")


if __name__ == "__main__":
    main()