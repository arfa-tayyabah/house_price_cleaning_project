import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import RidgeCV
import logging
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Dynamic Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "raw" / "test.csv"
SUBMISSION_PATH = PROJECT_ROOT / "submission.csv"

FEATURES = ["Quality_Area", "TotalSF", "TotalBath", "HouseAge", "HasGarage"]

GARAGE_TEXT_COLS = ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]
# --- Helper Functions ---

def fix_garage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Fix garage columns: NaN = 'No Garage'."""
    for col in GARAGE_TEXT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("No_Garage")
    if "GarageYrBlt" in df.columns:
        df["GarageYrBlt"] = df["GarageYrBlt"].fillna(0)
    return df

def compute_impute_values(df: pd.DataFrame) -> dict:
    """Compute median (numeric) / mode (categorical) fill values from a reference df."""
    values = {}
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                values[col] = df[col].median()
            else:
                mode = df[col].mode(dropna=True)
                values[col] = mode.iloc[0] if not mode.empty else "Unknown"
    return values


def apply_impute_values(df: pd.DataFrame, values: dict) -> pd.DataFrame:
    """Apply precomputed fill values to a df (train or test)."""
    for col, val in values.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)  # plain reassignment, not chained inplace
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features used by the final model (+ a couple extras kept for reference)."""
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalBath"] = (
        df["FullBath"]
        + 0.5 * df["HalfBath"]
        + df["BsmtFullBath"]
        + 0.5 * df["BsmtHalfBath"]
    )
    df["TotalPorchSF"] = (
        df["OpenPorchSF"] + df["EnclosedPorch"] + df["3SsnPorch"] + df["ScreenPorch"]
    )
    df["Quality_Area"] = df["OverallQual"] * df["TotalSF"]
    df["HouseAge"] = (df["YrSold"] - df["YearBuilt"]).clip(lower=0)
    df["RemodAge"] = (df["YrSold"] - df["YearRemodAdd"]).clip(lower=0)
    df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    return df


def clean_pipeline(df: pd.DataFrame, impute_values: dict = None) -> tuple[pd.DataFrame, dict]:
    """
    Apply cleaning steps to a DataFrame.
    If impute_values is None, this is treated as the reference (train) set:
    impute values are computed from it and returned for reuse on test.
    If impute_values is provided, those exact values are applied (no recomputation).
    """
    df = fix_garage_columns(df)

    if impute_values is None:
        impute_values = compute_impute_values(df)

    df = apply_impute_values(df, impute_values)
    df = engineer_features(df)

    # Safety net: if test has NaNs in columns train never saw missing, catch it explicitly
    remaining = df.isnull().sum()
    remaining = remaining[remaining > 0]
    if len(remaining):
        logging.warning(
            "Columns still containing NaN after imputation (unseen-in-train pattern): %s",
            remaining.to_dict(),
        )
        for col in remaining.index:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna("Unknown")

    return df, impute_values


def main():
    logging.info("Starting prediction pipeline...")

    # --- 1. Load and Clean Training Data ---
    df_train = pd.read_csv(TRAIN_PATH)
    df_train, impute_values = clean_pipeline(df_train)

    missing_features = [f for f in FEATURES if f not in df_train.columns]
    if missing_features:
        raise ValueError(f"Missing expected features in training data: {missing_features}")

    # --- 2. Train Final Model on 100% of Training Data ---
    X_train = df_train[FEATURES]
    y_train = df_train["SalePrice"]

    model = RidgeCV(alphas=[0.1, 1.0, 10.0])
    model.fit(X_train, y_train)
    logging.info("Model trained on 100%% of training data. Chosen alpha: %.3f", model.alpha_)

    # --- 3. Load and Clean Test Data (reusing train's impute values) ---
    df_test_raw = pd.read_csv(TEST_PATH)
    expected_rows = len(df_test_raw)
    expected_ids = df_test_raw["Id"] if "Id" in df_test_raw.columns else None

    df_test, _ = clean_pipeline(df_test_raw.copy(), impute_values=impute_values)

    if "Id" not in df_test.columns:
        raise ValueError("Test data is missing required 'Id' column.")

    # --- Row-count / Id integrity guard ---
    if len(df_test) != expected_rows:
        raise ValueError(
            f"Row count changed during cleaning: raw test.csv has {expected_rows} rows, "
            f"but cleaned df_test has {len(df_test)} rows. Something in clean_pipeline "
            f"is adding/dropping rows (check for merges, dropna(), or dedup calls)."
        )

    dup_ids = df_test["Id"].duplicated().sum()
    if dup_ids > 0:
        raise ValueError(
            f"Found {dup_ids} duplicate 'Id' values in test data after cleaning. "
            f"Submission requires exactly one row per test Id."
        )

    if expected_ids is not None and not df_test["Id"].equals(expected_ids):
        raise ValueError(
            "Id column order/values changed during cleaning - submission row order "
            "must match test.csv exactly."
        )

    nan_count = df_test[FEATURES].isnull().sum().sum()
    logging.info("Remaining NaN in features before prediction: %d", nan_count)
    if nan_count > 0:
        raise ValueError("Unresolved NaNs remain in test features - aborting.")

    # --- 4. Predict ---
    X_test = df_test[FEATURES]
    predictions = model.predict(X_test)
    predictions = np.clip(predictions, a_min=0, a_max=None)  # SalePrice can't be negative

    # --- 5. Save Submission ---
    submission = pd.DataFrame({"Id": df_test["Id"], "SalePrice": predictions})

    # Final guard: submission must exactly match raw test.csv row count before writing
    if len(submission) != expected_rows:
        raise ValueError(
            f"Submission has {len(submission)} rows but test.csv has {expected_rows} rows. "
            f"Aborting write to avoid an invalid Kaggle submission."
        )

    submission.to_csv(SUBMISSION_PATH, index=False)
    logging.info("Submission saved to %s (%d rows, matches test.csv).", SUBMISSION_PATH, len(submission))


if __name__ == "__main__":
    main()
