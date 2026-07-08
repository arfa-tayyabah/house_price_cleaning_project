# src/evaluate_model.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
CLEANED_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_house_prices.csv"

N_SPLITS = 5
RANDOM_STATE = 42


def rmsle_scorer():
    """
    Kaggle's House Prices competition is scored on RMSLE (RMSE of log-prices),
    not raw dollar RMSE. We train on log1p(SalePrice) directly so cross_val_score's
    'neg_mean_squared_error' scoring is already operating in log-space, and we
    just take the sqrt of the negated score.
    """
    return "neg_mean_squared_error"


def evaluate(df, features, target="SalePrice", model=None, label=""):
    """
    Run k-fold CV on log1p(target) using the given features/model.
    Returns (mean_rmsle, std_rmsle) across folds — much more reliable than a
    single train/test split.
    """
    df_clean = df[features + [target]].dropna()
    X = df_clean[features]
    y_log = np.log1p(df_clean[target])

    if model is None:
        model = LinearRegression()

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    neg_mse_scores = cross_val_score(model, X, y_log, cv=kf, scoring=rmsle_scorer())
    rmsle_scores = np.sqrt(-neg_mse_scores)

    logging.info(f"{label} — RMSLE: {rmsle_scores.mean():.4f} (+/- {rmsle_scores.std():.4f}) "
                 f"across {N_SPLITS} folds")
    return rmsle_scores.mean(), rmsle_scores.std()


def evaluate_raw():
    """Baseline: simple raw numeric features, no cleaning/engineering."""
    df = pd.read_csv(RAW_PATH)
    raw_features = ['LotArea', 'GrLivArea', 'BedroomAbvGr', 'YearBuilt']
    return evaluate(df, raw_features, model=LinearRegression(), label="RAW (LinearRegression)")


def evaluate_cleaned():
    """Cleaned + engineered features, with a regularized model to handle
    the multicollinearity introduced by Quality_Area / TotalSF overlap."""
    df = pd.read_csv(CLEANED_PATH)
    engineered_features = ['Quality_Area', 'TotalSF', 'TotalBath', 'HouseAge', 'HasGarage']
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 13))
    return evaluate(df, engineered_features, model=ridge, label="CLEANED (RidgeCV)")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MODEL EVALUATION: RAW vs CLEANED  (metric: RMSLE, 5-fold CV)")
    print("=" * 60)

    raw_mean, raw_std = evaluate_raw()
    clean_mean, clean_std = evaluate_cleaned()

    improvement = ((raw_mean - clean_mean) / raw_mean) * 100
    print("\n" + "=" * 60)
    print(f"IMPROVEMENT: {improvement:.1f}% lower RMSLE with the cleaned pipeline")
    print(f"   RAW     RMSLE: {raw_mean:.4f} +/- {raw_std:.4f}")
    print(f"   CLEANED RMSLE: {clean_mean:.4f} +/- {clean_std:.4f}")
    print("=" * 60)