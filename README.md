# 🏠 House Price Prediction — Data Cleaning & Modeling Pipeline

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.0.3-150458?logo=pandas)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-1.3.0-F7931E?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

An end-to-end data cleaning, feature engineering, and evaluation pipeline for the Kaggle [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) competition.

---

## 📖 Overview

Housing prices depend on dozens of interacting factors — location, size, age, and quality — but raw data is rarely ready to model. This project turns the raw Kaggle dataset into a clean, leakage-free, model-ready dataset, and quantifies exactly how much that cleaning is worth.

The pipeline:

- Profiles the data for missing values, inconsistent codes, and known data-entry errors
- Applies domain-specific fixes (e.g. `NaN` in garage columns means *"no garage,"* not *"unknown"*)
- Imputes remaining missing values (median for numeric columns, mode for categorical)
- Removes the two documented `GrLivArea` outliers flagged by the dataset's author
- Encodes ordinal quality columns (`Po` < `Fa` < `TA` < `Gd` < `Ex`) numerically
- Engineers 7 high-signal features, including `Quality_Area` (correlation with `SalePrice`: 0.86)
- Persists imputation statistics so the exact same transformation can be applied to `test.csv` for a real Kaggle submission — no re-fitting on unseen data
- Evaluates impact with **5-fold cross-validation on RMSLE**, the competition's official metric

---

## 📊 Key Results

| Metric | Raw Data | Cleaned + Engineered |
|---|---|---|
| Rows | 1,460 | 1,458 *(2 outliers removed)* |
| Columns | 81 | 76 |
| Missing values | ~2,847 | 0 |
| Model | Linear Regression | RidgeCV (L2-regularized) |
| RMSLE (5-fold CV) | 0.2152 ± 0.0352 | 0.1580 ± 0.0095 |
| **Improvement** | — | **26.6% lower RMSLE** |

Cross-validated standard deviation also drops considerably (0.035 → 0.010), meaning the cleaned pipeline isn't just more accurate on average — it's a more *stable* estimate.

---

## 🧹 Cleaning Steps

| Step | What it does | Why |
|---|---|---|
| 1. Profiling | Scans all 81 columns for missing values | Establishes where intervention is needed |
| 2. Drop high-missing columns | Removes columns >50% missing | Low-information columns add noise and overfitting risk |
| 3. Garage fix | Fills garage text columns with `"No_Garage"`, year with `0` | `NaN` here is a valid category, not missing data |
| 4. Ordinal encoding | Maps quality/condition columns to a 0–5 numeric scale | Preserves the natural ordering that one-hot encoding would discard |
| 5. Outlier removal | Drops the 2 documented `GrLivArea` outliers (train only) | Known data errors that distort linear model fits |
| 6. Imputation | Numeric → median, categorical → mode; statistics fit on train, saved, and reapplied to test | Avoids leakage and keeps train/test consistent |
| 7. Feature engineering | Adds 7 new features (below) | Encodes domain knowledge the raw columns don't capture directly |

---

## 🛠️ Engineered Features

| Feature | Formula | Correlation with `SalePrice` |
|---|---|---|
| `Quality_Area` | `OverallQual × TotalSF` | **0.86** *(strongest single feature)* |
| `TotalSF` | `TotalBsmtSF + 1stFlrSF + 2ndFlrSF` | 0.78 |
| `TotalBath` | `FullBath + 0.5·HalfBath + BsmtFullBath + 0.5·BsmtHalfBath` | 0.63 |
| `HouseAge` | `YrSold − YearBuilt` (clipped at 0) | -0.52 |
| `RemodAge` | `YrSold − YearRemodAdd` (clipped at 0) | -0.51 |
| `HasGarage` | `GarageArea > 0` | 0.24 |
| `TotalPorchSF` | `OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch` | 0.20 |

---

## 🔬 Evaluation Methodology

| Choice | Reasoning |
|---|---|
| **RMSLE**, not raw RMSE | RMSLE is the Kaggle competition's official metric — it penalizes underestimation more than overestimation and handles the right-skewed price distribution |
| **5-fold cross-validation**, not a single split | A single `train_test_split` gives one noisy point estimate; CV gives a mean ± standard deviation, showing how stable the improvement actually is |
| **RidgeCV** on engineered features | `Quality_Area` overlaps with `TotalSF` by construction, introducing multicollinearity; L2 regularization keeps coefficients stable |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Git

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/house_price_cleaning_project.git
cd house_price_cleaning_project

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add the Kaggle data
#    Place train.csv (and optionally test.csv) in data/raw/

# 5. Run the cleaning pipeline
python src/clean_pipeline.py

# 6. Evaluate raw vs. cleaned performance
python src/evaluate_model.py
```

### Expected output

```
MODEL EVALUATION: RAW vs CLEANED  (metric: RMSLE, 5-fold CV)
RAW (LinearRegression) — RMSLE: 0.2152 (+/- 0.0352) across 5 folds
CLEANED (RidgeCV)      — RMSLE: 0.1580 (+/- 0.0095) across 5 folds
IMPROVEMENT: 26.6% lower RMSLE with the cleaned pipeline
```

---

## 🏗️ Project Structure

```
house_price_cleaning_project/
├── data/
│   ├── raw/
│   │   ├── train.csv                  # Kaggle training data (gitignored)
│   │   └── test.csv                   # Kaggle test data (gitignored)
│   └── processed/
│       ├── cleaned_house_prices.csv   # Generated by clean_pipeline.py
│       ├── cleaned_test.csv           # Test set, same transforms applied
│       └── imputation_stats.json      # Stats fit on train, reused on test
├── notebooks/
│   └── 01_eda_and_cleaning.ipynb      # Exploratory analysis with visualizations
├── src/
│   ├── __init__.py
│   ├── clean_pipeline.py              # Cleaning & feature engineering
│   └── evaluate_model.py              # 5-fold CV: raw vs. cleaned (RMSLE)
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🤔 Design Decisions

**Why drop columns with >50% missing?**
Imputing across that much missingness is statistically unreliable, and the columns tend to contribute more noise than signal.

**Why median over mean for numeric imputation?**
The dataset contains genuine high-end outliers (mansions, luxury lots); the median is robust to them where the mean isn't.

**Why remove the `GrLivArea` outliers?**
These two points are documented by the dataset's original author as data anomalies, not genuine market signal — leaving them in disproportionately affects a linear model's fit.

**Why RidgeCV instead of plain Linear Regression on the engineered set?**
`Quality_Area` is constructed from `TotalSF`, so the two are collinear by design. L2 regularization stabilizes the coefficient estimates without discarding either feature.

**Why fit imputation statistics on train and reuse them on test?**
Recomputing medians/modes on the test set — even for imputation — is a form of data leakage. Statistics are fit once, saved to `imputation_stats.json`, and applied identically to both sets.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 🙏 Acknowledgements

- Dataset: Kaggle's [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- Outlier guidance: Dean De Cock, *Ames Housing Dataset* documentation
- [scikit-learn documentation](https://scikit-learn.org/) for cross-validation and regularization tools
