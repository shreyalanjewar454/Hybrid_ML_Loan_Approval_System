"""
data_preprocessing.py
======================
Hybrid ML Loan Approval & Alternate Credit Assessment System

Handles:
  - Synthetic data generation for BOTH engines (used to bootstrap/train
    the models on first run, and to power the live Admin/Analytics demo).
  - Preprocessing pipelines (encoding, scaling, imputation) for:
      Engine A -> Standard Credit Bureau (CIBIL-based) applicants
      Engine B -> Thin-File / Alternate-Data (UPI & Cashflow) applicants
  - Persistence helpers (joblib) so the fitted preprocessors can be
    reused at inference time without refitting.

No placeholders. Every function is fully implemented and runnable.
"""

from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Tuple, List

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

TRADITIONAL_RAW_PATH = os.path.join(DATA_DIR, "traditional_applicants.csv")
THINFILE_RAW_PATH = os.path.join(DATA_DIR, "thinfile_applicants.csv")
APPLICATIONS_LOG_PATH = os.path.join(DATA_DIR, "applications_log.csv")

TRADITIONAL_PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "traditional_preprocessor.joblib")
THINFILE_PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "thinfile_preprocessor.joblib")

RANDOM_STATE = 42

# --------------------------------------------------------------------------
# Feature schema definitions
# --------------------------------------------------------------------------
TRADITIONAL_NUMERIC_FEATURES = [
    "age",
    "annual_income",
    "years_employed",
    "cibil_score",
    "existing_loans_count",
    "existing_emi_monthly",
    "requested_loan_amount",
    "loan_tenure_months",
    "dependents",
    "dti_ratio",
    "savings_balance",
]
TRADITIONAL_CATEGORICAL_FEATURES = ["employment_type", "previous_default"]
TRADITIONAL_TARGET = "approved"

THINFILE_NUMERIC_FEATURES = [
    "age",
    "monthly_upi_txn_count",
    "monthly_upi_turnover",
    "avg_daily_balance",
    "cashflow_volatility",
    "utility_payment_reliability",
    "rent_payment_reliability",
    "months_of_transaction_history",
    "requested_loan_amount",
    "alternate_credit_score",
]
THINFILE_CATEGORICAL_FEATURES = ["occupation_type"]
THINFILE_TARGET = "approved"


# --------------------------------------------------------------------------
# Synthetic data generation - Engine A (Traditional / CIBIL)
# --------------------------------------------------------------------------
def generate_synthetic_traditional_data(n: int = 4000, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Generates a realistic synthetic dataset of traditional (CIBIL-scored) loan applicants."""
    rng = np.random.default_rng(random_state)

    age = rng.integers(21, 65, size=n)
    employment_type = rng.choice(
        ["Salaried", "Self-Employed", "Business Owner"], size=n, p=[0.55, 0.25, 0.20]
    )
    base_income = rng.normal(650000, 280000, size=n).clip(120000, 4500000)
    annual_income = base_income.round(-3)

    years_employed = rng.integers(0, 30, size=n)
    cibil_score = rng.normal(700, 95, size=n).clip(300, 900).round().astype(int)

    existing_loans_count = rng.poisson(0.9, size=n).clip(0, 6)
    existing_emi_monthly = (existing_loans_count * rng.normal(9000, 4000, size=n)).clip(0, None).round(-2)

    requested_loan_amount = rng.normal(900000, 650000, size=n).clip(50000, 8000000).round(-3)
    loan_tenure_months = rng.choice([12, 24, 36, 48, 60, 84, 120, 180, 240], size=n)

    dependents = rng.integers(0, 5, size=n)
    monthly_income = annual_income / 12.0
    dti_ratio = ((existing_emi_monthly + (requested_loan_amount / loan_tenure_months)) / monthly_income).clip(0, 3)

    savings_balance = rng.normal(300000, 250000, size=n).clip(0, None).round(-3)
    previous_default = rng.choice([0, 1], size=n, p=[0.88, 0.12])

    # Probabilistic approval logic (ground truth generator) -----------------
    score = (
        0.0032 * (cibil_score - 650)
        + 0.0000009 * (annual_income - 500000)
        - 1.15 * dti_ratio
        - 0.85 * previous_default
        - 0.10 * existing_loans_count
        + 0.05 * (years_employed.clip(0, 15))
        + 0.0000006 * savings_balance
        - 0.004 * dependents
    )
    prob_approve = 1 / (1 + np.exp(-score))
    noise = rng.normal(0, 0.07, size=n)
    approved = ((prob_approve + noise) > 0.5).astype(int)

    df = pd.DataFrame(
        {
            "applicant_id": [f"TRAD-{i:06d}" for i in range(n)],
            "age": age,
            "employment_type": employment_type,
            "annual_income": annual_income,
            "years_employed": years_employed,
            "cibil_score": cibil_score,
            "existing_loans_count": existing_loans_count,
            "existing_emi_monthly": existing_emi_monthly,
            "requested_loan_amount": requested_loan_amount,
            "loan_tenure_months": loan_tenure_months,
            "dependents": dependents,
            "dti_ratio": dti_ratio.round(3),
            "previous_default": previous_default,
            "savings_balance": savings_balance,
            "approved": approved,
        }
    )
    return df


# --------------------------------------------------------------------------
# Synthetic data generation - Engine B (Thin-File / Alternate Data)
# --------------------------------------------------------------------------
def generate_synthetic_thinfile_data(n: int = 3000, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Generates a realistic synthetic dataset of thin-file / unbanked applicants
    scored purely on UPI transaction & cashflow behaviour (no CIBIL)."""
    rng = np.random.default_rng(random_state + 7)

    age = rng.integers(19, 60, size=n)
    occupation_type = rng.choice(
        ["Vendor", "Shopkeeper", "Gig Worker", "Freelancer"], size=n, p=[0.30, 0.30, 0.25, 0.15]
    )

    monthly_upi_txn_count = rng.integers(15, 800, size=n)
    monthly_upi_turnover = rng.normal(85000, 55000, size=n).clip(5000, 600000).round(-2)
    avg_daily_balance = rng.normal(9000, 8000, size=n).clip(200, None).round(-1)

    # volatility = coefficient of variation of daily turnover (lower = more stable)
    cashflow_volatility = rng.beta(2, 6, size=n).round(3)  # 0 (stable) - 1 (volatile)

    utility_payment_reliability = rng.beta(6, 2, size=n).round(3)  # 0-1, higher better
    rent_payment_reliability = rng.beta(5, 2.5, size=n).round(3)

    months_of_transaction_history = rng.integers(3, 60, size=n)
    requested_loan_amount = rng.normal(120000, 90000, size=n).clip(5000, 1000000).round(-3)

    # Alternate credit score computed via the dedicated scoring engine ------
    from src.alternate_credit_scoring import AlternateCreditScorer

    scorer = AlternateCreditScorer()
    alternate_credit_score = np.array(
        [
            scorer.compute_final_score(
                monthly_turnover=monthly_upi_turnover[i],
                cashflow_volatility=cashflow_volatility[i],
                utility_reliability=utility_payment_reliability[i],
                rent_reliability=rent_payment_reliability[i],
                months_history=months_of_transaction_history[i],
                txn_count=monthly_upi_txn_count[i],
                avg_daily_balance=avg_daily_balance[i],
            )["final_score"]
            for i in range(n)
        ]
    )

    score = (
        0.0055 * (alternate_credit_score - 400)
        - 1.8 * cashflow_volatility
        + 0.000004 * (monthly_upi_turnover - requested_loan_amount / 6)
        + 0.02 * (months_of_transaction_history.clip(0, 36))
    )
    prob_approve = 1 / (1 + np.exp(-score))
    noise = rng.normal(0, 0.08, size=n)
    approved = ((prob_approve + noise) > 0.5).astype(int)

    df = pd.DataFrame(
        {
            "applicant_id": [f"THIN-{i:06d}" for i in range(n)],
            "age": age,
            "occupation_type": occupation_type,
            "monthly_upi_txn_count": monthly_upi_txn_count,
            "monthly_upi_turnover": monthly_upi_turnover,
            "avg_daily_balance": avg_daily_balance,
            "cashflow_volatility": cashflow_volatility,
            "utility_payment_reliability": utility_payment_reliability,
            "rent_payment_reliability": rent_payment_reliability,
            "months_of_transaction_history": months_of_transaction_history,
            "requested_loan_amount": requested_loan_amount,
            "alternate_credit_score": alternate_credit_score.round().astype(int),
            "approved": approved,
        }
    )
    return df


# --------------------------------------------------------------------------
# Preprocessing pipeline wrappers
# --------------------------------------------------------------------------
@dataclass
class FittedPreprocessor:
    """Thin wrapper bundling a fitted ColumnTransformer with its schema,
    so downstream modules (predict.py, xai_explainer.py) can recover
    human-readable feature names after one-hot encoding."""

    column_transformer: ColumnTransformer
    numeric_features: List[str]
    categorical_features: List[str]
    output_feature_names: List[str] = field(default_factory=list)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.column_transformer.transform(df)

    def get_feature_names(self) -> List[str]:
        return self.output_feature_names


def _build_column_transformer(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def fit_traditional_preprocessor(df: pd.DataFrame) -> FittedPreprocessor:
    ct = _build_column_transformer(TRADITIONAL_NUMERIC_FEATURES, TRADITIONAL_CATEGORICAL_FEATURES)
    ct.fit(df)
    cat_names = list(ct.named_transformers_["cat"]["onehot"].get_feature_names_out(TRADITIONAL_CATEGORICAL_FEATURES))
    output_names = TRADITIONAL_NUMERIC_FEATURES + cat_names
    fp = FittedPreprocessor(ct, TRADITIONAL_NUMERIC_FEATURES, TRADITIONAL_CATEGORICAL_FEATURES, output_names)
    joblib.dump(fp, TRADITIONAL_PREPROCESSOR_PATH)
    return fp


def fit_thinfile_preprocessor(df: pd.DataFrame) -> FittedPreprocessor:
    ct = _build_column_transformer(THINFILE_NUMERIC_FEATURES, THINFILE_CATEGORICAL_FEATURES)
    ct.fit(df)
    cat_names = list(ct.named_transformers_["cat"]["onehot"].get_feature_names_out(THINFILE_CATEGORICAL_FEATURES))
    output_names = THINFILE_NUMERIC_FEATURES + cat_names
    fp = FittedPreprocessor(ct, THINFILE_NUMERIC_FEATURES, THINFILE_CATEGORICAL_FEATURES, output_names)
    joblib.dump(fp, THINFILE_PREPROCESSOR_PATH)
    return fp


def load_traditional_preprocessor() -> FittedPreprocessor:
    return joblib.load(TRADITIONAL_PREPROCESSOR_PATH)


def load_thinfile_preprocessor() -> FittedPreprocessor:
    return joblib.load(THINFILE_PREPROCESSOR_PATH)


def get_or_create_traditional_dataset(n: int = 4000, force: bool = False) -> pd.DataFrame:
    if not force and os.path.exists(TRADITIONAL_RAW_PATH):
        return pd.read_csv(TRADITIONAL_RAW_PATH)
    df = generate_synthetic_traditional_data(n)
    df.to_csv(TRADITIONAL_RAW_PATH, index=False)
    return df


def get_or_create_thinfile_dataset(n: int = 3000, force: bool = False) -> pd.DataFrame:
    if not force and os.path.exists(THINFILE_RAW_PATH):
        return pd.read_csv(THINFILE_RAW_PATH)
    df = generate_synthetic_thinfile_data(n)
    df.to_csv(THINFILE_RAW_PATH, index=False)
    return df


def split_X_y(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target])
    if "applicant_id" in X.columns:
        X = X.drop(columns=["applicant_id"])
    y = df[target]
    return X, y


def append_application_log(record: dict) -> None:
    """Appends a single processed application (with prediction results) to the
    persistent applications log used by the Admin Dashboard & Analytics pages."""
    df_row = pd.DataFrame([record])
    if os.path.exists(APPLICATIONS_LOG_PATH):
        df_row.to_csv(APPLICATIONS_LOG_PATH, mode="a", header=False, index=False)
    else:
        df_row.to_csv(APPLICATIONS_LOG_PATH, mode="w", header=True, index=False)


def load_application_log() -> pd.DataFrame:
    if os.path.exists(APPLICATIONS_LOG_PATH):
        return pd.read_csv(APPLICATIONS_LOG_PATH)
    return pd.DataFrame()
