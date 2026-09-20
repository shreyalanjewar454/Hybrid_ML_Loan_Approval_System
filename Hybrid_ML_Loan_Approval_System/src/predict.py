"""
predict.py
===========
Hybrid ML Loan Approval & Alternate Credit Assessment System

Implements:
  - Engine A: HybridStackingEnsemble
        Stacking Ensemble (XGBoost + CatBoost + LightGBM base learners +
        Logistic Regression meta-learner) for traditional, CIBIL-scored
        applicants. Gracefully falls back to scikit-learn's
        GradientBoostingClassifier / ExtraTreesClassifier / HistGradientBoosting
        as drop-in base learners if xgboost/catboost/lightgbm are not
        installed in the runtime environment, so the system NEVER fails
        to run end-to-end.
  - Engine B: ThinFileAlternateEngine
        Gradient-boosted classifier trained on UPI/cash-flow derived
        alternate-data features for applicants with no CIBIL history.
  - FraudAnomalyDetector
        Isolation-Forest based real-time anomaly / fraud filter that flags
        suspicious applications (e.g. artificial UPI turnover spikes,
        mismatched income-to-expense ratios).
  - HybridLoanPredictionSystem
        Top-level orchestrator used by the Streamlit app: trains (or loads)
        both engines + the anomaly detector, routes an applicant to the
        correct engine, and returns a fully structured prediction result
        (probability, risk grade, max eligible loan, fraud flag).

No placeholders. Fully implemented and runnable end-to-end even without
xgboost / catboost / lightgbm installed (falls back to sklearn boosting).
"""

from __future__ import annotations

import os
import warnings
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, Optional

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    StackingClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    IsolationForest,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

from src.data_preprocessing import (
    MODEL_DIR,
    get_or_create_traditional_dataset,
    get_or_create_thinfile_dataset,
    fit_traditional_preprocessor,
    fit_thinfile_preprocessor,
    load_traditional_preprocessor,
    load_thinfile_preprocessor,
    split_X_y,
    TRADITIONAL_PREPROCESSOR_PATH,
    THINFILE_PREPROCESSOR_PATH,
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

ENGINE_A_MODEL_PATH = os.path.join(MODEL_DIR, "engine_a_stacking_model.joblib")
ENGINE_B_MODEL_PATH = os.path.join(MODEL_DIR, "engine_b_thinfile_model.joblib")
FRAUD_MODEL_PATH = os.path.join(MODEL_DIR, "fraud_isolation_forest.joblib")
ENGINE_A_METRICS_PATH = os.path.join(MODEL_DIR, "engine_a_metrics.joblib")
ENGINE_B_METRICS_PATH = os.path.join(MODEL_DIR, "engine_b_metrics.joblib")


# ==========================================================================
# Base-learner resolution (real libraries preferred, sklearn fallback)
# ==========================================================================
def _resolve_base_learners(random_state: int = RANDOM_STATE):
    """Returns a list of (name, estimator) tuples for the stacking ensemble.
    Prefers XGBoost / CatBoost / LightGBM when available; otherwise falls
    back to functionally-equivalent scikit-learn boosted-tree models so the
    system remains 100% runnable in any environment."""
    learners = []

    try:
        from xgboost import XGBClassifier

        learners.append(
            (
                "xgboost",
                XGBClassifier(
                    n_estimators=250,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    eval_metric="logloss",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            )
        )
    except ImportError:
        learners.append(
            (
                "xgboost_fallback_hgb",
                HistGradientBoostingClassifier(
                    max_iter=250, learning_rate=0.05, max_depth=4, random_state=random_state
                ),
            )
        )

    try:
        from catboost import CatBoostClassifier

        learners.append(
            (
                "catboost",
                CatBoostClassifier(
                    iterations=250,
                    depth=6,
                    learning_rate=0.05,
                    verbose=False,
                    random_state=random_state,
                ),
            )
        )
    except ImportError:
        learners.append(
            (
                "catboost_fallback_gbc",
                GradientBoostingClassifier(
                    n_estimators=200, max_depth=3, learning_rate=0.05, random_state=random_state
                ),
            )
        )

    try:
        from lightgbm import LGBMClassifier

        learners.append(
            (
                "lightgbm",
                LGBMClassifier(
                    n_estimators=250,
                    max_depth=-1,
                    num_leaves=31,
                    learning_rate=0.05,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    random_state=random_state,
                    verbosity=-1,
                ),
            )
        )
    except ImportError:
        learners.append(
            (
                "lightgbm_fallback_extratrees",
                ExtraTreesClassifier(n_estimators=300, max_depth=8, random_state=random_state, n_jobs=-1),
            )
        )

    return learners


# ==========================================================================
# Engine A: Standard Credit Bureau (CIBIL) Stacking Ensemble
# ==========================================================================
@dataclass
class HybridStackingEnsemble:
    """Stacking Ensemble (XGBoost + CatBoost + LightGBM base learners with a
    Logistic-Regression meta-learner) for traditional CIBIL-scored
    applicants. Implements the standard sklearn estimator interface."""

    random_state: int = RANDOM_STATE
    model: Optional[StackingClassifier] = None

    def build(self) -> "HybridStackingEnsemble":
        base_learners = _resolve_base_learners(self.random_state)
        meta_learner = LogisticRegression(max_iter=1000, random_state=self.random_state)
        self.model = StackingClassifier(
            estimators=base_learners,
            final_estimator=meta_learner,
            stack_method="predict_proba",
            passthrough=False,
            cv=5,
            n_jobs=-1,
        )
        return self

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HybridStackingEnsemble":
        if self.model is None:
            self.build()
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


# ==========================================================================
# Engine B: Thin-File / Alternate-Data Credit Engine
# ==========================================================================
@dataclass
class ThinFileAlternateEngine:
    """Gradient-boosted classifier trained purely on alternate/behavioural
    data (UPI transaction logs, cash-flow consistency, turnover, utility &
    rent payment reliability) for applicants with NO CIBIL history."""

    random_state: int = RANDOM_STATE
    model: Optional[Any] = None

    def build(self) -> "ThinFileAlternateEngine":
        try:
            from lightgbm import LGBMClassifier

            self.model = LGBMClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.04,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=self.random_state,
                verbosity=-1,
            )
        except ImportError:
            self.model = GradientBoostingClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.04, random_state=self.random_state
            )
        return self

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ThinFileAlternateEngine":
        if self.model is None:
            self.build()
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


# ==========================================================================
# Fraud / Anomaly Detector (Isolation Forest)
# ==========================================================================
@dataclass
class FraudAnomalyDetector:
    """Real-time fraud & anomaly filter using Isolation Forest to flag
    suspicious applications: e.g. sudden artificial UPI turnover spikes,
    mismatched income-to-expense ratios, or statistically outlying
    combinations of financial features."""

    contamination: float = 0.06
    random_state: int = RANDOM_STATE
    model: Optional[IsolationForest] = None

    def build(self) -> "FraudAnomalyDetector":
        self.model = IsolationForest(
            n_estimators=300,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        return self

    def fit(self, X: np.ndarray) -> "FraudAnomalyDetector":
        if self.model is None:
            self.build()
        self.model.fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Returns anomaly scores: lower (more negative) = more anomalous."""
        return self.model.decision_function(X)

    def is_anomalous(self, X: np.ndarray) -> np.ndarray:
        """Returns True for flagged / suspicious rows."""
        return self.model.predict(X) == -1

    def anomaly_probability(self, X: np.ndarray) -> np.ndarray:
        """Converts the raw decision_function score into an intuitive
        0-1 'fraud likelihood' via a min-max-normalised, inverted sigmoid."""
        raw = self.score(X)
        squashed = 1 / (1 + np.exp(raw * 4))  # more negative raw -> closer to 1
        return np.clip(squashed, 0.0, 1.0)


# ==========================================================================
# Business logic helpers: Risk Grade & Max Eligible Loan
# ==========================================================================
def probability_to_risk_grade(probability_approve: float) -> str:
    if probability_approve >= 0.72:
        return "Low"
    elif probability_approve >= 0.45:
        return "Medium"
    else:
        return "High"


def compute_max_eligible_loan_traditional(
    annual_income: float,
    existing_emi_monthly: float,
    probability_approve: float,
    cibil_score: float,
    loan_tenure_months: int,
) -> float:
    """FOIR (Fixed Obligation to Income Ratio) style eligibility: bank
    generally allows up to ~50% of monthly income towards all EMIs."""
    monthly_income = annual_income / 12.0
    max_total_emi = 0.50 * monthly_income
    available_emi_capacity = max(max_total_emi - existing_emi_monthly, 0.0)

    # Approx affordable principal given a flat annualised rate assumption
    annual_rate = 0.11 if cibil_score >= 750 else (0.14 if cibil_score >= 650 else 0.18)
    monthly_rate = annual_rate / 12.0
    n = max(loan_tenure_months, 1)
    if monthly_rate > 0:
        affordable_principal = available_emi_capacity * ((1 - (1 + monthly_rate) ** (-n)) / monthly_rate)
    else:
        affordable_principal = available_emi_capacity * n

    risk_multiplier = np.clip(probability_approve, 0.05, 1.0)
    max_loan = affordable_principal * risk_multiplier
    return float(np.clip(round(max_loan, -3), 0, 10_000_000))


def compute_max_eligible_loan_thinfile(
    monthly_upi_turnover: float,
    alternate_credit_score: float,
    probability_approve: float,
) -> float:
    """Thin-file eligibility is anchored to monthly turnover (as a proxy for
    repayment capacity) and scaled by the alternate credit score band and
    model-estimated approval probability."""
    score_multiplier = np.clip(alternate_credit_score / 800.0, 0.1, 1.0)
    base_capacity = monthly_upi_turnover * 4.0  # up to ~4 months turnover as principal ceiling
    risk_multiplier = np.clip(probability_approve, 0.05, 1.0)
    max_loan = base_capacity * score_multiplier * risk_multiplier
    return float(np.clip(round(max_loan, -3), 0, 1_500_000))


# ==========================================================================
# Top-level orchestrator used by the Streamlit application
# ==========================================================================
class HybridLoanPredictionSystem:
    """Loads (or trains, on first run) both credit engines and the fraud
    detector, and exposes a single clean interface used throughout the
    Streamlit app: `predict_traditional(...)` and `predict_thinfile(...)`."""

    def __init__(self, auto_load: bool = True):
        self.engine_a: Optional[HybridStackingEnsemble] = None
        self.engine_b: Optional[ThinFileAlternateEngine] = None
        self.fraud_detector_a: Optional[FraudAnomalyDetector] = None
        self.fraud_detector_b: Optional[FraudAnomalyDetector] = None
        self.traditional_preprocessor = None
        self.thinfile_preprocessor = None
        self.metrics_a: Dict[str, float] = {}
        self.metrics_b: Dict[str, float] = {}
        if auto_load:
            self.load_or_train_all()

    # ------------------------------------------------------------------
    def _all_artifacts_exist(self) -> bool:
        return all(
            os.path.exists(p)
            for p in [
                ENGINE_A_MODEL_PATH,
                ENGINE_B_MODEL_PATH,
                FRAUD_MODEL_PATH,
                TRADITIONAL_PREPROCESSOR_PATH,
                THINFILE_PREPROCESSOR_PATH,
            ]
        )

    def load_or_train_all(self, force_retrain: bool = False) -> None:
        if not force_retrain and self._all_artifacts_exist():
            self._load_all()
        else:
            self._train_all()

    # ------------------------------------------------------------------
    def _load_all(self) -> None:
        self.engine_a = joblib.load(ENGINE_A_MODEL_PATH)
        self.engine_b = joblib.load(ENGINE_B_MODEL_PATH)
        fraud_bundle = joblib.load(FRAUD_MODEL_PATH)
        self.fraud_detector_a = fraud_bundle["engine_a"]
        self.fraud_detector_b = fraud_bundle["engine_b"]
        self.traditional_preprocessor = load_traditional_preprocessor()
        self.thinfile_preprocessor = load_thinfile_preprocessor()
        if os.path.exists(ENGINE_A_METRICS_PATH):
            self.metrics_a = joblib.load(ENGINE_A_METRICS_PATH)
        if os.path.exists(ENGINE_B_METRICS_PATH):
            self.metrics_b = joblib.load(ENGINE_B_METRICS_PATH)

    # ------------------------------------------------------------------
    def _train_all(self) -> None:
        # ---------------- Engine A: Traditional -------------------------
        df_a = get_or_create_traditional_dataset()
        X_a_raw, y_a = split_X_y(df_a, "approved")
        self.traditional_preprocessor = fit_traditional_preprocessor(X_a_raw)
        X_a = self.traditional_preprocessor.transform(X_a_raw)

        X_a_train, X_a_test, y_a_train, y_a_test = train_test_split(
            X_a, y_a, test_size=0.2, random_state=RANDOM_STATE, stratify=y_a
        )
        self.engine_a = HybridStackingEnsemble(random_state=RANDOM_STATE).build()
        self.engine_a.fit(X_a_train, y_a_train)
        proba_a = self.engine_a.predict_proba(X_a_test)[:, 1]
        pred_a = (proba_a >= 0.5).astype(int)
        self.metrics_a = {
            "roc_auc": float(roc_auc_score(y_a_test, proba_a)),
            "accuracy": float(accuracy_score(y_a_test, pred_a)),
            "n_train": int(len(X_a_train)),
            "n_test": int(len(X_a_test)),
        }
        joblib.dump(self.engine_a, ENGINE_A_MODEL_PATH)
        joblib.dump(self.metrics_a, ENGINE_A_METRICS_PATH)

        self.fraud_detector_a = FraudAnomalyDetector(contamination=0.06, random_state=RANDOM_STATE).build()
        self.fraud_detector_a.fit(X_a)

        # ---------------- Engine B: Thin-File ----------------------------
        df_b = get_or_create_thinfile_dataset()
        X_b_raw, y_b = split_X_y(df_b, "approved")
        self.thinfile_preprocessor = fit_thinfile_preprocessor(X_b_raw)
        X_b = self.thinfile_preprocessor.transform(X_b_raw)

        X_b_train, X_b_test, y_b_train, y_b_test = train_test_split(
            X_b, y_b, test_size=0.2, random_state=RANDOM_STATE, stratify=y_b
        )
        self.engine_b = ThinFileAlternateEngine(random_state=RANDOM_STATE).build()
        self.engine_b.fit(X_b_train, y_b_train)
        proba_b = self.engine_b.predict_proba(X_b_test)[:, 1]
        pred_b = (proba_b >= 0.5).astype(int)
        self.metrics_b = {
            "roc_auc": float(roc_auc_score(y_b_test, proba_b)),
            "accuracy": float(accuracy_score(y_b_test, pred_b)),
            "n_train": int(len(X_b_train)),
            "n_test": int(len(X_b_test)),
        }
        joblib.dump(self.engine_b, ENGINE_B_MODEL_PATH)
        joblib.dump(self.metrics_b, ENGINE_B_METRICS_PATH)

        self.fraud_detector_b = FraudAnomalyDetector(contamination=0.06, random_state=RANDOM_STATE).build()
        self.fraud_detector_b.fit(X_b)

        joblib.dump(
            {"engine_a": self.fraud_detector_a, "engine_b": self.fraud_detector_b}, FRAUD_MODEL_PATH
        )

    # ------------------------------------------------------------------
    def predict_traditional(self, applicant: Dict[str, Any]) -> Dict[str, Any]:
        df_input = pd.DataFrame([applicant])
        X = self.traditional_preprocessor.transform(df_input)

        proba_approve = float(self.engine_a.predict_proba(X)[0, 1])
        risk_grade = probability_to_risk_grade(proba_approve)
        max_loan = compute_max_eligible_loan_traditional(
            annual_income=applicant["annual_income"],
            existing_emi_monthly=applicant["existing_emi_monthly"],
            probability_approve=proba_approve,
            cibil_score=applicant["cibil_score"],
            loan_tenure_months=applicant["loan_tenure_months"],
        )
        fraud_prob = float(self.fraud_detector_a.anomaly_probability(X)[0])
        is_flagged = bool(self.fraud_detector_a.is_anomalous(X)[0])

        decision = "Approved" if (proba_approve >= 0.5 and not is_flagged) else "Rejected"
        if is_flagged:
            decision = "Manual Review Required"

        return {
            "engine": "Engine A - Standard Credit Bureau",
            "approval_probability": round(proba_approve * 100, 2),
            "risk_grade": risk_grade,
            "max_eligible_loan": max_loan,
            "decision": decision,
            "fraud_probability": round(fraud_prob * 100, 2),
            "is_fraud_flagged": is_flagged,
            "processed_features": X,
            "feature_names": self.traditional_preprocessor.get_feature_names(),
            "raw_input": applicant,
        }

    # ------------------------------------------------------------------
    def predict_thinfile(self, applicant: Dict[str, Any]) -> Dict[str, Any]:
        df_input = pd.DataFrame([applicant])
        X = self.thinfile_preprocessor.transform(df_input)

        proba_approve = float(self.engine_b.predict_proba(X)[0, 1])
        risk_grade = probability_to_risk_grade(proba_approve)
        max_loan = compute_max_eligible_loan_thinfile(
            monthly_upi_turnover=applicant["monthly_upi_turnover"],
            alternate_credit_score=applicant["alternate_credit_score"],
            probability_approve=proba_approve,
        )
        fraud_prob = float(self.fraud_detector_b.anomaly_probability(X)[0])
        is_flagged = bool(self.fraud_detector_b.is_anomalous(X)[0])

        decision = "Approved" if (proba_approve >= 0.5 and not is_flagged) else "Rejected"
        if is_flagged:
            decision = "Manual Review Required"

        return {
            "engine": "Engine B - Thin-File Alternate Credit",
            "approval_probability": round(proba_approve * 100, 2),
            "risk_grade": risk_grade,
            "max_eligible_loan": max_loan,
            "decision": decision,
            "fraud_probability": round(fraud_prob * 100, 2),
            "is_fraud_flagged": is_flagged,
            "processed_features": X,
            "feature_names": self.thinfile_preprocessor.get_feature_names(),
            "raw_input": applicant,
        }
