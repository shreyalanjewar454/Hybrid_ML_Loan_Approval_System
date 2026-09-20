"""
xai_explainer.py
==================
Hybrid ML Loan Approval & Alternate Credit Assessment System

Explainable AI (XAI) Module.

Strategy
--------
The Stacking Ensemble (Engine A) and the Thin-File Engine (Engine B) are
both potentially heterogeneous, multi-model estimators, so a single
tree-specific SHAP TreeExplainer cannot cleanly explain either end-to-end.
This module therefore:

  1. Tries to use the *real* `shap` library (`shap.Explainer`, model-agnostic
     `Permutation` algorithm) when it is installed, which produces true
     Shapley-value-based additive explanations.
  2. Falls back automatically -- with IDENTICAL output schema -- to a
     hand-rolled, dependency-free "Simulated SHAP" engine that estimates
     additive per-feature contributions via sampling-based marginal
     contribution averaging (a simplified Shapley-value estimator, in the
     same spirit as `shap.SamplingExplainer` / `shap.KernelExplainer`),
     evaluated against a background reference set. This guarantees the
     system is 100% runnable even where `shap` cannot be installed.

Both paths produce a per-feature "contribution" (in probability-points,
additive, summing to `prediction - base_value`) that the Streamlit app
renders as a SHAP-style waterfall / bar chart.

No placeholders. Fully implemented and runnable end-to-end.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any, Callable

RANDOM_STATE = 42

# --------------------------------------------------------------------------
# Human-readable labels for raw (post one-hot-encoding) feature names,
# used so the XAI narrative reads like a real credit-decision explanation.
# --------------------------------------------------------------------------
FEATURE_DISPLAY_NAMES = {
    # Engine A (traditional)
    "age": "Applicant Age",
    "annual_income": "Annual Income",
    "years_employed": "Years of Employment",
    "cibil_score": "CIBIL Score",
    "existing_loans_count": "Number of Existing Loans",
    "existing_emi_monthly": "Existing Monthly EMI Burden",
    "requested_loan_amount": "Requested Loan Amount",
    "loan_tenure_months": "Loan Tenure",
    "dependents": "Number of Dependents",
    "dti_ratio": "Debt-to-Income Ratio",
    "savings_balance": "Savings Balance",
    # Engine B (thin-file)
    "monthly_upi_txn_count": "Monthly UPI Transaction Count",
    "monthly_upi_turnover": "Monthly UPI Turnover",
    "avg_daily_balance": "Average Daily Balance",
    "cashflow_volatility": "Cash-Flow Volatility",
    "utility_payment_reliability": "Utility Payment Reliability",
    "rent_payment_reliability": "Rent Payment Reliability",
    "months_of_transaction_history": "Transaction History Depth",
    "alternate_credit_score": "Alternate Credit Score",
}


def _humanize(raw_feature_name: str) -> str:
    """Maps an (possibly one-hot-encoded) feature name like
    'employment_type_Salaried' to a readable label 'Employment Type: Salaried'."""
    if raw_feature_name in FEATURE_DISPLAY_NAMES:
        return FEATURE_DISPLAY_NAMES[raw_feature_name]
    for base_col in ["employment_type", "previous_default", "occupation_type"]:
        if raw_feature_name.startswith(base_col + "_"):
            suffix = raw_feature_name[len(base_col) + 1 :]
            pretty_base = base_col.replace("_", " ").title()
            return f"{pretty_base}: {suffix}"
    return raw_feature_name.replace("_", " ").title()


@dataclass
class XAIExplainer:
    """Model-agnostic explainability engine producing additive, SHAP-style
    per-feature contribution values for a single prediction."""

    n_samples: int = 120
    random_state: int = RANDOM_STATE

    # ------------------------------------------------------------------
    def explain_instance(
        self,
        predict_proba_fn: Callable[[np.ndarray], np.ndarray],
        instance: np.ndarray,
        background: np.ndarray,
        feature_names: List[str],
        top_k: int = 8,
    ) -> Dict[str, Any]:
        """Explains a single prediction (instance: shape (1, n_features)).

        Attempts a true SHAP `Permutation` explanation first; falls back to
        the dependency-free sampling-based Shapley estimator otherwise.
        Returns a dict with: base_value, prediction, contributions (list of
        {feature, display_name, value, contribution} sorted by |contribution|
        descending), and the method used ('shap' or 'simulated_shap').
        """
        try:
            result = self._explain_with_shap(predict_proba_fn, instance, background, feature_names)
            method = "shap"
        except Exception:
            result = self._explain_with_simulation(predict_proba_fn, instance, background, feature_names)
            method = "simulated_shap"

        base_value, prediction, raw_contributions = result

        contributions = []
        for i, fname in enumerate(feature_names):
            contributions.append(
                {
                    "feature": fname,
                    "display_name": _humanize(fname),
                    "value": float(instance[0, i]),
                    "contribution": float(raw_contributions[i]),
                }
            )
        contributions.sort(key=lambda c: abs(c["contribution"]), reverse=True)

        return {
            "method": method,
            "base_value": float(base_value),
            "prediction": float(prediction),
            "contributions": contributions[:top_k],
            "all_contributions": contributions,
        }

    # ------------------------------------------------------------------
    def _explain_with_shap(self, predict_proba_fn, instance, background, feature_names):
        import shap  # noqa: local optional import

        bg_sample_size = min(50, background.shape[0])
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(background.shape[0], size=bg_sample_size, replace=False)
        bg_sample = background[idx]

        def f(x):
            return predict_proba_fn(x)[:, 1]

        explainer = shap.Explainer(f, bg_sample, algorithm="permutation", seed=self.random_state)
        shap_values = explainer(instance, max_evals=max(2 * len(feature_names) + 1, 200))

        base_value = float(np.asarray(shap_values.base_values).reshape(-1)[0])
        contributions = np.asarray(shap_values.values).reshape(-1)
        prediction = base_value + contributions.sum()
        return base_value, prediction, contributions

    # ------------------------------------------------------------------
    def _explain_with_simulation(self, predict_proba_fn, instance, background, feature_names):
        """Dependency-free sampling-based Shapley-value estimator.

        For each feature, estimates its marginal contribution by averaging,
        over `n_samples` random coalitions (feature orderings), the change
        in model output caused by revealing that feature (swapping it from
        a background/baseline value to the instance's actual value) versus
        keeping it masked at the background value. This converges to the
        true Shapley value as n_samples grows and is the same Monte-Carlo
        approximation strategy used internally by KernelSHAP / SamplingSHAP.
        """
        rng = np.random.default_rng(self.random_state)
        n_features = instance.shape[1]

        bg_sample_size = min(30, background.shape[0])
        idx = rng.choice(background.shape[0], size=bg_sample_size, replace=False)
        bg_sample = background[idx]

        base_value = float(predict_proba_fn(bg_sample)[:, 1].mean())
        prediction = float(predict_proba_fn(instance)[:, 1][0])

        contributions = np.zeros(n_features)
        counts = np.zeros(n_features)

        for _ in range(self.n_samples):
            baseline_row = bg_sample[rng.integers(0, bg_sample_size)].copy()
            perm = rng.permutation(n_features)

            current = baseline_row.copy().reshape(1, -1)
            prev_pred = float(predict_proba_fn(current)[:, 1][0])

            for feat_idx in perm:
                current[0, feat_idx] = instance[0, feat_idx]
                new_pred = float(predict_proba_fn(current)[:, 1][0])
                contributions[feat_idx] += new_pred - prev_pred
                counts[feat_idx] += 1
                prev_pred = new_pred

        contributions = np.divide(contributions, np.where(counts == 0, 1, counts))

        # Rescale so contributions exactly sum to (prediction - base_value),
        # correcting for Monte-Carlo sampling noise (standard SHAP practice).
        total_attributed = contributions.sum()
        target_total = prediction - base_value
        if abs(total_attributed) > 1e-9:
            contributions = contributions * (target_total / total_attributed)

        return base_value, prediction, contributions


# --------------------------------------------------------------------------
# Narrative generation - turns raw contributions into plain-English reasons
# --------------------------------------------------------------------------
def generate_decision_narrative(explanation: Dict[str, Any], decision: str, top_n: int = 3) -> Dict[str, List[str]]:
    """Builds a human-readable list of the top positive (approval-supporting)
    and negative (rejection-supporting) factors for the Applicant Portal
    result screen."""
    contributions = explanation["all_contributions"]
    positive = [c for c in contributions if c["contribution"] > 0]
    negative = [c for c in contributions if c["contribution"] < 0]

    positive.sort(key=lambda c: c["contribution"], reverse=True)
    negative.sort(key=lambda c: c["contribution"])

    def fmt(c):
        direction = "increased" if c["contribution"] > 0 else "decreased"
        return f"{c['display_name']} {direction} the approval likelihood by {abs(c['contribution']) * 100:.1f} pts"

    return {
        "supporting_factors": [fmt(c) for c in positive[:top_n]],
        "opposing_factors": [fmt(c) for c in negative[:top_n]],
    }
