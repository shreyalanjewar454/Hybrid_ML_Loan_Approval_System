"""
alternate_credit_scoring.py
=============================
Hybrid ML Loan Approval & Alternate Credit Assessment System

Engine B support module: computes a 0-800 "Alternate Credit Score" for
Thin-File / Unbanked applicants (small vendors, shopkeepers, gig workers)
who have NO CIBIL bureau history, using purely behavioural / transactional
signals:

    1. Cashflow Consistency Score   (0-250)  - stability of daily cashflow
    2. Turnover / Income Stability  (0-200)  - size & adequacy of turnover
    3. Payment Reliability Index    (0-200)  - utility & rent payment history
    4. Transaction History Depth    (0-150)  - length & density of UPI history

    Final Alternate Credit Score = sum of the four components, clipped to [0, 800]

No placeholders. Fully implemented and deterministic given identical inputs.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class AlternateCreditScorer:
    """Computes the Alternate Credit Score (0-800) from UPI / digital
    transaction logs and cash-flow behaviour for thin-file applicants."""

    max_cashflow_points: float = 250.0
    max_turnover_points: float = 200.0
    max_reliability_points: float = 200.0
    max_history_points: float = 150.0

    # ------------------------------------------------------------------
    # Component 1: Cash-flow Consistency Score (0 - 250)
    # ------------------------------------------------------------------
    def compute_cashflow_consistency(self, cashflow_volatility: float) -> float:
        """Lower volatility (coefficient of variation, 0=stable .. 1=erratic)
        yields a higher consistency score. Uses an exponential decay curve so
        that small volatility increases near zero are penalised gently, while
        high volatility is penalised heavily."""
        volatility = float(np.clip(cashflow_volatility, 0.0, 1.0))
        consistency_ratio = np.exp(-3.0 * volatility)  # 1.0 at volatility=0 -> ~0.05 at volatility=1
        return round(consistency_ratio * self.max_cashflow_points, 2)

    # ------------------------------------------------------------------
    # Component 2: Turnover / Income Stability Score (0 - 200)
    # ------------------------------------------------------------------
    def compute_turnover_score(self, monthly_turnover: float, avg_daily_balance: float) -> float:
        """Rewards both a healthy monthly UPI turnover and a healthy average
        daily balance (liquidity cushion), using log-scaled saturation so
        very large turnovers don't dominate disproportionately."""
        turnover = max(monthly_turnover, 0.0)
        balance = max(avg_daily_balance, 0.0)

        # Log-saturating scale: 0 at turnover=0, approaches 1 as turnover grows large.
        turnover_component = np.log1p(turnover) / np.log1p(300000)  # normalised around a healthy benchmark
        turnover_component = float(np.clip(turnover_component, 0.0, 1.0))

        balance_component = np.log1p(balance) / np.log1p(50000)
        balance_component = float(np.clip(balance_component, 0.0, 1.0))

        blended = 0.7 * turnover_component + 0.3 * balance_component
        return round(blended * self.max_turnover_points, 2)

    # ------------------------------------------------------------------
    # Component 3: Utility / Rent Payment Reliability Index (0 - 200)
    # ------------------------------------------------------------------
    def compute_payment_reliability(self, utility_reliability: float, rent_reliability: float) -> float:
        """Weighted average of utility-bill and rent payment on-time ratios
        (each expressed as a 0-1 fraction of payments made on time)."""
        utility = float(np.clip(utility_reliability, 0.0, 1.0))
        rent = float(np.clip(rent_reliability, 0.0, 1.0))
        blended = 0.55 * utility + 0.45 * rent
        return round(blended * self.max_reliability_points, 2)

    # ------------------------------------------------------------------
    # Component 4: Transaction History Depth Score (0 - 150)
    # ------------------------------------------------------------------
    def compute_history_depth(self, months_history: int, txn_count: int) -> float:
        """Rewards longer observed transaction history and a healthy
        transaction frequency (more txns = richer behavioural signal),
        both saturating so very long histories don't disproportionately
        dominate applicants who are simply new to digital payments."""
        months = max(months_history, 0)
        txns = max(txn_count, 0)

        months_component = min(months / 36.0, 1.0)  # saturates at 36 months (3 years)
        txn_component = min(txns / 400.0, 1.0)  # saturates at 400 monthly transactions

        blended = 0.6 * months_component + 0.4 * txn_component
        return round(blended * self.max_history_points, 2)

    # ------------------------------------------------------------------
    # Composite score
    # ------------------------------------------------------------------
    def compute_final_score(
        self,
        monthly_turnover: float,
        cashflow_volatility: float,
        utility_reliability: float,
        rent_reliability: float,
        months_history: int,
        txn_count: int,
        avg_daily_balance: float,
    ) -> Dict[str, float]:
        cashflow_pts = self.compute_cashflow_consistency(cashflow_volatility)
        turnover_pts = self.compute_turnover_score(monthly_turnover, avg_daily_balance)
        reliability_pts = self.compute_payment_reliability(utility_reliability, rent_reliability)
        history_pts = self.compute_history_depth(months_history, txn_count)

        raw_total = cashflow_pts + turnover_pts + reliability_pts + history_pts
        final_score = float(np.clip(raw_total, 0.0, 800.0))

        return {
            "cashflow_consistency_points": cashflow_pts,
            "turnover_stability_points": turnover_pts,
            "payment_reliability_points": reliability_pts,
            "history_depth_points": history_pts,
            "final_score": round(final_score, 1),
        }

    # ------------------------------------------------------------------
    # Human-readable score band (mirrors CIBIL-style bands for familiarity)
    # ------------------------------------------------------------------
    @staticmethod
    def score_band(final_score: float) -> str:
        if final_score >= 650:
            return "Excellent"
        elif final_score >= 500:
            return "Good"
        elif final_score >= 350:
            return "Fair"
        else:
            return "Poor"


def simulate_upi_monthly_statement(
    monthly_turnover: float,
    txn_count: int,
    n_days: int = 30,
    seed: int | None = None,
) -> List[float]:
    """Simulates a synthetic day-by-day UPI transaction turnover series for
    a given month, used by the Thin-File Alternate Scoring Portal to power
    the 'simulate UPI monthly statement' visualisation. Returns a list of
    `n_days` daily net-turnover values that sum approximately to
    `monthly_turnover`.
    """
    rng = np.random.default_rng(seed)
    # Distribute turnover unevenly across days (vendors have peak/slow days)
    weights = rng.dirichlet(np.ones(n_days) * 1.5)
    daily_values = weights * monthly_turnover
    # Add small transaction-level noise proportional to txn density
    noise_scale = max(monthly_turnover / max(txn_count, 1), 1.0) * 0.15
    noise = rng.normal(0, noise_scale, size=n_days)
    daily_values = np.clip(daily_values + noise, 0, None)
    return daily_values.round(2).tolist()


def compute_volatility_from_series(daily_values: List[float]) -> float:
    """Computes coefficient-of-variation-based volatility (0-1 clipped) from
    a daily turnover series, used to feed `compute_cashflow_consistency`."""
    arr = np.asarray(daily_values, dtype=float)
    if arr.mean() <= 0:
        return 1.0
    cv = arr.std() / (arr.mean() + 1e-9)
    return float(np.clip(cv, 0.0, 1.0))
