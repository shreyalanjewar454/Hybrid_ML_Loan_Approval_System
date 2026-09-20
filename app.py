"""
app.py
=======
Hybrid ML Loan Approval & Alternate Credit Assessment System
Main Streamlit FinTech Dashboard.

Run with:  streamlit run app.py

Pages:
    1. Applicant Portal (Dual Engine)
    2. Thin-File Alternate Scoring Engine
    3. Bank Admin Dashboard
    4. System Analytics & Financial Inclusion

No placeholders. Fully implemented and runnable (given requirements.txt
installed).
"""

from __future__ import annotations

import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from src.data_preprocessing import (
    append_application_log,
    load_application_log,
    get_or_create_traditional_dataset,
    get_or_create_thinfile_dataset,
    APPLICATIONS_LOG_PATH,
)
from src.alternate_credit_scoring import (
    AlternateCreditScorer,
    simulate_upi_monthly_statement,
    compute_volatility_from_series,
)
from src.predict import HybridLoanPredictionSystem
from src.xai_explainer import XAIExplainer, generate_decision_narrative

# ==========================================================================
# Page configuration
# ==========================================================================
st.set_page_config(
    page_title="Hybrid ML Loan Approval & Alternate Credit System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================================
# Session state defaults
# ==========================================================================
_DEFAULTS = {
    "theme": "Dark",
    "nav_page": "Applicant Portal (Dual Engine)",
    "applicant_step": 1,
    "applicant_mode": "Has CIBIL Score",
    "form_data": {},
    "last_result": None,
    "thinfile_lab_result": None,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ==========================================================================
# Custom CSS - Modern high-end FinTech dashboard styling
# ==========================================================================
def inject_css(theme: str) -> None:
    if theme == "Dark":
        bg = "#0b0f19"
        bg_secondary = "#111827"
        card_bg = "linear-gradient(145deg, #141b2d, #0f1521)"
        text_primary = "#e5e9f0"
        text_secondary = "#8b95a7"
        border_color = "rgba(255,255,255,0.08)"
        shadow = "0 8px 32px rgba(0,0,0,0.45)"
    else:
        bg = "#f4f6fb"
        bg_secondary = "#ffffff"
        card_bg = "linear-gradient(145deg, #ffffff, #f0f3fa)"
        text_primary = "#131722"
        text_secondary = "#5b6472"
        border_color = "rgba(15,23,42,0.08)"
        shadow = "0 8px 28px rgba(15,23,42,0.08)"

    accent = "#5b8cff"
    accent_2 = "#22d3ee"
    success = "#22c55e"
    warning = "#f59e0b"
    danger = "#ef4444"

    st.markdown(
        f"""
        <style>
        html, body, [class*="css"] {{
            font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
        }}
        .stApp {{
            background: {bg};
            color: {text_primary};
        }}
        section[data-testid="stSidebar"] {{
            background: {bg_secondary};
            border-right: 1px solid {border_color};
        }}
        h1, h2, h3, h4 {{
            color: {text_primary} !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em;
        }}
        p, span, label, div {{
            color: {text_primary};
        }}
        .subtle {{
            color: {text_secondary} !important;
            font-size: 0.92rem;
        }}
        .hero-banner {{
            background: linear-gradient(120deg, rgba(91,140,255,0.16), rgba(34,211,238,0.10));
            border: 1px solid {border_color};
            border-radius: 20px;
            padding: 28px 32px;
            margin-bottom: 22px;
            box-shadow: {shadow};
        }}
        .hero-title {{
            font-size: 1.9rem;
            font-weight: 800;
            background: linear-gradient(90deg, {accent}, {accent_2});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 4px;
        }}
        .glass-card {{
            background: {card_bg};
            border: 1px solid {border_color};
            border-radius: 18px;
            padding: 22px 24px;
            box-shadow: {shadow};
            margin-bottom: 18px;
        }}
        .metric-badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 5px 14px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.82rem;
            letter-spacing: 0.02em;
        }}
        .badge-low {{ background: rgba(34,197,94,0.16); color: {success}; border: 1px solid rgba(34,197,94,0.35); }}
        .badge-medium {{ background: rgba(245,158,11,0.16); color: {warning}; border: 1px solid rgba(245,158,11,0.35); }}
        .badge-high {{ background: rgba(239,68,68,0.16); color: {danger}; border: 1px solid rgba(239,68,68,0.35); }}
        .badge-approved {{ background: rgba(34,197,94,0.16); color: {success}; border: 1px solid rgba(34,197,94,0.35); }}
        .badge-rejected {{ background: rgba(239,68,68,0.16); color: {danger}; border: 1px solid rgba(239,68,68,0.35); }}
        .badge-review {{ background: rgba(245,158,11,0.16); color: {warning}; border: 1px solid rgba(245,158,11,0.35); }}
        .fraud-alert {{
            background: linear-gradient(120deg, rgba(239,68,68,0.18), rgba(239,68,68,0.06));
            border: 1px solid rgba(239,68,68,0.45);
            border-radius: 14px;
            padding: 14px 18px;
            color: {danger};
            font-weight: 600;
            margin: 10px 0;
        }}
        .step-pill {{
            display: inline-block;
            padding: 6px 16px;
            border-radius: 999px;
            margin-right: 6px;
            font-size: 0.8rem;
            font-weight: 700;
            border: 1px solid {border_color};
            color: {text_secondary};
        }}
        .step-pill-active {{
            background: linear-gradient(90deg, {accent}, {accent_2});
            color: #08111f;
            border: none;
        }}
        .stButton>button {{
            border-radius: 10px;
            font-weight: 700;
            border: 1px solid {border_color};
        }}
        .stButton>button[kind="primary"] {{
            background: linear-gradient(90deg, {accent}, {accent_2});
            color: #08111f;
            border: none;
        }}
        div[data-testid="stMetric"] {{
            background: {card_bg};
            border: 1px solid {border_color};
            border-radius: 14px;
            padding: 14px 16px;
            box-shadow: {shadow};
        }}
        .sidebar-brand {{
            font-size: 1.25rem;
            font-weight: 800;
            background: linear-gradient(90deg, {accent}, {accent_2});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 2px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==========================================================================
# Cached system loader (trains once, on first run, then reused)
# ==========================================================================
@st.cache_resource(show_spinner="Bootstrapping dual-engine ML system (first run trains models)...")
def get_system() -> HybridLoanPredictionSystem:
    return HybridLoanPredictionSystem()


# ==========================================================================
# Small reusable UI helpers
# ==========================================================================
def risk_badge_html(risk_grade: str) -> str:
    cls = {"Low": "badge-low", "Medium": "badge-medium", "High": "badge-high"}.get(risk_grade, "badge-medium")
    return f'<span class="metric-badge {cls}">⬤ {risk_grade} Risk</span>'


def decision_badge_html(decision: str) -> str:
    cls = {
        "Approved": "badge-approved",
        "Rejected": "badge-rejected",
        "Manual Review Required": "badge-review",
    }.get(decision, "badge-review")
    icon = {"Approved": "✅", "Rejected": "⛔", "Manual Review Required": "🕵️"}.get(decision, "•")
    return f'<span class="metric-badge {cls}">{icon} {decision}</span>'


def gauge_chart(value: float, title: str, suffix: str = "%") -> go.Figure:
    color = "#22c55e" if value >= 70 else ("#f59e0b" if value >= 45 else "#ef4444")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": suffix, "font": {"size": 34}},
            title={"text": title, "font": {"size": 15}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8b95a7"},
                "bar": {"color": color},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 45], "color": "rgba(239,68,68,0.18)"},
                    {"range": [45, 72], "color": "rgba(245,158,11,0.18)"},
                    {"range": [72, 100], "color": "rgba(34,197,94,0.18)"},
                ],
            },
        )
    )
    fig.update_layout(height=240, margin=dict(l=20, r=20, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)", font_color="#8b95a7")
    return fig


def shap_waterfall_chart(explanation: dict, top_n: int = 8) -> go.Figure:
    contribs = explanation["contributions"][:top_n]
    contribs = sorted(contribs, key=lambda c: c["contribution"])
    labels = [c["display_name"] for c in contribs]
    values = [round(c["contribution"] * 100, 2) for c in contribs]

    fig = go.Figure(
        go.Waterfall(
            orientation="h",
            measure=["relative"] * len(values),
            y=labels,
            x=values,
            connector={"line": {"color": "rgba(150,150,150,0.35)"}},
            increasing={"marker": {"color": "#22c55e"}},
            decreasing={"marker": {"color": "#ef4444"}},
            text=[f"{v:+.1f}" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=f"SHAP-Style Explanation — base {explanation['base_value']*100:.1f}% → predicted {explanation['prediction']*100:.1f}%",
        height=360,
        margin=dict(l=10, r=10, t=60, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#8b95a7",
    )
    return fig


def render_prediction_result(result: dict, explanation: dict, narrative: dict) -> None:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown(f"#### 📄 Prediction Result — {result['engine']}")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(decision_badge_html(result["decision"]), unsafe_allow_html=True)
    with c2:
        st.markdown(risk_badge_html(result["risk_grade"]), unsafe_allow_html=True)
    with c3:
        st.markdown(f"**Max Eligible Loan:** ₹{result['max_eligible_loan']:,.0f}")

    if result["is_fraud_flagged"]:
        st.markdown(
            f"<div class='fraud-alert'>🚨 FRAUD / ANOMALY FILTER TRIGGERED — this application shows statistically "
            f"unusual patterns (fraud likelihood {result['fraud_probability']:.1f}%). Routed to manual review.</div>",
            unsafe_allow_html=True,
        )

    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(gauge_chart(result["approval_probability"], "Approval Probability"), use_container_width=True)
    with g2:
        st.plotly_chart(gauge_chart(result["fraud_probability"], "Fraud / Anomaly Likelihood"), use_container_width=True)

    st.markdown("##### 🧠 Explainable AI (XAI) — Why this decision?")
    st.plotly_chart(shap_waterfall_chart(explanation), use_container_width=True)

    nc1, nc2 = st.columns(2)
    with nc1:
        st.markdown("**✅ Factors supporting approval**")
        for f in narrative["supporting_factors"]:
            st.markdown(f"- {f}")
        if not narrative["supporting_factors"]:
            st.markdown("_No strongly positive factors detected._")
    with nc2:
        st.markdown("**⚠️ Factors working against approval**")
        for f in narrative["opposing_factors"]:
            st.markdown(f"- {f}")
        if not narrative["opposing_factors"]:
            st.markdown("_No strongly negative factors detected._")

    st.markdown(
        f"<p class='subtle'>Explanation method: {explanation['method']} "
        f"({'true Shapley values via the shap library' if explanation['method']=='shap' else 'dependency-free sampling-based Shapley approximation'}).</p>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ==========================================================================
# PAGE 1: Applicant Portal (Dual Engine)
# ==========================================================================
def page_applicant_portal(system: HybridLoanPredictionSystem) -> None:
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-title">Applicant Portal — Dual Evaluation Engine</div>
            <div class="subtle">Traditional CIBIL-based scoring or Thin-File Alternate-Data scoring, routed automatically to the right ML engine.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mode = st.radio(
        "Applicant Type",
        ["Has CIBIL Score", "First-Time / Thin-File Borrower (UPI & Cashflow Data)"],
        horizontal=True,
        index=0 if st.session_state.applicant_mode == "Has CIBIL Score" else 1,
        key="applicant_mode_radio",
    )
    if mode != st.session_state.applicant_mode:
        st.session_state.applicant_mode = mode
        st.session_state.applicant_step = 1
        st.session_state.form_data = {}
        st.session_state.last_result = None

    is_traditional = mode == "Has CIBIL Score"
    total_steps = 4
    step = st.session_state.applicant_step

    step_labels = ["1. Personal Details", "2. Financial Profile", "3. Loan Request", "4. Review & Submit"]
    pills = "".join(
        f'<span class="step-pill {"step-pill-active" if i+1==step else ""}">{label}</span>'
        for i, label in enumerate(step_labels)
    )
    st.markdown(pills, unsafe_allow_html=True)
    st.progress(step / total_steps)
    st.markdown("<br>", unsafe_allow_html=True)

    fd = st.session_state.form_data

    # ---------------- Step 1: Personal Details ---------------------------
    if step == 1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            fd["applicant_name"] = st.text_input("Full Name", value=fd.get("applicant_name", ""))
            fd["age"] = st.number_input("Age", min_value=18, max_value=75, value=fd.get("age", 30))
        with c2:
            if is_traditional:
                fd["employment_type"] = st.selectbox(
                    "Employment Type", ["Salaried", "Self-Employed", "Business Owner"],
                    index=["Salaried", "Self-Employed", "Business Owner"].index(fd.get("employment_type", "Salaried")),
                )
                fd["dependents"] = st.number_input("Number of Dependents", 0, 10, value=fd.get("dependents", 0))
            else:
                fd["occupation_type"] = st.selectbox(
                    "Occupation Type", ["Vendor", "Shopkeeper", "Gig Worker", "Freelancer"],
                    index=["Vendor", "Shopkeeper", "Gig Worker", "Freelancer"].index(fd.get("occupation_type", "Vendor")),
                )
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- Step 2: Financial Profile ---------------------------
    elif step == 2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        if is_traditional:
            c1, c2 = st.columns(2)
            with c1:
                fd["annual_income"] = st.number_input("Annual Income (₹)", min_value=50000, value=fd.get("annual_income", 600000), step=10000)
                fd["years_employed"] = st.number_input("Years of Employment", 0, 45, value=fd.get("years_employed", 5))
                fd["cibil_score"] = st.slider("CIBIL Score", 300, 900, value=fd.get("cibil_score", 720))
            with c2:
                fd["existing_loans_count"] = st.number_input("Existing Loans Count", 0, 10, value=fd.get("existing_loans_count", 0))
                fd["existing_emi_monthly"] = st.number_input("Existing Monthly EMI (₹)", 0, value=fd.get("existing_emi_monthly", 0), step=500)
                fd["savings_balance"] = st.number_input("Savings Balance (₹)", 0, value=fd.get("savings_balance", 100000), step=5000)
                fd["previous_default"] = 1 if st.checkbox("Has a previous loan default on record?", value=bool(fd.get("previous_default", 0))) else 0
        else:
            c1, c2 = st.columns(2)
            with c1:
                fd["monthly_upi_txn_count"] = st.number_input("Monthly UPI Transaction Count", 5, 2000, value=fd.get("monthly_upi_txn_count", 200))
                fd["monthly_upi_turnover"] = st.number_input("Monthly UPI Turnover (₹)", 1000, value=fd.get("monthly_upi_turnover", 80000), step=1000)
                fd["avg_daily_balance"] = st.number_input("Average Daily Balance (₹)", 0, value=fd.get("avg_daily_balance", 8000), step=500)
            with c2:
                fd["cashflow_volatility"] = st.slider("Cash-Flow Volatility (0=stable, 1=erratic)", 0.0, 1.0, value=fd.get("cashflow_volatility", 0.25), step=0.01)
                fd["utility_payment_reliability"] = st.slider("Utility Payment Reliability", 0.0, 1.0, value=fd.get("utility_payment_reliability", 0.8), step=0.01)
                fd["rent_payment_reliability"] = st.slider("Rent Payment Reliability", 0.0, 1.0, value=fd.get("rent_payment_reliability", 0.75), step=0.01)
                fd["months_of_transaction_history"] = st.number_input("Months of Transaction History", 1, 120, value=fd.get("months_of_transaction_history", 18))
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- Step 3: Loan Request ---------------------------------
    elif step == 3:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        if is_traditional:
            c1, c2 = st.columns(2)
            with c1:
                fd["requested_loan_amount"] = st.number_input("Requested Loan Amount (₹)", 10000, value=fd.get("requested_loan_amount", 500000), step=10000)
            with c2:
                fd["loan_tenure_months"] = st.selectbox(
                    "Loan Tenure (months)", [12, 24, 36, 48, 60, 84, 120, 180, 240],
                    index=[12, 24, 36, 48, 60, 84, 120, 180, 240].index(fd.get("loan_tenure_months", 60)),
                )
        else:
            fd["requested_loan_amount"] = st.number_input("Requested Loan Amount (₹)", 5000, value=fd.get("requested_loan_amount", 100000), step=5000)
            scorer = AlternateCreditScorer()
            breakdown = scorer.compute_final_score(
                monthly_turnover=fd.get("monthly_upi_turnover", 80000),
                cashflow_volatility=fd.get("cashflow_volatility", 0.25),
                utility_reliability=fd.get("utility_payment_reliability", 0.8),
                rent_reliability=fd.get("rent_payment_reliability", 0.75),
                months_history=fd.get("months_of_transaction_history", 18),
                txn_count=fd.get("monthly_upi_txn_count", 200),
                avg_daily_balance=fd.get("avg_daily_balance", 8000),
            )
            fd["alternate_credit_score"] = int(breakdown["final_score"])
            st.info(f"📊 Computed Alternate Credit Score: **{breakdown['final_score']:.0f} / 800** ({scorer.score_band(breakdown['final_score'])})")
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- Step 4: Review & Submit -------------------------------
    elif step == 4:
        if is_traditional:
            monthly_income = max(fd.get("annual_income", 1) / 12.0, 1.0)
            implied_new_emi = fd.get("requested_loan_amount", 0) / max(fd.get("loan_tenure_months", 1), 1)
            fd["dti_ratio"] = round(
                min((fd.get("existing_emi_monthly", 0) + implied_new_emi) / monthly_income, 3.0), 3
            )
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("##### 🔍 Review Your Application")
        review_df = pd.DataFrame(
            [{"Field": k.replace("_", " ").title(), "Value": v} for k, v in fd.items() if k != "applicant_name"]
        )
        st.dataframe(review_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("🚀 Submit Application for Instant Prediction", type="primary", use_container_width=True):
            try:
                if is_traditional:
                    applicant = {k: v for k, v in fd.items() if k != "applicant_name"}
                    result = system.predict_traditional(applicant)
                    background_df = get_or_create_traditional_dataset().sample(min(40, 4000), random_state=1)
                    from src.data_preprocessing import split_X_y

                    Xbg_raw, _ = split_X_y(background_df, "approved")
                    background = system.traditional_preprocessor.transform(Xbg_raw)
                    predict_fn = system.engine_a.predict_proba
                else:
                    applicant = {k: v for k, v in fd.items() if k != "applicant_name"}
                    result = system.predict_thinfile(applicant)
                    background_df = get_or_create_thinfile_dataset().sample(min(40, 3000), random_state=1)
                    from src.data_preprocessing import split_X_y

                    Xbg_raw, _ = split_X_y(background_df, "approved")
                    background = system.thinfile_preprocessor.transform(Xbg_raw)
                    predict_fn = system.engine_b.predict_proba

                explainer = XAIExplainer(n_samples=60)
                explanation = explainer.explain_instance(
                    predict_fn, result["processed_features"], background, result["feature_names"]
                )
                narrative = generate_decision_narrative(explanation, result["decision"])

                append_application_log(
                    {
                        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                        "applicant_name": fd.get("applicant_name", "Anonymous"),
                        "engine": result["engine"],
                        "decision": result["decision"],
                        "risk_grade": result["risk_grade"],
                        "approval_probability": result["approval_probability"],
                        "fraud_probability": result["fraud_probability"],
                        "is_fraud_flagged": result["is_fraud_flagged"],
                        "max_eligible_loan": result["max_eligible_loan"],
                        "requested_loan_amount": fd.get("requested_loan_amount", 0),
                    }
                )
                st.session_state.last_result = (result, explanation, narrative)
                st.success("Application processed successfully.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

        if st.session_state.last_result is not None:
            render_prediction_result(*st.session_state.last_result)

    # ---------------- Navigation controls -----------------------------------
    nav1, nav2, nav3 = st.columns([1, 4, 1])
    with nav1:
        if step > 1 and st.button("← Back"):
            st.session_state.applicant_step -= 1
            st.rerun()
    with nav3:
        if step < total_steps and st.button("Next →", type="primary"):
            st.session_state.applicant_step += 1
            st.rerun()


# ==========================================================================
# PAGE 2: Thin-File Alternate Scoring Engine (dedicated lab)
# ==========================================================================
def page_thinfile_lab(system: HybridLoanPredictionSystem) -> None:
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-title">Thin-File Alternate Scoring Engine</div>
            <div class="subtle">Simulate a UPI monthly statement and build an Alternate Credit Score (0–800) purely from behavioural cash-flow data.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        monthly_turnover = st.number_input("Monthly UPI Turnover (₹)", 1000, 1000000, 85000, step=1000)
        txn_count = st.number_input("Monthly UPI Transaction Count", 5, 2000, 250)
    with c2:
        avg_daily_balance = st.number_input("Average Daily Balance (₹)", 0, 500000, 9000, step=500)
        months_history = st.number_input("Months of Transaction History", 1, 120, 20)
    with c3:
        utility_reliability = st.slider("Utility Payment Reliability", 0.0, 1.0, 0.82, 0.01)
        rent_reliability = st.slider("Rent Payment Reliability", 0.0, 1.0, 0.77, 0.01)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("##### 📈 Simulated UPI Monthly Statement")
    seed = st.slider("Simulation Seed (vary the day-to-day pattern)", 0, 100, 7)
    daily_series = simulate_upi_monthly_statement(monthly_turnover, txn_count, n_days=30, seed=seed)
    volatility = compute_volatility_from_series(daily_series)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=[f"Day {i+1}" for i in range(30)], y=daily_series, marker_color="#5b8cff", name="Daily Net UPI Turnover")
    )
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#8b95a7",
        xaxis=dict(showticklabels=False),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"<p class='subtle'>Derived cash-flow volatility (coefficient of variation) from this simulated statement: <b>{volatility:.3f}</b></p>", unsafe_allow_html=True)

    use_derived_volatility = st.checkbox("Use this simulated volatility in the score below", value=True)
    st.markdown("</div>", unsafe_allow_html=True)

    final_volatility = volatility if use_derived_volatility else 0.25

    scorer = AlternateCreditScorer()
    breakdown = scorer.compute_final_score(
        monthly_turnover=monthly_turnover,
        cashflow_volatility=final_volatility,
        utility_reliability=utility_reliability,
        rent_reliability=rent_reliability,
        months_history=months_history,
        txn_count=txn_count,
        avg_daily_balance=avg_daily_balance,
    )
    band = scorer.score_band(breakdown["final_score"])

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("##### 🧮 Alternate Credit Score Breakdown (0–800 scale)")
    g1, g2 = st.columns([1, 2])
    with g1:
        fig_score = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=breakdown["final_score"],
                number={"font": {"size": 34}},
                title={"text": f"Alternate Credit Score — {band}", "font": {"size": 15}},
                gauge={
                    "axis": {"range": [0, 800]},
                    "bar": {"color": "#22d3ee"},
                    "steps": [
                        {"range": [0, 350], "color": "rgba(239,68,68,0.18)"},
                        {"range": [350, 500], "color": "rgba(245,158,11,0.18)"},
                        {"range": [500, 650], "color": "rgba(59,130,246,0.18)"},
                        {"range": [650, 800], "color": "rgba(34,197,94,0.18)"},
                    ],
                },
            )
        )
        fig_score.update_layout(height=280, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)", font_color="#8b95a7")
        st.plotly_chart(fig_score, use_container_width=True)
    with g2:
        comp_df = pd.DataFrame(
            {
                "Component": ["Cash-Flow Consistency", "Turnover Stability", "Payment Reliability", "History Depth"],
                "Points": [
                    breakdown["cashflow_consistency_points"],
                    breakdown["turnover_stability_points"],
                    breakdown["payment_reliability_points"],
                    breakdown["history_depth_points"],
                ],
                "Max": [250, 200, 200, 150],
            }
        )
        fig_bar = px.bar(comp_df, x="Points", y="Component", orientation="h", text="Points", range_x=[0, 250])
        fig_bar.update_traces(marker_color="#5b8cff", textposition="outside")
        fig_bar.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#8b95a7")
        st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("##### ⚡ Quick Eligibility Check (Engine B)")
    occupation = st.selectbox("Occupation Type", ["Vendor", "Shopkeeper", "Gig Worker", "Freelancer"])
    age_q = st.number_input("Age", 18, 75, 30, key="thinfile_lab_age")
    requested = st.number_input("Requested Loan Amount (₹)", 5000, 1000000, 100000, step=5000, key="thinfile_lab_loan")

    if st.button("Run Quick Eligibility Check", type="primary"):
        applicant = {
            "age": age_q,
            "occupation_type": occupation,
            "monthly_upi_txn_count": txn_count,
            "monthly_upi_turnover": monthly_turnover,
            "avg_daily_balance": avg_daily_balance,
            "cashflow_volatility": final_volatility,
            "utility_payment_reliability": utility_reliability,
            "rent_payment_reliability": rent_reliability,
            "months_of_transaction_history": months_history,
            "requested_loan_amount": requested,
            "alternate_credit_score": int(breakdown["final_score"]),
        }
        result = system.predict_thinfile(applicant)

        background_df = get_or_create_thinfile_dataset().sample(min(40, 3000), random_state=1)
        from src.data_preprocessing import split_X_y

        Xbg_raw, _ = split_X_y(background_df, "approved")
        background = system.thinfile_preprocessor.transform(Xbg_raw)
        explainer = XAIExplainer(n_samples=60)
        explanation = explainer.explain_instance(
            system.engine_b.predict_proba, result["processed_features"], background, result["feature_names"]
        )
        narrative = generate_decision_narrative(explanation, result["decision"])

        append_application_log(
            {
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                "applicant_name": "Thin-File Lab Applicant",
                "engine": result["engine"],
                "decision": result["decision"],
                "risk_grade": result["risk_grade"],
                "approval_probability": result["approval_probability"],
                "fraud_probability": result["fraud_probability"],
                "is_fraud_flagged": result["is_fraud_flagged"],
                "max_eligible_loan": result["max_eligible_loan"],
                "requested_loan_amount": requested,
            }
        )
        st.session_state.thinfile_lab_result = (result, explanation, narrative)

    if st.session_state.thinfile_lab_result is not None:
        render_prediction_result(*st.session_state.thinfile_lab_result)
    st.markdown("</div>", unsafe_allow_html=True)


# ==========================================================================
# PAGE 3: Bank Admin Dashboard
# ==========================================================================
def page_admin_dashboard(system: HybridLoanPredictionSystem) -> None:
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-title">Bank Admin Dashboard</div>
            <div class="subtle">Recent applications, fraud/anomaly alerts, alternate-data score logs, and manual override controls.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    log_df = load_application_log()

    if log_df.empty:
        st.info("No applications have been processed yet. Submit an application via the Applicant Portal to populate this dashboard.")
        return

    if "override_decision" not in log_df.columns:
        log_df["override_decision"] = ""
    if "application_index" not in log_df.columns:
        log_df.insert(0, "application_index", range(len(log_df)))

    flagged = log_df[log_df["is_fraud_flagged"] == True]  # noqa: E712

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Applications", len(log_df))
    m2.metric("Approved", int((log_df["decision"] == "Approved").sum()))
    m3.metric("Rejected", int((log_df["decision"] == "Rejected").sum()))
    m4.metric("🚨 Fraud Flagged", len(flagged))

    if not flagged.empty:
        st.markdown(
            f"<div class='fraud-alert'>🚨 {len(flagged)} application(s) flagged by the real-time Isolation Forest "
            f"fraud/anomaly filter and routed to manual review. See highlighted rows below.</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("##### 📋 Recent Applications")

    def highlight_fraud(row):
        if row.get("is_fraud_flagged", False):
            return ["background-color: rgba(239,68,68,0.18)"] * len(row)
        return [""] * len(row)

    display_cols = [
        "application_index", "timestamp", "applicant_name", "engine", "decision", "risk_grade",
        "approval_probability", "fraud_probability", "is_fraud_flagged", "max_eligible_loan",
        "requested_loan_amount", "override_decision",
    ]
    display_cols = [c for c in display_cols if c in log_df.columns]
    styled = log_df[display_cols].sort_values("timestamp", ascending=False).style.apply(highlight_fraud, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("##### 🧾 Alternate-Data Score Logs (Engine B applicants)")
    alt_df = log_df[log_df["engine"].str.contains("Thin-File", na=False)]
    if alt_df.empty:
        st.markdown("_No thin-file applications logged yet._")
    else:
        st.dataframe(
            alt_df[["timestamp", "applicant_name", "risk_grade", "approval_probability", "max_eligible_loan"]]
            .sort_values("timestamp", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("##### 🛠️ Manual Override Controls")
    oc1, oc2, oc3 = st.columns([2, 2, 1])
    with oc1:
        selected_idx = st.selectbox("Select Application Index", log_df["application_index"].tolist())
    with oc2:
        override_choice = st.selectbox("Override Decision To", ["", "Approved", "Rejected", "Manual Review Required"])
    with oc3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Apply Override", type="primary"):
            if override_choice:
                log_df.loc[log_df["application_index"] == selected_idx, "override_decision"] = override_choice
                log_df.drop(columns=["application_index"]).to_csv(APPLICATIONS_LOG_PATH, index=False)
                st.success(f"Application #{selected_idx} overridden to '{override_choice}'. Refresh to see changes.")
            else:
                st.warning("Please select an override decision.")
    st.markdown("</div>", unsafe_allow_html=True)


# ==========================================================================
# PAGE 4: System Analytics & Financial Inclusion
# ==========================================================================
def page_analytics(system: HybridLoanPredictionSystem) -> None:
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-title">System Analytics & Financial Inclusion</div>
            <div class="subtle">Portfolio-level insights across both credit engines, model performance, and financial inclusion impact.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Engine A ROC-AUC", f"{system.metrics_a.get('roc_auc', 0):.3f}")
    m2.metric("Engine A Accuracy", f"{system.metrics_a.get('accuracy', 0)*100:.1f}%")
    m3.metric("Engine B ROC-AUC", f"{system.metrics_b.get('roc_auc', 0):.3f}")
    m4.metric("Engine B Accuracy", f"{system.metrics_b.get('accuracy', 0)*100:.1f}%")

    log_df = load_application_log()

    df_trad = get_or_create_traditional_dataset()
    df_thin = get_or_create_thinfile_dataset()

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("##### 📊 Training Portfolio: Approval Rate by Engine")
    approval_rates = pd.DataFrame(
        {
            "Engine": ["Engine A — Traditional (CIBIL)", "Engine B — Thin-File (Alternate Data)"],
            "Approval Rate (%)": [
                df_trad["approved"].mean() * 100,
                df_thin["approved"].mean() * 100,
            ],
            "Applicant Volume": [len(df_trad), len(df_thin)],
        }
    )
    a1, a2 = st.columns(2)
    with a1:
        fig1 = px.bar(approval_rates, x="Engine", y="Approval Rate (%)", color="Engine", text_auto=".1f",
                       color_discrete_sequence=["#5b8cff", "#22d3ee"])
        fig1.update_layout(height=340, showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#8b95a7")
        st.plotly_chart(fig1, use_container_width=True)
    with a2:
        fig2 = px.pie(approval_rates, names="Engine", values="Applicant Volume", hole=0.55,
                       color_discrete_sequence=["#5b8cff", "#22d3ee"])
        fig2.update_layout(height=340, paper_bgcolor="rgba(0,0,0,0)", font_color="#8b95a7", title="Traditional vs Alternate Loan Volume")
        st.plotly_chart(fig2, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("##### 🎯 Portfolio Credit Risk Distribution")
    from src.predict import probability_to_risk_grade

    trad_probs = system.engine_a.predict_proba(
        system.traditional_preprocessor.transform(df_trad.drop(columns=["approved", "applicant_id"]))
    )[:, 1]
    thin_probs = system.engine_b.predict_proba(
        system.thinfile_preprocessor.transform(df_thin.drop(columns=["approved", "applicant_id"]))
    )[:, 1]
    risk_grades = [probability_to_risk_grade(p) for p in np.concatenate([trad_probs, thin_probs])]
    risk_counts = pd.Series(risk_grades).value_counts().reindex(["Low", "Medium", "High"]).fillna(0)

    r1, r2 = st.columns([1, 2])
    with r1:
        fig3 = px.pie(
            names=risk_counts.index, values=risk_counts.values, hole=0.5,
            color=risk_counts.index,
            color_discrete_map={"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"},
        )
        fig3.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", font_color="#8b95a7")
        st.plotly_chart(fig3, use_container_width=True)
    with r2:
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(x=["Engine A (Traditional)"], y=[trad_probs.mean() * 100], name="Avg Approval Prob (A)", marker_color="#5b8cff"))
        fig4.add_trace(go.Bar(x=["Engine B (Thin-File)"], y=[thin_probs.mean() * 100], name="Avg Approval Prob (B)", marker_color="#22d3ee"))
        fig4.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#8b95a7", yaxis_title="Avg Approval Probability (%)")
        st.plotly_chart(fig4, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("##### 🌍 Financial Inclusion Impact")
    inclusion_rate = df_thin["approved"].mean() * 100
    ic1, ic2, ic3 = st.columns(3)
    ic1.metric("Thin-File Applicants Modeled", f"{len(df_thin):,}")
    ic2.metric("Thin-File Approval Rate", f"{inclusion_rate:.1f}%")
    ic3.metric("Borrowers With NO CIBIL History Served", f"{int(df_thin['approved'].sum()):,}")
    st.markdown(
        "<p class='subtle'>Every applicant approved through Engine B represents a borrower who would have been "
        "invisible to a traditional CIBIL-only underwriting system — extending formal credit access to vendors, "
        "shopkeepers, gig workers and freelancers based purely on verifiable digital cash-flow behaviour.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if not log_df.empty:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("##### 🕒 Live Application Decisions (this session's submissions)")
        fig5 = px.histogram(log_df, x="decision", color="engine", barmode="group",
                             color_discrete_sequence=["#5b8cff", "#22d3ee"])
        fig5.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#8b95a7")
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ==========================================================================
# MAIN — Sidebar navigation & routing
# ==========================================================================
def main() -> None:
    inject_css(st.session_state.theme)

    with st.sidebar:
        st.markdown("<div class='sidebar-brand'>💳 HybridCredit AI</div>", unsafe_allow_html=True)
        st.markdown("<p class='subtle'>Dual-Engine Loan Approval System</p>", unsafe_allow_html=True)
        st.markdown("---")

        st.session_state.theme = st.select_slider("Theme", options=["Light", "Dark"], value=st.session_state.theme)

        st.markdown("###### Navigation")
        st.session_state.nav_page = st.radio(
            "Go to",
            [
                "Applicant Portal (Dual Engine)",
                "Thin-File Alternate Scoring Engine",
                "Bank Admin Dashboard",
                "System Analytics & Financial Inclusion",
            ],
            index=[
                "Applicant Portal (Dual Engine)",
                "Thin-File Alternate Scoring Engine",
                "Bank Admin Dashboard",
                "System Analytics & Financial Inclusion",
            ].index(st.session_state.nav_page),
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown("<p class='subtle'>Engine A: XGBoost + CatBoost + LightGBM Stacking Ensemble<br>Engine B: Alternate-Data Gradient Boosting<br>Fraud Filter: Isolation Forest<br>XAI: SHAP / Simulated Shapley</p>", unsafe_allow_html=True)

    system = get_system()

    if st.session_state.nav_page == "Applicant Portal (Dual Engine)":
        page_applicant_portal(system)
    elif st.session_state.nav_page == "Thin-File Alternate Scoring Engine":
        page_thinfile_lab(system)
    elif st.session_state.nav_page == "Bank Admin Dashboard":
        page_admin_dashboard(system)
    elif st.session_state.nav_page == "System Analytics & Financial Inclusion":
        page_analytics(system)


if __name__ == "__main__":
    main()
