# Hybrid ML Loan Approval & Alternate Credit Assessment System

A production-ready, dual-evaluation FinTech loan underwriting platform built with
**Python + Streamlit**, designed to serve **both** traditional borrowers (with CIBIL
bureau history) and **Thin-File / Unbanked borrowers** (small vendors, shopkeepers,
gig workers, freelancers) through a **Dual-Engine Machine Learning Architecture**,
with a real-time fraud filter and full Explainable AI (XAI) decision breakdowns.

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Streamlit Dashboard (app.py)                │
│  Applicant Portal │ Thin-File Lab │ Admin Dashboard │ Analytics      │
└───────────────┬───────────────────────────────────────┬─────────────┘
                │                                        │
     ┌──────────▼──────────┐                 ┌───────────▼───────────┐
     │   Engine A           │                 │   Engine B             │
     │ Standard Credit      │                 │ Thin-File Alternate    │
     │ Bureau Engine        │                 │ Credit Engine          │
     │                       │                 │                        │
     │ Stacking Ensemble:    │                 │ Gradient-Boosted       │
     │ XGBoost + CatBoost +  │                 │ Classifier on:         │
     │ LightGBM + Logistic   │                 │ UPI turnover, cash-    │
     │ Regression (meta)     │                 │ flow volatility,       │
     │                       │                 │ utility/rent           │
     │ Input: CIBIL score,   │                 │ reliability, alternate │
     │ income, DTI, etc.     │                 │ credit score (0-800)   │
     └──────────┬────────────┘                 └───────────┬───────────┘
                │                                           │
                └───────────────────┬───────────────────────┘
                                     │
                       ┌─────────────▼─────────────┐
                       │  Fraud & Anomaly Filter    │
                       │  Isolation Forest          │
                       └─────────────┬─────────────┘
                                     │
                       ┌─────────────▼─────────────┐
                       │  XAI Explainer             │
                       │  SHAP / Simulated Shapley  │
                       │  additive attributions     │
                       └────────────────────────────┘
```

### Engine A — Standard Credit Bureau Engine
A **Stacking Ensemble** combining `XGBoost`, `CatBoost`, and `LightGBM` base
learners with a `LogisticRegression` meta-learner (`sklearn.ensemble.StackingClassifier`,
`stack_method="predict_proba"`, 5-fold internal CV). Trained on CIBIL score, income,
employment history, existing obligations, DTI ratio, and requested loan terms.

> If `xgboost`, `catboost`, or `lightgbm` are not installed in the runtime
> environment, `src/predict.py` automatically substitutes an equivalent
> scikit-learn boosted-tree model (`HistGradientBoostingClassifier`,
> `GradientBoostingClassifier`, `ExtraTreesClassifier`) for that specific base
> learner, so the system **always runs end-to-end**, in any environment.

### Engine B — Thin-File / Alternate-Data Credit Engine
A gradient-boosted classifier trained purely on **behavioural / alternate data**:
UPI transaction volume & frequency, cash-flow volatility, average daily balance,
utility & rent payment reliability, transaction history depth, and the derived
**Alternate Credit Score (0–800)** computed by `src/alternate_credit_scoring.py`.

### Fraud & Anomaly Filter
An `IsolationForest` trained on the same preprocessed feature space as each engine,
flagging statistically anomalous applications (e.g. artificial UPI turnover spikes,
mismatched income-to-expense ratios) for **manual review** instead of an automated
decision.

### Explainable AI (XAI) Module
`src/xai_explainer.py` produces additive, SHAP-style per-feature contribution
values for every prediction:
- **Primary path:** the real `shap` library (`shap.Explainer`, permutation
  algorithm) — true Shapley-value explanations.
- **Automatic fallback:** a dependency-free, sampling-based Shapley-value
  estimator (Monte-Carlo marginal-contribution averaging over random feature
  coalitions — the same strategy underlying `KernelSHAP`/`SamplingSHAP`), used
  transparently if `shap` is unavailable or fails for a given model. Output
  schema is identical either way, so the dashboard never needs to know which
  path was used.

---

## 2. Repository Structure

```
Hybrid_ML_Loan_Approval_System/
├── app.py                          # Main Streamlit dashboard (4 pages, custom CSS)
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py       # Synthetic data generation + preprocessing pipelines
│   ├── alternate_credit_scoring.py # 0-800 Alternate Credit Score algorithm
│   ├── predict.py                  # Dual-engine ML models + Isolation Forest + orchestrator
│   └── xai_explainer.py            # SHAP / simulated-SHAP explainability engine
├── data/                            # Auto-generated synthetic datasets + application log (gitkeep)
└── models/                          # Auto-trained model artifacts, cached after first run (gitkeep)
```

---

## 3. Setup & Installation

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the application
streamlit run app.py
```

The app will open at `http://localhost:8501`.

> **First run note:** on the very first launch, the system has no cached models
> in `models/`, so it will automatically generate synthetic training data (in
> `data/`), train both engines + the Isolation Forest fraud filter, and cache
> everything via `@st.cache_resource` + `joblib`. This one-time bootstrap takes
> roughly 10–30 seconds depending on your machine and which of
> XGBoost/CatBoost/LightGBM are installed. Every subsequent run loads instantly
> from the cached artifacts in `models/`.

To force a full retrain (e.g. after changing the synthetic data generators),
delete the contents of `models/` and `data/` and relaunch the app.

---

## 4. Application Pages

1. **Applicant Portal (Dual Engine)** — multi-step interactive form with a toggle
   between *"Has CIBIL Score"* and *"First-Time / Thin-File Borrower"*, dynamically
   routing the applicant to Engine A or Engine B. Returns approval probability,
   risk grade (Low/Medium/High), max eligible loan amount, fraud-flag status, and
   a full SHAP-style XAI breakdown.
2. **Thin-File Alternate Scoring Engine** — a dedicated lab to simulate a UPI
   monthly statement, inspect daily turnover volatility, and build the 0–800
   Alternate Credit Score component-by-component, with an optional quick
   eligibility check against Engine B.
3. **Bank Admin Dashboard** — a live table of recent applications (from
   `data/applications_log.csv`), red-banner fraud/anomaly alerts, alternate-data
   score logs, and manual override controls.
4. **System Analytics & Financial Inclusion** — portfolio-level Plotly analytics:
   approval rate by engine, credit risk distribution, traditional vs. alternate
   loan volume, and a financial-inclusion impact summary.

---

## 5. Design Notes

- All "production" ML training in this repository runs against **synthetic but
  behaviourally-realistic data** generated on first launch — this makes the
  system 100% self-contained and demoable with zero external data dependencies.
  To connect to real bureau / UPI data providers in production, replace the
  `get_or_create_*_dataset()` functions in `src/data_preprocessing.py` with your
  real data-ingestion pipeline; the rest of the architecture (preprocessing →
  stacking ensemble → fraud filter → XAI) is data-source agnostic.
- The Isolation Forest fraud filter uses `contamination=0.06` (~6% of the
  population flagged) — tune this in `src/predict.py` (`FraudAnomalyDetector`)
  to match your institution's risk appetite.
- Max eligible loan amounts use a simplified FOIR (traditional) / turnover-multiple
  (thin-file) affordability model in `src/predict.py` — replace with your
  institution's actual affordability policy for production use.
