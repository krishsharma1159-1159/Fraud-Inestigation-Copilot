# Arthadrishti — AI-Powered Fraud Investigation Workspace

> **Arthadrishti (अर्थदृष्टि)** is a production-grade, explainable fraud investigation copilot designed for AML/Fraud compliance analysts. It bridges raw tabular datasets, graph network topologies, tree-based machine learning, SHAP mathematical attribution, and LLM synthesis inside a single deterministic, hallucination-proof workspace.

---

## 📑 Table of Contents
1. [Overview & Architecture](#-overview--architecture)
2. [What We Built (Complete System Breakdown)](#-what-we-built-complete-system-breakdown)
3. [Project Directory Structure](#-project-directory-structure)
4. [Prerequisites & Setup](#-prerequisites--setup)
5. [How to Run the Project](#-how-to-run-the-project)
6. [API Endpoints Reference](#-api-endpoints-reference)
7. [5 Curated Test Cases](#-5-curated-test-cases)
8. [Automated Test Suite](#-automated-test-suite)
9. [Security & Isolation Boundaries](#-security--isolation-boundaries)

---

## 🏛️ Overview & Architecture

Modern fraud analysts are overwhelmed by raw probability scores without explanation or network context. Arthadrishti solves this with an end-to-end evidence chain:

```
                  IEEE-CIS Transaction Dataset
                              │
                              ▼
           Feature Engineering Pipeline (506 Features)
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
   LightGBM Tuned Fraud Model      SHAP TreeExplainer
  (Predicts Fraud Probability)    (Margin & Additivity Attribution)
               │                             │
               └──────────────┬──────────────┘
                              ▼
                 TigerGraph FraudGraph Layer
              (14 Installed Read-Only Queries)
                              │
                              ▼
            Deterministic Evidence Engine (R01 - R08)
                              │
                              ▼
                    Evidence Validation Gate
                              │
                              ▼
             OpenRouter Multi-Model LLM Layer
        (Grounded Narrative Summary from Evidence Only)
                              │
                              ▼
             FastAPI Production REST API (Port 8000)
                              │
                              ▼
        Arthadrishti Frontend Workspace (code.html)
     (Ice Glass UI, Interactive SVG Graph, Audit Stream)
```

---

## 🛠️ What We Built (Complete System Breakdown)

### 1. Feature Engineering & Dataset Pipeline (`src/evidence_engine.py`)
- Engineered **506 tabular features** from the IEEE-CIS Fraud Detection dataset.
- Added temporal delta aggregations (`time_since_prev_tx`, `tx_count_24h`, `tx_amt_sum_24h`), cardholder expenditure baselines (`card1_mean`, `card1_std`), z-score deviations (`amt_card1_zscore`), and cyclic trigonometric time encoding (`sin_tx_hour`, `cos_tx_hour`).

### 2. LightGBM Tuned Model (`models/tuned/lightgbm_tuned.pkl`)
- Calibrated `LGBMClassifier` (`n_estimators=220`, `learning_rate=0.04`, `num_leaves=150`, `random_state=47`).
- Strictly evaluated in evaluation margin and calibrated probability space.

### 3. SHAP Model Explainability (`src/shap_explainer.py`)
- Implemented SHAP `TreeExplainer` providing local mathematical attributions for all 506 features.
- Strict consistency validation: ensures base value + sum of SHAP values matches raw model log-odds margin within `1e-5` tolerance (`Sigmoid(margin) == predict_proba`).

### 4. TigerGraph FraudGraph Layer
- Configured graph database schema with `Transaction`, `Card`, `Device`, `Address`, and `Email` vertices.
- 14 verified read-only GSQL queries installed:
  - `get_transaction_details`, `get_transaction_card`, `get_transaction_device`, `get_transaction_address`, `get_transaction_emails`
  - `get_card_related_transactions`, `get_device_related_transactions`, `get_address_related_transactions`
  - `get_p_email_related_transactions`, `get_r_email_related_transactions`
  - `get_temporal_transaction_context`, `get_transaction_entity_counts`, `get_entity_temporal_status`
  - `get_transaction_investigation_context`

### 5. Deterministic Reason Codes (`src/evidence_engine.py`)
- Rules `R01` through `R08` computed deterministically from verified data:
  - `R01`: Amount deviation vs card historical mean.
  - `R02`–`R05`: Entity graph degrees (Card, Device, Address, Email Domain connectivity).
  - `R06`: Short-interval repeated transactions (< 300s).
  - `R07`–`R08`: 24-hour burst frequency and volume velocity.

### 6. Evidence Validation Gate (`src/evidence_engine.py:validate_evidence`)
- Hard gate verifying:
  - Valid target transaction ID.
  - Target transaction is **never** included in its own supporting transactions.
  - No future transactions are labeled as historical.
  - Prediction matches model output within `1e-10` floating tolerance.

### 7. Multi-Model OpenRouter LLM Layer (`src/investigation_llm.py`)
- OpenRouter OpenAI-compatible API client with strict system instructions prohibiting external knowledge, identity fabrication, and hallucinated facts.
- Automatic payload optimization (<15 KB) prioritizing card, device, and temporal proximity.
- Supports single-model summaries and multi-model consensus across:
  - `openai/gpt-4.1-mini`
  - `deepseek/deepseek-v4.1-flash`
  - `google/gemini-3.8-flash`
  - `qwen/qwen3.8-max-0902`
  - `qwen/qwen3.5-flash-02-23`
- Deterministic mock fallback mode for offline testing.

### 8. Production-Grade FastAPI Backend (`backend/`)
- High-performance asynchronous REST API built with FastAPI and Pydantic v2.
- **Endpoints**: Health telemetry, single transaction investigation, multi-model investigation, forensic JSON export, similar cases vector lookup, analyst feedback, compliance review escalation, and audit trail lookup.
- **Layout Math Engine**: Dynamically calculates `(x, y)` SVG coordinates for target transactions, identity hubs, and supporting nodes.
- **Audit Logging**: Persists analyst decisions to `data/audit_log.json`.

### 9. Interactive Frontend Workspace (`code.html`)
- Built with Tailwind CSS, Ice Glass aesthetic, and native SVG rendering.
- Connected via asynchronous JavaScript to FastAPI endpoints.
- Features interactive zoom/pan, active node inspection panel, supporting transactions ledger, chronological activity timeline, dual-mode brief comparison (LLM vs Rule Template), and human sign-off controls.

### 10. Comprehensive Test Suite (`tests/`)
- 41 automated unit, integration, and regression tests (100% passing).

---

## 📁 Project Directory Structure

```
.
├── backend/
│   ├── routes/
│   │   ├── actions.py               # Feedback, review escalation, audit trail routes
│   │   ├── health.py                # System health diagnostics (LightGBM, TG, LLM)
│   │   └── investigation.py         # Investigation, export, and similar cases routes
│   ├── services/
│   │   ├── audit_service.py         # Atomic persistence of analyst dispositions
│   │   └── investigation_service.py # Core pipeline orchestration & SVG graph layout math
│   ├── main.py                      # FastAPI app, CORS, route mounts, code.html root serving
│   ├── schemas.py                   # Pydantic v2 request/response contracts
│   └── README.md                    # Backend microservice documentation
├── src/
│   ├── evidence_engine.py           # Feature engineering, model runner, TG client, reason codes
│   ├── investigation_pipeline.py    # End-to-end investigation orchestration
│   ├── investigation_llm.py         # OpenRouter multi-model LLM layer & anti-hallucination guard
│   └── shap_explainer.py            # SHAP TreeExplainer & margin additivity validator
├── tests/
│   ├── test_backend_api.py          # FastAPI endpoint integration tests (10 tests)
│   ├── test_evidence_engine.py      # Feature engineering & reason codes tests (3 tests)
│   ├── test_investigation_pipeline.py# Full E2E pipeline verification (10 tests)
│   ├── test_investigation_llm.py    # LLM output validation & safety tests (5 tests)
│   ├── test_shap_explainer.py       # SHAP explanation & additivity tests (4 tests)
│   └── test_openrouter_integration.py# Live OpenRouter API connectivity tests (1 test)
├── code.html                        # Arthadrishti interactive frontend workspace
├── requirements.txt                 # Project dependencies
├── .env.example                     # Environment configuration template
├── .gitignore                       # Git ignore rules for secrets and large binaries
└── README.md                        # Project documentation (this file)
```

---

## ⚙️ Prerequisites & Setup

### Requirements
- **Python**: 3.10 to 3.14
- **Operating System**: Windows, Linux, or macOS

### 1. Clone the Repository
```bash
git clone https://github.com/<your-org>/arthadrishti.git
cd arthadrishti
```

### 2. Create and Activate a Virtual Environment
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Open `.env` and fill in your credentials:
```ini
# OpenRouter LLM Configuration
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your_openrouter_api_key_here
LLM_MODEL=openai/gpt-4.1-mini
LLM_MODELS=openai/gpt-4.1-mini,deepseek/deepseek-v4.1-flash,google/gemini-3.8-flash,qwen/qwen3.8-max-0902,qwen/qwen3.5-flash-02-23

# TigerGraph Cloud Configuration
TG_HOST=https://your-tigergraph-host.i.tgcloud.io
TG_GRAPH=FraudGraph
TG_SECRET=your_tigergraph_secret_here

# Paths
MODEL_PATH=models/tuned/lightgbm_tuned.pkl
BEH_PATH=data skillcred/train_behavioral.parquet
```

---

## 🚀 How to Run the Project

### Start the FastAPI Backend
Run the following command from the project root:
```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### Access the Applications
- **Interactive Workspace UI**: Open your browser at [http://localhost:8000/](http://localhost:8000/)
- **Swagger API Docs**: Explore interactive REST documentation at [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check Endpoint**: [http://localhost:8000/api/health](http://localhost:8000/api/health)

*(Note: `code.html` can also be opened via Live Server on port 5500; it automatically detects and connects to the backend at `http://localhost:8000`).*

---

## 🔌 API Endpoints Reference

| Method | Path | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves `code.html` frontend directly |
| `GET` | `/api/health` | System health check for LightGBM, TigerGraph, OpenRouter, and Parquet dataset |
| `POST` | `/api/investigate` | Runs full E2E pipeline for a single transaction ID |
| `POST` | `/api/investigate/multi-model` | Evaluates evidence across multiple OpenRouter LLMs simultaneously |
| `GET` | `/api/investigate/{id}/export` | Downloads full forensic case dossier as JSON |
| `GET` | `/api/similar-cases?transaction_id={id}` | Returns historically correlated similar cases |
| `POST` | `/api/actions/feedback` | Records analyst disposition (`CONFIRMED_SUSPICIOUS`, `NOT_SUSPICIOUS`, `NEEDS_REVIEW`) |
| `POST` | `/api/actions/review` | Escalates transaction to the Senior Compliance Review queue |
| `GET` | `/api/actions/audit-trail` | Returns compliance audit trail history |

---

## 🧪 5 Curated Test Cases

You can test these directly by typing the ID into Section 1 or clicking the live sample buttons:

### 1. `2987240` — Confirmed High-Risk Fraud (Live Dataset)
- **Prediction**: `SUSPICIOUS` | **Risk Score**: `86.3%`
- **Key Evidence**: Ground-truth fraud in IEEE-CIS. Card degree 348, Device degree 5, P-Email degree 45,250, R-Email degree 27,509.
- **Expected UI**: Section 2 risk bar turns deep red. Reason codes highlight high card connectivity and velocity. AI brief synthesizes risk signals.

### 2. `2987004` — Dense Legitimate Baseline (Live Dataset)
- **Prediction**: `NORMAL` | **Risk Score**: `0.8%` (High Confidence)
- **Key Evidence**: Baseline purchase of $50.00. Card 4497 (18 txns), Device Samsung SM-G892A (9 txns), Address 420/87 (3,581 txns).
- **Expected UI**: Green safe indicator. 5 deterministic reason codes within normal limits. 7-node interactive SVG graph rendered with Card and Device hubs.

### 3. `2987203` — Severe Amount Spike / Anomaly (Live Dataset)
- **Prediction**: `NORMAL / ELEVATED` | **Risk Score**: `27.5%`
- **Key Evidence**: High amount ($445.00) vs card average. Dense Card 18268 reuse (1,129 txns).
- **Expected UI**: Triggers `AMOUNT_SPIKE` reason code. Demonstrates how human analysts can click "Mark for Review" to escalate ambiguous cases.

### 4. `2987000` — Sparse Baseline Account (Live Dataset)
- **Prediction**: `NORMAL` | **Risk Score**: `8.2%`
- **Key Evidence**: Minimal device footprint transaction ($68.50), Card 13926 (6 txns), Address 315/87.
- **Expected UI**: Verifies that the system functions accurately with sparse data without hallucinating entities or connections.

### 5. `TXN1045` / Scenario 1 — Multi-Hop Mule Ring (Demo Preset)
- **Prediction**: `SUSPICIOUS` | **Risk Score**: `94.2%`
- **Key Evidence**: 4.48× amount deviation ($82,500 vs $18,400 avg). 24-minute burst velocity. 2-hop TigerGraph routing to Recipient B204 (Mule Aggregator) and onward hop to C301.
- **Expected UI**: Click nodes to inspect forensic properties. Click "Mode: Grounded Brief vs Template" in Section 7 to compare LLM reasoning with deterministic rule strings.

---

## 🧪 Automated Test Suite

To run all 41 unit, integration, and regression tests:

```bash
python -m unittest discover tests -v
```

### Test Coverage Breakdown:
- **`tests/test_backend_api.py`** (10 tests): Health check, valid/invalid investigate, feedback, review escalation, audit trail, dossier export, similar cases, and HTML serving.
- **`tests/test_investigation_pipeline.py`** (10 tests): E2E populated/sparse transaction flow, prediction consistency, SHAP consistency, target isolation, and temporal classification.
- **`tests/test_investigation_llm.py`** (5 tests): LLM prompt boundaries, safety filters, prohibited language enforcement, and schema validation.
- **`tests/test_evidence_engine.py`** (3 tests): Populated/sparse evidence construction and negative rule validation.
- **`tests/test_shap_explainer.py`** (4 tests): TreeExplainer initialization, 506-feature attribution, margin additivity, and invalid input rejection.
- **`tests/test_openrouter_integration.py`** (1 test): Live multi-model OpenRouter execution across 5 models.

---

## 🛡️ Security & Isolation Boundaries

1. **Strict Evidence-Only LLM Sandbox**: The LLM receives **only** the validated Evidence JSON dictionary. It has **no** direct access to TigerGraph, database connections, or the filesystem.
2. **Anti-Hallucination Guardrails**: Prompts strictly prohibit the LLM from declaring customer guilt, creating new reason codes, or assuming real-world legal identities.
3. **Deterministic Grounding**: Every metric in Section 3 and every edge in Section 4 is computed deterministically by LightGBM and TigerGraph.
4. **Secret Protection**: `.env` and sensitive API keys are strictly excluded via `.gitignore` and never logged or exposed to the client.

---

## 📄 License
This project is proprietary and maintained for the Arthadrishti Fraud Investigation Workspace.
