# Arthadrishti — FastAPI Backend Service

Production-grade REST backend for the **Arthadrishti — AI-Powered Fraud Investigation Workspace**.

## Architecture & Integration Pipeline

```
code.html Frontend (Vanilla HTML5 / Ice Glass UI)
        ↓
FastAPI Backend (Port 8000)
        ↓
Investigation Pipeline (src/investigation_pipeline.py)
  ├── 1. Feature Engineering (506 IEEE-CIS features)
  ├── 2. LightGBM Tuned Fraud Model (models/tuned/lightgbm_tuned.pkl)
  ├── 3. SHAP TreeExplainer (Additivity & Margin Verification)
  ├── 4. TigerGraph FraudGraph (14 Installed GSQL Queries)
  ├── 5. Deterministic Reason Codes (R01 - R08)
  └── 6. Evidence Engine & Validation Gate
        ↓
OpenRouter Multi-Model LLM Layer (src/investigation_llm.py)
        ↓
Structured UI Payload (schemas.py: InvestigationUIResponse)
        ↓
code.html Interactive Workspace (Sections 1 - 8)
```

## Available API Endpoints

### 1. Health & Telemetry
- `GET /api/health`: System health status across LightGBM, TigerGraph, OpenRouter, and Dataset.
- `GET /docs`: Interactive Swagger API documentation.

### 2. Forensic Investigation
- `POST /api/investigate`: Runs the end-to-end investigation pipeline for a transaction ID.
  - Body: `{"transaction_id": 2987004, "mock_llm": false}`
  - Returns complete UI schema for `code.html`.
- `POST /api/investigate/multi-model`: Dispatches validated Evidence JSON to all configured OpenRouter models.
- `GET /api/investigate/{transaction_id}/export`: Downloads complete forensic case dossier as JSON.
- `GET /api/similar-cases?transaction_id=2987004`: Returns historically correlated cases from the dataset.

### 3. Analyst Actions & Audit Trail
- `POST /api/actions/feedback`: Records analyst disposition sign-off (`CONFIRMED_SUSPICIOUS`, `NOT_SUSPICIOUS`, `NEEDS_REVIEW`).
- `POST /api/actions/review`: Escalates transaction to the Senior Compliance Review queue.
- `GET /api/actions/audit-trail`: Retrieves compliance audit trail entries.

## Running the Backend

Start the server using `uvicorn`:
```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Then visit:
- **Frontend Workspace**: `http://localhost:8000/`
- **Swagger Docs**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/api/health`
