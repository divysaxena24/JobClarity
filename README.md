<div align="center">

# JobClarity — Backend

**Production-grade, AI-assisted fraudulent job posting risk detection service featuring XGBoost classification, native TreeSHAP explainability, and privacy-conscious MLOps telemetry.**

[![Live Web App](https://img.shields.io/badge/Live%20App-jobclarity.vercel.app-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://jobclarity.vercel.app/)
[![Live API](https://img.shields.io/badge/Live%20API-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://jobclarity-0tz8.onrender.com)
[![Swagger Docs](https://img.shields.io/badge/Swagger%20UI-Docs-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://jobclarity-0tz8.onrender.com/docs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

<br />

![Python](https://img.shields.io/badge/Python-3.12.1-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.128.8-009688?style=flat&logo=fastapi&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.3.0-orange?style=flat&logo=xgboost&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.0-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![SHAP](https://img.shields.io/badge/TreeSHAP-Native%20C++-512BD4?style=flat&logo=python&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-v2.12.5-E92063?style=flat&logo=pydantic&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Non--Root%20(UID%201001)-2496ED?style=flat&logo=docker&logoColor=white)
![Tests](https://img.shields.io/badge/PyTest-363%20Passed%20(100%25)-brightgreen?style=flat&logo=pytest&logoColor=white)

</div>

---

## Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. Core Capabilities](#2-core-capabilities)
- [3. Machine Learning Pipeline](#3-machine-learning-pipeline)
- [4. Leakage-Aware Model Evaluation](#4-leakage-aware-model-evaluation)
- [5. SHAP Explainability Methodology & Limitations](#5-shap-explainability-methodology--limitations)
- [6. API Documentation & Endpoint Reference](#6-api-documentation--endpoint-reference)
- [7. Repository Directory Structure](#7-repository-directory-structure)
- [8. Installation & Local Setup](#8-installation--local-setup)
- [9. Python Dependencies & Environment Lockfile](#9-python-dependencies--environment-lockfile)
- [10. Environment Variables](#10-environment-variables)
- [11. Running Tests & Evaluation Suites](#11-running-tests--evaluation-suites)
- [12. Docker Setup & Non-Root Hardening](#12-docker-setup--non-root-hardening)
- [13. MLflow Experiment Tracking & Model Governance](#13-mlflow-experiment-tracking--model-governance)
- [14. Production Deployment Setup (Render)](#14-production-deployment-setup-render)
- [15. Security, Privacy & Responsible-Use Boundaries](#15-security-privacy--responsible-use-boundaries)
- [16. Known Technical Inconsistencies & Maintenance Notes](#16-known-technical-inconsistencies--maintenance-notes)

---

## 1. Project Overview

**JobClarity** is an AI-assisted risk detection microservice engineered to evaluate employment listings for markers of recruitment fraud, advance-fee scams, phishing, and deceptive hiring practices.

Unlike conventional black-box security classifiers, JobClarity combines statistical machine learning inference with **native TreeSHAP feature attributions**. Every analyzed job description receives a continuous fraud probability estimate, an integer risk score (0–100), an operational risk tier, and the exact positive and negative keywords that drove the model's decision.

### Essential Operational Context:
- **Risk Estimation, Not Verification**: Model outputs are **statistical risk estimates**, not legal or factual determinations of fraud or legitimacy.
- **Active Production Model**: The service serves traffic using the **Production Baseline** pipeline (`TF-IDF + XGBoost`).
- **Shadow Staging Candidate (Candidate D)**: A feature-augmented model (`TF-IDF + 21 Domain Indicators + XGBoost`) is accessible on `/predict/staging` for experimental shadow benchmarking. Following an independent leakage-aware audit, **Candidate D was formally rejected for production deployment due to recall degradation**, and is never exposed to production `/predict` traffic.

---

## 2. Core Capabilities

### ⚡ FastAPI High-Performance Prediction Service
- Fully asynchronous REST API serving predictions with sub-50ms inference latency.
- Strict Pydantic v2 input sanitization and schema contracts.
- Automatic interactive documentation via Swagger UI and OpenAPI 3.1.

### 🌲 Gradient-Boosted XGBoost Classifier
- Tuned tree ensemble trained over sparse, high-dimensional n-gram representations.
- Continuous fraud probability estimate $p \in [0.0, 1.0]$ and standardized integer risk score ($0–100$).
- Tiered risk classification: **Low** (0–25), **Medium** (26–50), **High** (51–75), and **Critical** (76–100).

### 🔍 Native C++ TreeSHAP Explainability
- Evaluates exact Shapley values directly using XGBoost's native `booster.predict(..., pred_contribs=True)`.
- Reconstructs exact model decision margins: $\sum \phi_i + \phi_0 = \text{margin}$.
- Completely isolated from inference failure: if explainability fails, the endpoint returns an empty factor list without failing the prediction.

### 🛡️ Defensive Input Validation & Sliding-Window Rate Limiting
- **Length Bounds**: Rejects payloads $< 20$ characters or $> 50,000$ characters with structured HTTP 422 errors.
- **Rate Limiting**: Built-in, in-memory sliding-window limiter enforcing 60 requests/minute per client IP.
- **Anti-Spoofing Proxy Awareness**: Safely parses `X-Forwarded-For` when deployed behind reverse proxies (Render, Cloudflare) and validates candidate strings with `ipaddress.ip_address` to prevent header injection. Healthcheck routes remain unthrottled.

### 🔒 Privacy-Preserving Structured Telemetry
- Thread-safe JSONL prediction logging (`logs/predictions.jsonl`).
- **Strict Privacy Guarantee**: Never persists raw job descriptions or PII. Logs non-reversible SHA-256 hashes (`input_sha256`), token lengths, risk scores, and latency metrics for drift auditing.

---

## 3. Machine Learning Pipeline

```
+---------------------------------------------------------------------------------------------------+
|                                     DATA & TRAINING PIPELINE                                      |
+---------------------------------------------------------------------------------------------------+
                                                  |
           EMSCAD Dataset (17,880 Total Postings: 866 Fraud [4.84%], 17,014 Legit)
                                                  |
                     Pairwise Cosine Similarity Graph (Threshold >= 0.90)
                       Graph Connected Component Clustering (12,337 clusters)
                                                  |
                  +-------------------------------+-------------------------------+
                  |                                                               |
                  v                                                               v
         Train Set (80%, N = 14,304)                                     Test Set (20%, N = 3,576)
      626 Fraud (4.38%), 13,678 Legit                                 240 Fraud (6.71%), 3,336 Legit
                  |                                                               |
                  v                                                               |
       Text Cleaning & Normalization                                              |
    (Strips HTML, URLs, phone, email,                                             |
       whitespace, lowercase tokenization)                                        |
                  |                                                               |
                  v                                                               |
         TfidfVectorizer (Fitted on Train Only)                                   |
       (5,000 max features, sublinear TF scaling,                                 |
           unigrams & bigrams, English stop words)                                |
                  |                                                               |
                  v                                                               |
           XGBoost Classifier Tuning                                              |
      (100 estimators, max depth 6, scale_pos_weight)                             |
                  |                                                               |
                  v                                                               v
      +------------------------+                                      +------------------------+
      |  models/best_model.pkl | -----------------------------------> | Out-of-Distribution   |
      |  models/tfidf_...pkl   |                                      | Evaluation Audit       |
      +------------------------+                                      +------------------------+
```

### 3.1 Dataset Description
The model was developed using the **EMSCAD** (Employment Scam Aegean Dataset) benchmark:
- **Total Postings**: 17,880
- **Fraudulent Postings**: 866 ($4.84\%$)
- **Legitimate Postings**: 17,014 ($95.16\%$)
- **Severe Class Imbalance**: ~1 fraudulent listing per 20 authentic listings.

### 3.2 Preprocessing (`src/preprocessing/text_cleaner.py`)
Incoming text passes through deterministic sanitization:
1. **Entity Decoding**: Converts HTML entities (`&amp;`, `&quot;`, `&#39;`).
2. **Structural Scrubbing**: Strips HTML tags (`<p>`, `<div>`, `<br>`).
3. **Contact Anonymization**: Removes raw URLs, email addresses, and phone numbers to prevent memorizing transient scam domains or spoofed recruiter emails.
4. **Case & Unicode Normalization**: Normalizes whitespace and converts text to lowercase.

### 3.3 Feature Engineering (`src/features/feature_builder.py`)
- **Production Baseline Vectorizer**: Unigrams and bigrams, 5,000 maximum vocabulary size, sublinear term-frequency scaling ($1 + \log(\text{tf})$), English stop-word filtering.
- **Strict Anti-Leakage Boundary**: The vectorizer is fitted **exclusively on training data**. Out-of-vocabulary tokens in test or serving traffic are silently discarded without altering feature dimensions.

---

## 4. Leakage-Aware Model Evaluation

### 4.1 Evolution of Evaluation Methodologies

Evaluating fraud detection on the EMSCAD benchmark requires careful handling of repeated job listings:

1. **Historical Row-Level Random Split (Day 2)**:
   - *Protocol*: Naive random stratified 80/20 train/test partition (`train_test_split`, seed=42; source: `artifacts/evaluation/day2/final_evaluation_report.json`).
   - *Reported Metrics*: Accuracy $97.82\%$, Fraud Precision $98.97\%$, Fraud Recall $55.49\%$, ROC-AUC $0.9492$, PR-AUC $0.7867$.
   - *Methodological Flaw*: High reported performance is distorted by exact and near-duplicate job listings syndicated across multiple locations, which leak verbatim training templates into the test set.

2. **Exact-Group Split (Day 3 Session 1)**:
   - *Protocol*: Grouped partitioning on exact MD5 description hashes (`StratifiedGroupKFold`, 5-fold; source: `artifacts/evaluation/day3/group_evaluation_report.json`).
   - *Reported Metrics*: PR-AUC $0.7029$, ROC-AUC $0.9270$, Recall $36.42\%$.
   - *Methodological Limitation*: Eliminates verbatim identical postings, but near-duplicate templates (e.g., listings with $85\%\text{--}95\%$ text similarity with minor city or recruiter edits) continue to leak across folds.

3. **Near-Duplicate Graph Cluster Split (Day 3 Session 3 & Phase 3 Audit)**:
   - *Protocol*: Postings with TF-IDF cosine similarity $\ge 0.90$ are joined as graph edges. Transitive closures (Union-Find) group postings into 12,337 disjoint component families. Clusters are atomically allocated to train (80%, $N = 14,304$) or held-out test (20%, $N = 3,576$).
   - *Evaluation Rigor*: Completely prevents cross-split template leakage, establishing an out-of-distribution evaluation benchmark.

---

### 4.2 Verified Evaluation Results on Held-Out Test Set ($N = 3,576$)

Metrics below are sourced from the Phase 3 independent audit (`artifacts/evaluation/phase3_audit/phase3_audit_report.md` and `phase3_audit_metrics.json`), evaluated across 1,000 empirical bootstrap iterations on the held-out near-duplicate cluster split ($N = 3,576$, 240 Fraud, 3,336 Legit):

| Metric | Production Baseline | Candidate D (Staging) | Paired Difference ($\Delta = \text{Base} - \text{Cand}$) [95% CI] | Empirical Observation |
| :--- | :---: | :---: | :---: | :--- |
| **PR-AUC (Average Precision)** | **0.7333** [0.6835, 0.7786] | 0.6225 [0.5640, 0.6753] | **+0.1108** [+0.0836, +0.1385] | Baseline empirical advantage in average precision |
| **ROC-AUC** | **0.9608** [0.9484, 0.9704] | 0.9257 [0.9083, 0.9421] | **+0.0351** [+0.0232, +0.0472] | Baseline achieves higher ranking separation |
| **Brier Score (Lower is better)** | **0.0403** [0.0348, 0.0461] | 0.0466 [0.0406, 0.0529] | **-0.0063** [-0.0080, -0.0048] | Baseline displays lower squared error |
| **Expected Calibration Error (ECE)**| **1.98%** | 2.41% | -0.43% | Baseline aligns closer with empirical frequency bins |
| **Recall at $\tau = 0.50$ (Default)** | **31.25%** (75/240 TP, 0 FP) | 16.67% (40/240 TP, 0 FP) | **+14.58%** | Candidate D suffered severe recall reduction |
| **Precision at $\tau = 0.25$** | **100.00%** (91/240 TP, 0 FP) | 80.72% (67/240 TP, 16 FP) | +19.28% precision | Baseline retains 100% precision down to 0.25 |
| **Upfront Payment Scam Detection** | **24.00%** (6/25 detected) | **0.00%** (0/25 detected) | +24.00% | Candidate D failed completely on advance-fee scams |

> **Evaluation Methodology & Verification Boundaries**:
> - Confidence intervals represent empirical 95% percentile bootstrap intervals derived from 1,000 resamples. No formal null-hypothesis testing or p-values were computed, and no Phase 4.1 statistical hypothesis verification was conducted.
> - The Phase 4 review (`artifacts/evaluation/phase4_review/production_model_decision.md`) was an architectural decision review evaluating the Phase 3 audit findings. It concluded that Candidate D's added regex heuristics caused decision-tree split fragmentation, degrading scam recall.
> - **Calibration Scope**: While Brier score ($0.0403$) and ECE ($1.98\%$) were measured during Phase 3, the production pipeline outputs raw sigmoid probabilities directly from the XGBoost model without post-hoc probability calibration curves (such as isotonic regression or Platt scaling). Output probabilities and risk scores represent statistical risk estimates, not legal proof of fraud or verified legitimacy.

### 4.3 Why Accuracy Alone is Misleading
In a dataset with $4.84\%$ positive prevalence, **accuracy is a dangerous and deceptive metric**:
- A degenerate dummy classifier predicting "Legitimate" for 100% of inputs achieves **93.29% accuracy** on the held-out test set while detecting **0 out of 240 scams**.
- Both Baseline ($95.39\%$ accuracy) and Candidate D ($94.41\%$ accuracy) appear superficially strong, yet Candidate D misses $83.33\%$ of scams at default threshold.
- Consequently, JobClarity prioritizes **Precision-Recall AUC (PR-AUC)**, recall at bounded precision thresholds, and Brier calibration scores.

### 4.4 Operational Threshold Trade-Off Analysis
Empirical performance across operating points on the held-out test set ($N = 3,576$, 240 Fraud, 3,336 Legit):

```
+-------------------------------------------------------------------------------------------------------------+
| Threshold (tau) | Precision | Recall | F1-Score | True Pos | False Pos | Flagged % | Operational Meaning    |
+-------------------------------------------------------------------------------------------------------------+
| 0.50 (Default)  | 100.00%   | 31.25% | 0.4762   |       75 |         0 |     2.10% | Zero false alarms; 68.8% missed |
| 0.33            | 100.00%   | 36.67% | 0.5366   |       88 |         0 |     2.46% | Safe capture: +13 scams caught  |
| 0.25            | 100.00%   | 37.92% | 0.5498   |       91 |         0 |     2.54% | Auto-Flag Sweet Spot: +16 scams |
| 0.14 (Review)   |  73.72%   | 47.92% | 0.5808   |      115 |        41 |     4.36% | Human Moderation Queue (Max F1) |
| 0.10            |  62.44%   | 55.42% | 0.5872   |      133 |        80 |     5.96% | High moderator noise (80 FP)    |
+-------------------------------------------------------------------------------------------------------------+
```

---

## 5. SHAP Explainability Methodology & Limitations

### 5.1 Methodology
JobClarity utilizes native XGBoost C++ TreeSHAP (`src/explainability/shap_explainer.py`):
1. **Input Matrix**: Incoming cleaned text is transformed into the 5,000-dimensional sparse TF-IDF vector $X$.
2. **Attribution Extraction**: The booster computes exact feature attributions:
   ```python
   contribs = booster.predict(xgb.DMatrix(X), pred_contribs=True)
   values = contribs[0][:-1]  # Exclude base bias term
```
3. **Filtering & Direction Mapping**: Zero-impact tokens ($|\phi| \le 10^{-6}$) are excluded. Features are sorted by absolute magnitude, mapped to vectorizer vocabulary, and tagged:
   - $\phi_i > 0 \rightarrow$ `"Increases fraud risk"`
   - $\phi_i < 0 \rightarrow$ `"Decreases fraud risk"`
4. **Fail-Safe Shielding**: Explainability execution is isolated in a `try/except` block. Any attribution calculation error returns `top_reasons: []`, guaranteeing that predictions never fail due to explainability issues.

### 5.2 Critical Limitations of SHAP Explanations
1. **Statistical Association $\neq$ Real-World Causation**: SHAP values indicate how strongly individual tokens influenced the mathematical decision margin of this specific XGBoost model. **They do not prove why an employer acted or whether fraud actually occurred**.
2. **Vocabulary Artifacts**: Standard terms (e.g., `services`, `based`, `phone`) may appear with negative or positive impacts due to co-occurrence distributions in historical training data rather than intrinsic legitimacy or risk.
3. **The "Corporate Camouflage" Blind Spot**: In Phase 3 audit analysis, **51.67% of missed scams** were verbatim or near-verbatim copies of authentic Fortune 500 job descriptions. Because the copied text is bona fide, TreeSHAP attributions will show genuine corporate terminology pushing the fraud score *down*, even though the listing is fraudulent.

---

## 6. API Documentation & Endpoint Reference

- **Live Production URL**: `https://jobclarity-0tz8.onrender.com`
- **Swagger UI**: [`/docs`](https://jobclarity-0tz8.onrender.com/docs)
- **OpenAPI Schema**: [`/openapi.json`](https://jobclarity-0tz8.onrender.com/openapi.json)

The deployed production OpenAPI schema (`/openapi.json`) defines four routes, categorized into **Active Production Endpoints** and a **Staging / Benchmark Endpoint**:

| Method | Endpoint | Category | Powered By | Description |
| :--- | :--- | :---: | :--- | :--- |
| `GET` | `/` | Production | FastAPI Router | Returns service metadata, API version, and documentation links. |
| `GET` | `/health` | Production | Memory Probe | Unmetered healthcheck verifying model and vectorizer memory status. |
| `POST` | `/predict` | Production | **Production Baseline** | Primary production prediction endpoint with TreeSHAP explainability. |
| `POST` | `/predict/staging` | Staging / Benchmark | **Candidate D** (Rejected) | Shadow benchmark endpoint for multi-tier decision action experimentation. |

---

### 6.1 Production Endpoint: `POST /predict`

This is the primary production endpoint for customer-facing applications (such as the JobClarity frontend). It executes the validated **Production Baseline** pipeline (`TF-IDF + XGBoost`).

#### Request Headers
```http
POST /predict HTTP/1.1
Host: jobclarity-0tz8.onrender.com
Content-Type: application/json
```

#### Request Schema (`PredictionRequest`)
- `job_description` (*string, required*): Minimum 20 characters, maximum 50,000 characters.

```json
{
  "job_description": "We are seeking a Remote Data Entry Specialist. Earn $4500 weekly with no experience needed. Wire transfer required for home office software."
}
```

#### Successful Response (`HTTP 200 OK`)
```json
{
  "prediction": "Fake",
  "fraud_probability": 0.8842,
  "fraud_risk_score": 88,
  "risk_level": "Critical",
  "top_reasons": [
    {
      "feature": "wire",
      "impact": 1.8421,
      "effect": "Increases fraud risk"
    },
    {
      "feature": "weekly",
      "impact": 1.2104,
      "effect": "Increases fraud risk"
    },
    {
      "feature": "entry",
      "impact": 0.4512,
      "effect": "Increases fraud risk"
    },
    {
      "feature": "experience",
      "impact": -0.6215,
      "effect": "Decreases fraud risk"
    }
  ],
  "model_version": "1.0.0"
}
```

#### Response Headers
```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 59
X-RateLimit-Reset: 60
```

---

### 6.2 Staging / Benchmark Endpoint: `POST /predict/staging`

> [!WARNING]
> **Staging / Non-Production Status**: While `POST /predict/staging` is registered in the deployed OpenAPI schema on Render for shadow benchmarking, it executes **Candidate D** (`staging_candidate_d_model.joblib`). Because Candidate D was **formally rejected for production promotion** in Phase 4 due to significant recall degradation, this endpoint is strictly intended for internal evaluation and comparative analysis. It must not be used for production client integrations.

#### Staging Response (`HTTP 200 OK`)
```json
{
  "job_description_snippet": "We are seeking a Remote Data Entry Specialist. Earn $4500 weekly with no experience...",
  "fraud_probability": 0.9125,
  "risk_tier": "HIGH_RISK",
  "action": "BLOCK",
  "calibrated_thresholds": {
    "review": 0.14,
    "high_risk": 0.33
  },
  "explanation_summary": null,
  "staging_version": "staging-candidate-d-1.0.0"
}
```
*(Note: The JSON key `calibrated_thresholds` reflects the Pydantic schema field name representing decision boundaries: $\tau < 0.14 \rightarrow \text{PASS}$, $0.14 \le \tau < 0.33 \rightarrow \text{FLAG\_FOR\_REVIEW}$, $\tau \ge 0.33 \rightarrow \text{BLOCK}$.)*

---

### Error Responses

#### 1. Input Validation Error (`HTTP 422 Unprocessable Entity`)
Returned when payload is $< 20$ chars, $> 50,000$ chars, missing, or non-string:
```json
{
  "detail": [
    {
      "type": "string_too_short",
      "loc": ["body", "job_description"],
      "msg": "String should have at least 20 characters",
      "input": "Too short"
    }
  ]
}
```

#### 2. Rate Limit Exceeded (`HTTP 429 Too Many Requests`)
Returned when a client IP exceeds 60 requests within a 60-second window:
```json
{
  "detail": "Rate limit exceeded. Maximum 60 requests per minute allowed."
}
```
*Headers include: `Retry-After: 42`, `X-RateLimit-Remaining: 0`.*

#### 3. Service Unavailable (`HTTP 503 Service Unavailable`)
Returned safely if model artifacts fail to load into memory:
```json
{
  "detail": "Model service is currently unavailable."
}
```

---

## 7. Repository Directory Structure

```
JobClarity/
├── artifacts/
│   ├── candidate/                    # Candidate split metadata and candidate models
│   └── evaluation/
│       ├── day2/                     # Stratified baseline validation reports
│       ├── day3/                     # Near-duplicate graph evaluation reports
│       ├── day4/                     # Staging candidate evaluation & latency benchmarks
│       ├── phase3_audit/             # Leakage-aware audit reports, CSVs & ROC/PR curves
│       ├── phase4_review/            # Production model retention decision report
│       └── phase7_audit/             # Production readiness audit & remediation reports
├── configs/                          # Experiment configuration files
├── data/
│   ├── raw/
│   │   └── fake_job_postings.csv     # EMSCAD dataset (17,880 rows)
│   └── processed/
│       └── cleaned_jobs.csv          # Preprocessed dataset
├── models/                           # Production model artifacts (frozen)
│   ├── tfidf_vectorizer.pkl          # 5,000-feature TfidfVectorizer (Serialized under 1.9.0)
│   ├── xgboost_model.pkl             # Initial baseline XGBoost Booster
│   └── xgboost_model_tuned.pkl       # Tuned production XGBoost Booster
├── logs/
│   └── predictions.jsonl             # Privacy-preserving runtime telemetry (git-ignored)
├── notebooks/                        # EDA, prototype vectorizers, and research notebooks
├── scripts/
│   ├── run_phase3_audit.py           # Independent leakage-aware evaluation runner
│   ├── run_day6_mlflow_logging.py    # MLflow experiment tracking reproduction script
│   └── run_staging_validation.py     # Staging candidate verification script
├── src/
│   ├── api/
│   │   ├── main.py                   # FastAPI initialization, CORS, and root metadata
│   │   ├── predictor.py              # Production inference execution & risk scoring
│   │   ├── rate_limiter.py           # Sliding-window rate limiter & proxy IP extraction
│   │   ├── routes.py                 # Endpoint router (/health, /predict, /predict/staging)
│   │   └── schemas.py                # Pydantic v2 request and response contracts
│   ├── explainability/
│   │   ├── candidate_shap_explainer.py # TreeSHAP extractor for Candidate D
│   │   └── shap_explainer.py         # Native C++ TreeSHAP explainer for Production Baseline
│   ├── features/
│   │   ├── feature_builder.py        # Text concatenation and TF-IDF matrix builder
│   │   ├── feature_pipeline.py       # Composite FeatureUnion pipeline
│   │   ├── suspicious_language.py    # 21 heuristic regex indicator extractors
│   │   └── tfidf_vectorizer.py       # Vectorizer building blocks
│   ├── models/
│   │   ├── evaluate.py               # Metric calculation (PR-AUC, ROC-AUC, Brier)
│   │   ├── experiment_tracker.py     # MLflow logging, sanitization, and PyFunc wrappers
│   │   ├── hyperparameter_tuner.py   # Bayesian hyperparameter optimization
│   │   ├── model_loader.py           # Thread-safe artifact deserialization
│   │   ├── model_trainer.py          # XGBoost training pipeline
│   │   ├── staging_predictor.py      # Candidate D predictor class
│   │   └── threshold_optimizer.py    # Pareto frontier threshold optimization
│   ├── preprocessing/
│   │   └── text_cleaner.py           # Regex text normalization & HTML/URL stripping
│   └── validation/
│       ├── dataset_audit.py          # Dataset schema and leakage auditor
│       ├── group_evaluator.py        # Group-aware cross-validation evaluator
│       └── near_duplicate_grouper.py # Union-Find graph connected components clustering
├── tests/                            # Automated test suite (363 tests passing)
│   ├── conftest.py                   # Test fixtures, mock models, and TestClient setup
│   ├── test_api.py                   # Route status, input boundary & 422 validation tests
│   ├── test_candidate_explainability.py # Candidate D TreeSHAP tests
│   ├── test_deployment_readiness.py  # Dockerfile non-root, .dockerignore, CORS tests
│   ├── test_explainability.py        # Baseline TreeSHAP formatting & fail-safe tests
│   ├── test_near_duplicate_grouper.py# Graph clustering & Union-Find tests
│   ├── test_rate_limiter.py          # Sliding-window rate limiting & anti-spoofing tests
│   └── test_staging_validation.py    # Staging threshold boundaries and schema tests
├── Dockerfile                        # Hardened container definition (non-root UID 1001)
├── pytest.ini                        # PyTest configuration and filter warnings
├── requirements.txt                  # Top-level dependencies
├── requirements.lock                 # Pinned reproducible dependency lockfile
└── README.md                         # Project documentation
```

---

## 8. Installation & Local Setup

### Prerequisites
- Python 3.12+ (Python 3.12.1 tested)
- Git

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/divysaxena24/JobClarity.git
   cd JobClarity
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows (PowerShell):
   .\.venv\Scripts\Activate.ps1
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install pinned dependencies:**
   ```bash
   # Option A: Install from reproducible lockfile (Recommended)
   pip install -r requirements.lock

   # Option B: Install standard top-level requirements
   pip install -r requirements.txt
   ```

4. **Launch the FastAPI development server:**
   ```bash
   uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
   ```

5. **Verify API availability:**
   Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) in your browser.

---

## 9. Python Dependencies & Environment Lockfile

JobClarity separates **production runtime/test dependencies** from **development/research dependencies** to ensure clean, conflict-free dependency resolution and minimal container attack surfaces:

- **Production Runtime & Testing**: Defined in [`requirements.txt`](requirements.txt) and pinned in [`requirements.lock`](requirements.lock).
- **Development & Research**: Defined in [`requirements-dev.txt`](requirements-dev.txt) for offline experiment tracking (`mlflow`), pipeline expectation audits (`great_expectations`), Bayesian optimization (`optuna`), exploratory UI demos (`gradio`), and code quality tooling (`black`, `flake8`, `jupyter`).

The production environment is pinned in [`requirements.lock`](requirements.lock):

| Package | Pinned Version | Role in JobClarity |
| :--- | :---: | :--- |
| `fastapi` | `0.128.8` | REST API framework |
| `uvicorn` | `0.41.0` | Production ASGI web server |
| `pydantic` | `2.12.5` | Data validation and schema enforcement |
| `starlette` | `0.52.1` | Underlying ASGI toolkit (fully compatible with FastAPI 0.128.8) |
| `xgboost` | `3.3.0` | Gradient-boosted decision trees and native TreeSHAP |
| `scikit-learn` | `1.5.0` | TF-IDF vectorization and evaluation metrics (see Note below) |
| `shap` | `0.46.0` | Interpretability and explainable AI |
| `pandas` | `2.2.0` | DataFrame processing and dataset handling |
| `numpy` | `1.26.4` | Matrix computations and numerical operations |
| `scipy` | `1.15.3` | Sparse matrix arithmetic |
| `nltk` | `3.10.0` | Natural language tokenization utilities |
| `matplotlib` | `3.10.8` | Metric visualization and evaluation curve computations |
| `pytest` | `9.0.2` | Automated testing framework |
| `httpx` | `0.28.1` | HTTP client for endpoint integration tests |

> **Note on `scikit-learn` Version Compatibility**: While `requirements.lock` pins `scikit-learn==1.5.0`, the serialized vectorizer artifact (`models/tfidf_vectorizer.pkl`) reports having been serialized under version `1.9.0`. Compatibility across versions remains an unresolved technical finding pending controlled verification (see [Section 16](#16-known-technical-inconsistencies--maintenance-notes)). Production model artifacts are preserved as frozen binaries without retraining or re-serializing.

---

## 10. Environment Variables

The backend reads configuration from the environment, with safe local defaults:

```bash
# ==============================================================================
# JobClarity Backend Configuration (.env)
# ==============================================================================

# Server port binding (honors Render dynamic $PORT)
PORT=8000

# Application environment metadata
APP_NAME=JobClarity
APP_ENV=production
APP_VERSION=1.0.0

# CORS Allowed Origins (Comma-separated)
# Defaults include localhost:3000 and https://jobclarity.vercel.app
ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000,https://jobclarity.vercel.app

# API Rate Limiting (Requests per minute per client IP)
RATE_LIMIT_PER_MINUTE=60

# Trust Reverse Proxy Headers (Set to "true" on Render/Cloudflare/AWS)
TRUST_PROXY_HEADERS=true

# Structured Prediction Telemetry Log File Path
PREDICTION_LOG_PATH=logs/predictions.jsonl

# MLflow Tracking Store URI
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
```

---

## 11. Running Tests & Evaluation Suites

### 11.1 Automated Test Suite
JobClarity maintains **363 automated unit, integration, ML, and regression tests** passing with $100\%$ success rate:

```bash
# Run the complete test suite
pytest

# Run with verbose test-by-test breakdown
pytest -v

# Run with coverage report
pytest --cov=src --cov-report=term-missing
```

#### Test Suite Breakdown (363 Passing Tests):
- `tests/test_api.py` (20 tests): Endpoints, input boundary lengths (20 and 50,000 chars), 422 errors, 503 shielding.
- `tests/test_rate_limiter.py` (13 tests): Sliding window enforcement, proxy IP validation, anti-spoofing, 429 status.
- `tests/test_deployment_readiness.py` (16 tests): Dockerfile non-root user, .dockerignore rules, CORS origin parsing.
- `tests/test_explainability.py` (14 tests): TreeSHAP formatting, top-k limits, feature alignment, exception fallbacks.
- `tests/test_candidate_explainability.py` (14 tests): Candidate D local accuracy, feature categorization.
- `tests/test_near_duplicate_grouper.py` (42 tests): Union-Find transitive components, residual leakage audits.
- `tests/test_ml_pipeline.py` (37 tests): Training determinism, OOV isolation, cross-validation tuning.
- `tests/test_staging_validation.py` (67 tests): Multi-tier calibrated threshold boundaries ($0.14 / 0.33$), NaN/Inf rejection.
- Other modules (140 tests): Preprocessing, feature builders, dataset audits, MLflow recovery, predictor boundaries.

### 11.2 Evaluation & Audit Scripts
```bash
# Run the leakage-aware Phase 3 evaluation audit (generates curves and metric JSONs)
python scripts/run_phase3_audit.py

# Run MLflow tracking reproduction and model recovery parity verification
python scripts/run_day6_mlflow_logging.py
```

---

## 12. Docker Setup & Non-Root Hardening

The backend container configuration is specified in [`Dockerfile`](Dockerfile) with the following implemented controls:
- **Base Image**: `python:3.12-slim` (minimal attack surface).
- **Non-Root Execution**: Creates dedicated system group `appgroup` (GID 1001) and system user `appuser` (UID 1001) with non-login shell `/sbin/nologin`.
- **Filesystem Permissions**: Sets non-root ownership on `/app` and creates `/app/logs` owned by `appuser:appgroup`. Container executes under `USER appuser`.
- **Reproducible Dependencies**: Conditionally installs from `requirements.lock` when present in the build context.
- **Dynamic Port**: Honors `${PORT:-8000}` dynamically via entrypoint `sh -c "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"`.
- **Healthcheck**: Uses a native Python probe (`urllib.request.urlopen`) checking `http://localhost:${PORT:-8000}/health` every 30 seconds.

> **Environment Verification Note**: The container image has been built from clean cache (`docker build --no-cache -t jobclarity:prod .`) and verified locally: non-root user execution (`appuser`, UID 1001), `/health` endpoint response (`200 OK`, `model_loaded: true`), and complete backend test execution have been validated inside the resulting image.

```bash
# Build the hardened Docker image
docker build -t jobclarity-api:latest .

# Run the container
docker run -d -p 8000:8000 --name jobclarity jobclarity-api:latest

# Verify health
curl http://localhost:8000/health
```

---

## 13. MLflow Experiment Tracking & Model Governance

JobClarity integrates MLflow for local experiment logging without external cloud dependencies:

```bash
# Launch MLflow UI backed by local SQLite database
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) to inspect:
- Model runs, hyperparameters, PR-AUC, ROC-AUC, and Brier scores.
- Artifacts: confusion matrices, feature importance plots, and serialized PyFunc wrappers.
- Model Registry: `JobClarity-Baseline` (Active Production) and `JobClarity-Candidate-D` (Shadow Staging).

---

## 14. Production Deployment Setup (Render)

The backend is continuously deployed on **Render** as a containerized Web Service:
- **Environment**: Docker (`Dockerfile`).
- **Build Command**: Automatically executed from `Dockerfile`.
- **Start Command**: Dynamically handled by `CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]`.
- **Health Check Path**: `/health`.
- **Live URL**: `https://jobclarity-0tz8.onrender.com`.

---

## 15. Security, Privacy & Responsible-Use Boundaries

1. **Statistical Risk Estimates, Not Verifications**: JobClarity evaluates linguistic and structural risk signals. A "Low Risk" verdict does not verify employer legitimacy; a "High Risk" verdict does not constitute legal proof of a crime.
2. **Text-Theft / Corporate Camouflage**: The model evaluates text alone. Postings that steal authentic job descriptions from legitimate enterprises cannot be caught by text NLP alone and require manual recruiter verification.
3. **Privacy by Design**: Runtime prediction logging records zero raw text and zero PII. All stored telemetry utilizes irreversible SHA-256 hashes (`input_sha256`) and length metrics.
4. **Mandatory Due Diligence**: Job seekers must never wire money, cash unverified checks, purchase equipment upfront, or disclose banking passwords.

---

## 16. Known Technical Inconsistencies & Maintenance Notes

During the Phase 7 audit, the following technical items were cataloged:

1. **Scikit-Learn Artifact Serialization Discrepancy**:
   - *Observation*: Deserializing `models/tfidf_vectorizer.pkl` raises a runtime warning:
     ```
     InconsistentVersionWarning: Trying to unpickle estimator TfidfVectorizer from version 1.9.0 when using version 1.5.0.
     ```
   - *Technical Fact*: `requirements.lock` pins `scikit-learn==1.5.0`, whereas the serialized vectorizer pickle metadata reports version `1.9.0`.
   - *Compatibility Status*: Passing test suite runs do not establish safe cross-version pickle compatibility or guarantee safety against subtle numerical or tokenization shifts across scikit-learn releases. Furthermore, re-serialization or retraining has not been established or validated as an approved fix. Compatibility remains an unresolved technical issue pending controlled verification in a staging test environment.
2. **Candidate D Staging Rejection & Production Baseline Retention**:
   - The primary production endpoint `/predict` is powered by the **Production Baseline** pipeline.
   - The staging endpoint `/predict/staging` evaluates **Candidate D** strictly for shadow benchmarking. The Phase 4 architectural review established that Candidate D's hand-crafted regex heuristics caused decision-tree split fragmentation, degrading held-out PR-AUC from $0.7333$ to $0.6225$ and cutting scam recall from $31.25\%$ to $16.67\%$ at $\tau=0.50$. Consequently, Candidate D was **formally rejected for production promotion** and is not deployed for production traffic.

---

<div align="center">

**JobClarity** — *Detect fake jobs. Restore trust in employment.*

[🌐 Live Application](https://jobclarity.vercel.app/) • [📖 API Documentation](https://jobclarity-0tz8.onrender.com/docs) • [🎨 Frontend Repository](https://github.com/divysaxena24/JobClarity-frontend)

</div>
