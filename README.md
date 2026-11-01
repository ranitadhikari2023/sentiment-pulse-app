# Sentiment Pulse — Customer Feedback Intelligence Platform

[![FastAPI](https://img.shields.io/badge/FastAPI-0.143+-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.23+-005CED.svg?style=flat&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.0+-D71F00.svg?style=flat&logo=sqlalchemy&logoColor=white)](https://www.sqlalchemy.org/)
[![Alembic](https://img.shields.io/badge/Alembic-Migrations-red.svg?style=flat)](https://alembic.sqlalchemy.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-psycopg3-336791.svg?style=flat&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![PWA Ready](https://img.shields.io/badge/PWA-Installable-blueviolet.svg?style=flat)](https://web.dev/progressive-web-apps/)
[![Tailwind CSS](https://img.shields.io/badge/TailwindCSS-v3_Static-38B2AC.svg?style=flat&logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![Render](https://img.shields.io/badge/Deploy-Render-46E3B7.svg?style=flat&logo=render&logoColor=white)](https://render.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Sentiment Pulse** is a production-grade customer feedback intelligence platform combining high-throughput NLP inference with an accessible, high-performance web interface. Built on **FastAPI**, **PostgreSQL / SQLite**, **SQLAlchemy 2.0**, **Alembic**, and **ONNX Runtime (INT8 Quantized Transformer)**, Sentiment Pulse empowers customer experience (CX), product, and support teams to ingest reviews via API or bulk upload, classify sentiment with sub-25ms latency, monitor customer happiness trends, inspect token attribution, and proactively detect churn risks.

---

## 📖 Project Overview

### What is Sentiment Pulse?
Organizations receive thousands of unstructured customer reviews, support transcripts, survey comments, and social feedback items every week across fragmented channels. Traditional text processing methods are either too slow, prohibitively expensive (calling external LLM APIs per review), or compromise customer data privacy.

### The Problem It Solves
1. **Low-Latency, Cost-Free Inference**: Replaces costly remote API calls with a local, INT8-quantized DistilBERT model executing in <25ms on commodity CPUs with zero GPU requirements.
2. **Unified Actionable Intelligence**: Merges real-time single analysis, sequential bulk evaluation, paginated feedback exploration, and executive BI trend visualization into a single cohesive SaaS workspace.
3. **Data Privacy & Local Control**: Provides client-side IndexedDB analysis history that never leaves the user's browser, alongside enterprise SQL persistence for team-shared records.
4. **Reliability & Resilience**: Functions as an installable Progressive Web App (PWA) with an offline application shell, allowing users to browse historic data and review metrics even when disconnected from the network.

---

## 🚀 Complete Feature List

### Core Inference & Machine Learning
* **⚡ Sub-25ms ONNX Inference**: INT8 dynamically quantized transformer (`sentiment_model.onnx`, 67.4MB) and Hugging Face tokenizer (`tokenizer.json`) running on CPU with minimal memory footprint (<90MB RAM) and zero PyTorch runtime dependencies.
* **🔍 Client-Side Explainability (Feature Attribution)**: Real-time word-level perturbation attribution highlighting positive and negative tokens that drove the model's sentiment prediction, computed via `/predict/batch` token masking.

### Analysis & Playgrounds
* **📝 Single Review Analysis**: Real-time sentiment classification with probability scores (Positive, Neutral, Negative), confidence metrics, character counter, keyboard shortcut execution (`Ctrl+Enter` / `Cmd+Enter`), and direct persistence to database.
* **📋 Bulk Paste Analysis**: Multiline textarea supporting up to 100 rows per run (up to 2,000 characters per row). Processes reviews in sequential chunks (<=50 items) with a 500ms safety cooldown, real-time progress bar, isolated row-level error handling, and one-click CSV export.
* **🔗 Web Share & Clipboard Fallback**: Share individual prediction summaries using the native Web Share API on supported devices, with seamless automatic fallback to clipboard copy and accessible feedback toasts.

### Data Management & History
* **💾 Local IndexedDB Analysis History**: Private, client-side history stored locally in browser IndexedDB (`sentiment-pulse-db`). Supports live text search, sentiment filtering, sorting by date and confidence, record deletion, bulk clearing with confirmation modals, and zero-predict rehydration.
* **🔎 Feedback Explorer**: Paginated server feedback table with search, sentiment filters, category tags, rating stars, source channels, and record detail inspection modals.
* **📦 Bounded CSV Batch Ingestion & Secure Export**: Upload CSV reviews up to 100 MB (104,857,600 bytes) and up to 1,050,000 records per batch with bounded streaming chunking (`ImportJob`), progress tracking, encoding autodetection (UTF-8, UTF-8-BOM, Latin-1, CP1252), and CSV formula injection sanitization (`=`, `+`, `-`, `@`) on export. Datasets exceeding Microsoft Excel's single-worksheet limit of 1,048,576 total rows (maximum 1,048,575 data records with a header row) are fully supported up to Sentiment Pulse's 1,050,000 data record capacity.

### Analytics & Business Intelligence
* **📊 Interactive Executive Analytics Dashboard**: Real-time visual metrics powered by dynamically loaded Chart.js:
  - Daily Sentiment Volume Trends (temporal line chart)
  - Sentiment Distribution (doughnut chart)
  - Category Sentiment Polarity (stacked horizontal bar chart)
  - Feedback Volumes by Source Channel (bar chart)
  - Model Confidence Distribution (binned bar chart: 0–50%, 50–70%, 70–90%, 90–100%)
* **🚨 Proactive Churn Detection**: Highlights recent negative customer reviews on the executive overview card with direct jump links to investigation.

### Frontend Architecture, Accessibility & Performance
* **📱 Progressive Web App (PWA)**: Installable to desktop and mobile home screens via `manifest.webmanifest` and native Service Worker (`sw.js`, cache version `sentiment-pulse-shell-v2`).
* **🌐 Offline Application Shell**: Full offline degradation with offline banner notification, preserved navigation, and instant access to IndexedDB history, while honestly communicating when server-side ML inference is unavailable.
* **♿ Built-in Web Accessibility (WCAG 2.1 AA)**: Skip-to-content bypass link, semantic `<main>` landmark, active navigation `aria-current`, visible keyboard focus rings, modal focus trapping with Escape key dismissal and focus restoration, polite ARIA live toast notifications, table keyboard navigation, and `prefers-reduced-motion` compliance.
* **⚡ High-Performance Zero-Build Runtime**: Native ES6 browser modules with zero build step requirements for local runtime, lazy loading of heavy libraries (Chart.js dynamically loaded only when navigating to `#analytics`), optimized font preloading, and pre-compiled static Tailwind CSS.

---

## 🛠️ Architecture & Tech Stack

```text
├── main.py                         # ASGI entry point, lifespan, CORS, middleware assembly
├── database.py                     # Backward-compatible database facade
├── model.py                        # Core ONNX SentimentAnalyzer singleton engine
├── core/
│   ├── config.py                   # Centralized Pydantic Settings & environment validation
│   ├── logging.py                  # Structured logging & request context formatting
│   ├── middleware.py               # X-Request-ID & X-Process-Time middleware
│   └── security.py                 # Sliding-window IP rate limiting utilities
├── db/
│   ├── session.py                  # SQLAlchemy engine, connection pooling & get_db dependency
│   └── models.py                   # ORM models (Feedback, ImportJob)
├── alembic/                        # Alembic migration environment
│   ├── env.py                      # Dynamic database URL & batch migration configuration
│   └── versions/                   # Versioned migration scripts
├── schemas/                        # Pydantic v2 validation contracts
│   ├── prediction.py               # Inference request & response schemas
│   ├── feedback.py                 # Feedback CRUD, filtering & stats schemas
│   ├── import_job.py               # Import job tracking schemas
│   ├── analytics.py                # Business intelligence & trend schemas
│   └── health.py                   # Health, liveness & readiness schemas
├── services/
│   ├── sentiment_service.py        # Inference lifecycle & latency benchmarking
│   ├── feedback_service.py         # Feedback CRUD, pagination & persistence
│   ├── analytics_service.py        # SQL aggregations & negative feedback tracking
│   └── import_service.py           # Chunked CSV ingestion & formula-safe exports
├── api/
│   └── routes/
│       ├── predictions.py          # /predict, /predict/batch
│       ├── feedback.py             # /feedback, /feedback/{id}, /feedback/stats, /feedback/export
│       ├── imports.py              # /feedback/import, /feedback/import/{job_id}
│       ├── analytics.py            # /feedback/analytics
│       └── health.py               # /health, /health/live, /health/ready
├── templates/                      # Frontend HTML templates
│   └── index.html                  # Accessible SaaS dashboard shell & semantic views
├── static/                         # Static assets & modular browser ES6 JavaScript
│   ├── css/
│   │   ├── app.css                 # Custom design tokens, card styles, badges & keyframe animations
│   │   ├── tailwind-input.css      # Tailwind base/components/utilities directive entry point
│   │   └── tailwind.min.css        # Precompiled, minified static production CSS (37.5 KB)
│   ├── js/
│   │   ├── api.js                  # Centralized REST API client & error handler
│   │   ├── app.js                  # Shell routing, navigation, modals, toasts & shortcuts
│   │   ├── overview.js             # Overview KPI metrics & recent alert card controller
│   │   ├── analyze.js              # Single & bulk sentiment prediction controller
│   │   ├── explain.js              # Client-side word perturbation explainability controller
│   │   ├── explorer.js             # Feedback table, pagination, search & detail modal controller
│   │   ├── analytics.js            # Chart.js visualizations & confidence distribution controller
│   │   ├── imports.js              # Drag-and-drop CSV batch import & export controller
│   │   ├── history.js              # Local IndexedDB analysis history view controller
│   │   ├── sw.js                   # Service Worker shell cache & offline handler
│   │   └── lib/
│   │       ├── chart.bundle.js     # Lazy-loaded Chart.js bundle (loaded only on #analytics)
│   │       ├── idb.js              # Lightweight IndexedDB Promise wrapper
│   │       └── store.js            # History storage facade & query engine
│   ├── icons/                      # PWA icons & manifest assets
│   ├── manifest.webmanifest        # Progressive Web App manifest
│   └── theme.js                    # Zero-FOUC light/dark theme persistence controller
├── tests/
│   ├── frontend/                   # Frontend unit tests (Node.js test runner)
│   └── ...                         # Backend pytest suites (API, models, DB, CSV)
├── reports/                        # Lighthouse audit reports & performance benchmarks
├── tailwind.config.js              # Tailwind CSS build configuration & dynamic class safelist
├── package.json                    # Frontend linting, formatting & test runner configuration
├── requirements.txt                # Production backend dependencies
└── render.yaml                     # Infrastructure-as-Code deployment blueprint
```

### Architectural Principles
1. **Zero-Build Browser Runtime**: Client-side code runs as native ES6 modules (`import`/`export`). No Webpack, Vite, or Babel build step is required to serve the frontend, eliminating build-pipeline complexity.
2. **Static Tailwind CSS**: The browser loads pre-compiled, static CSS (`static/css/tailwind.min.css`). The dynamic CDN compiler (`cdn.tailwindcss.com`) is **not** used in production, saving ~350KB of JavaScript execution and eliminating layout shift.
3. **Hash Routing**: Clean client-side routing via `#overview`, `#analyze`, `#explorer`, `#analytics`, `#import-export`, and `#history` without requiring complex server-side route rewrites.
4. **Data Privacy Isolation**: Local analysis history is strictly isolated to browser IndexedDB storage and is never transmitted to the backend server. Server-persisted feedback records require an explicit "Save to Database" action or submission via `/feedback`.
5. **Graceful Offline Degradation**: When disconnected, the Service Worker serves all shell assets from cache (`sentiment-pulse-shell-v2`). The user interface clearly signals the offline state via an accessible top banner and sidebar badge, while preserving full access to IndexedDB history.

---

## 💻 Local Setup & Installation

### 1. Prerequisites
- **Python**: 3.10+
- **Node.js**: 18.0+ (required for frontend testing and Tailwind CSS compilation)
- **Database**: SQLite (default, zero-configuration) or PostgreSQL 14+

### 2. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/ranitadhikari2023/sentiment-pulse-app.git
cd sentiment-pulse-app

# Create Python virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate
```

### 3. Install Backend & Frontend Dependencies
```bash
# Install Python backend dependencies
pip install -r requirements.txt

# Install Node.js tooling (ESLint, Prettier)
npm install
```

### 4. Configure Environment
```bash
# Copy example environment configuration
cp .env.example .env
```
*(By default, `DATABASE_URL` is configured to use local SQLite: `sqlite:///./sentiment.db`)*

### 5. Run Database Migrations
```bash
# Execute Alembic schema migrations
alembic upgrade head
```

### 6. Start the Development Server
```bash
uvicorn main:app --reload --port 8000
```
- Open **`http://127.0.0.1:8000`** in your browser.
- Interactive Swagger API docs: **`http://127.0.0.1:8000/docs`**.
- OpenAPI JSON specification: **`http://127.0.0.1:8000/openapi.json`**.

---

## 🎨 Tailwind CSS Regeneration

The application relies on static, minified CSS rather than the runtime Tailwind CDN.

### Configuration & Source Files
- **Tailwind Config**: [`tailwind.config.js`](./tailwind.config.js) (scans `./templates/**/*.html` and `./static/**/*.js`, with a dynamic utility safelist).
- **CSS Input**: [`static/css/tailwind-input.css`](./static/css/tailwind-input.css) (contains `@tailwind base;`, `@tailwind components;`, `@tailwind utilities;`).
- **CSS Output**: [`static/css/tailwind.min.css`](./static/css/tailwind.min.css) (precompiled, minified static stylesheet, ~37.5 KB).

### How to Regenerate Tailwind CSS
Whenever you modify template markup or add new Tailwind classes in JavaScript, regenerate the static CSS file:

```bash
# Regenerate static Tailwind CSS using npx
npx tailwindcss -i static/css/tailwind-input.css -o static/css/tailwind.min.css --minify
```

> **Important**: The browser does **NOT** load `https://cdn.tailwindcss.com`. All utility classes are resolved at compile time into `tailwind.min.css`, ensuring sub-second first contentful paint and zero CDN runtime overhead.

---

## 🧪 Testing & Code Quality

The repository includes a comprehensive, multi-layer automated test suite spanning backend inference, API contracts, database operations, frontend logic, linting, and formatting.

### 1. Frontend Unit Tests (Node.js Test Runner)
```bash
npm test
```
Runs the zero-dependency test suite via Node.js's built-in test runner (`tests/frontend/**/*.test.js`). Verifies pure helper algorithms, batch chunking, pagination math, explainability tokenization, IndexedDB wrappers, and offline mock degradation.

### 2. Frontend Code Quality & Formatting
```bash
# Check JavaScript code style and syntax rules
npm run lint

# Check code formatting with Prettier
npm run format:check

# Auto-format JavaScript source files
npm run format
```

### 3. Backend Unit & Integration Tests (pytest)
```bash
# Run backend test suite (Python virtual environment activated)
pytest -v
```
Verifies FastAPI routing, ONNX transformer inference accuracy, database transactions, batch prediction chunking, CSV formula injection protection, sliding-window rate limiting, and health probes.

### 4. Handwritten Diagnostic Checks
```bash
python check_handwritten.py
```
Executes targeted linguistic stress tests and sanity phrases through the ONNX model to verify edge-case sentiment polarity.

---

## 📸 Visual Tour & Application Views

The Sentiment Pulse web interface is built with responsive desktop and mobile viewports in mind:

| View | Route | Key Capabilities |
| :--- | :--- | :--- |
| **Executive Overview** | `/#overview` | KPI metric cards (Total Feedback, Avg Sentiment, Positive/Negative % ratios), 7-day trend chart, and proactive recent negative feedback alert list. |
| **Analyze Playground** | `/#analyze` | Single review inference with confidence score, word-level explainability attribution, and sequential bulk paste mode (up to 100 items) with CSV export. |
| **Feedback Explorer** | `/#explorer` | Paginated review table with full-text search, multi-field filtering (sentiment, rating, category, source channel), and accessible detail inspection modal. |
| **BI Analytics** | `/#analytics` | Lazy-loaded Chart.js visual dashboard showing daily sentiment volume, overall distribution, category polarity, source breakdown, and model confidence buckets. |
| **Import & Export** | `/#import-export` | Drag-and-drop CSV batch upload with background chunking and secure, formula-sanitized CSV export. |
| **Analysis History** | `/#history` | Private, browser-local IndexedDB record viewer with real-time text search, sentiment filters, date/confidence sorting, and rehydration without re-inference. |

*(Note: Live browser verification was executed across desktop `1280x800` and mobile `390x844` viewports. Complete interactive test recordings are stored in the development artifacts repository).*

---

## 📚 API Reference

### System & Health Probes
| Method | Endpoint | Description | Auth |
| :--- | :--- | :--- | :---: |
| `GET` | `/health` | Service uptime, model metadata & inference benchmark | None |
| `GET` | `/health/live` | Liveness probe for container orchestrators | None |
| `GET` | `/health/ready` | Readiness probe verifying DB & ONNX availability | None |

### Sentiment Predictions
| Method | Endpoint | Description | Auth |
| :--- | :--- | :--- | :---: |
| `POST` | `/predict` | Real-time sentiment prediction for a single string | None |
| `POST` | `/predict/batch` | Batch prediction for up to 50 text items per request | None |

### Customer Feedback Management
| Method | Endpoint | Description | Auth |
| :--- | :--- | :--- | :---: |
| `POST` | `/feedback` | Submit feedback, analyze sentiment & persist record | None |
| `GET` | `/feedback` | List paginated feedback with filtering & search | None |
| `GET` | `/feedback/{id}` | Retrieve single feedback record | None |
| `DELETE` | `/feedback/{id}` | Delete feedback record | None |
| `GET` | `/feedback/stats` | Aggregate summary statistics | None |
| `GET` | `/feedback/export` | Download CSV export with formula injection protection | None |

### Batch Import & Business Intelligence
| Method | Endpoint | Description | Auth |
| :--- | :--- | :--- | :---: |
| `POST` | `/feedback/import` | Upload CSV feedback file up to 100 MB / 1,050,000 rows (bounded streaming chunk processing) | None |
| `GET` | `/feedback/import/{id}`| Poll status and progress of import job | None |
| `GET` | `/feedback/analytics` | Retrieve sentiment distributions, daily trends & churn alerts | None |

---

## 🔒 Security Architecture

1. **CSV Formula Injection Sanitization**: Exported CSV fields beginning with `=`, `+`, `-`, `@`, `\t`, or `\r` are automatically prepended with a single quote (`'`), neutralizing spreadsheet macro execution vulnerabilities in Microsoft Excel and Google Sheets.
2. **Sliding-Window Rate Limiting**: In-memory IP rate limiter defends backend inference endpoints from denial-of-service spikes while supporting burst traffic.
3. **Structured Observability**: Every HTTP request receives a unique `X-Request-ID` correlation header and execution duration tracking via `X-Process-Time`.
4. **CORS Control**: Configurable via `ALLOWED_ORIGINS` in `.env` to restrict browser cross-origin requests in production.
5. **Separated Liveness & Readiness Probes**: Independent `/health/live` and `/health/ready` endpoints ensure container orchestrators (e.g., Kubernetes, Render) route traffic only after model weights and database connections are primed.

---

## 🌐 Offline & PWA Architecture

Sentiment Pulse delivers an offline application shell designed to uphold user trust and prevent false data representation:

- **Service Worker Lifecycle**: [`static/js/sw.js`](./static/js/sw.js) registers on startup and pre-caches the 26 critical shell assets under the versioned cache `sentiment-pulse-shell-v2`.
- **Honest Offline Handling**: Unlike naive PWAs that return mock or cached predictions, Sentiment Pulse honestly blocks `/predict` and `/feedback` submissions when offline, displaying informative toast messages and disabled action states.
- **Local Data Availability**: The IndexedDB analysis history store remains fully interactive offline, enabling users to search, filter, sort, and inspect past predictions without network connectivity.
- **Seamless Online Recovery**: Connection status is monitored via window `online`/`offline` listeners. When connectivity returns, the offline warning banner automatically dismisses, and server-dependent features re-engage immediately.

---

## # Backend follow-ups

The following three architectural follow-ups are documented for future backend iterations:

1. **Confidence buckets on `/feedback/analytics`**:
   A `confidence_buckets` field on the analytics endpoint would eliminate the client's bounded multi-page confidence aggregation and provide more efficient server-side aggregation. Currently, the client fetches up to 5 pages of feedback records (`GET /feedback?page=X&page_size=100`) to compute distribution metrics. Implementing this aggregation directly in SQL would reduce network overhead to a single sub-10ms query.

2. **Server-side explainability endpoint**:
   The current explainability workflow depends on client-side orchestration and network calls (`/predict/batch` token masking). A dedicated backend explainability endpoint could centralize, parallelize, and optimize the attribution operation directly within the ONNX execution session, eliminating multiple HTTP roundtrips.

3. **Optional history synchronization**:
   Phase 2 analysis history is intentionally stored in browser-local IndexedDB to preserve privacy and minimize database bloat. An optional future synchronization feature could allow authenticated users to selectively sync their local history to the cloud while maintaining strict tenant isolation, user authorization, and data privacy controls.

---

## ☁️ Render Production Deployment

The project includes an Infrastructure-as-Code blueprint in [`render.yaml`](./render.yaml).

1. Connect your repository to [Render](https://render.com).
2. Render provisions the **Web Service** and managed **PostgreSQL Database** (`sentiment-pulse-db`).
3. Build command:
   ```bash
   pip install -r requirements.txt && alembic upgrade head
   ```
4. Start command:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1 --proxy-headers --forwarded-allow-ips '*'
   ```
5. Health Check Path: `/health/ready` (ensures traffic is only routed after DB connection and model warmup).

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
