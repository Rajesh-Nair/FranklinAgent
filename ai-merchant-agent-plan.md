# AI Merchant Classification Agent — Implementation Plan

## Overview

A modular, cloud-deployable AI agent system that accepts transaction statement text, identifies the merchant/brand via web search, and classifies it against a predefined category taxonomy. Exposed as a FastAPI REST API, containerised for Google Cloud Run, with a Streamlit frontend for demo purposes.

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Database Schema](#2-database-schema)
3. [Agent Flow](#3-agent-flow)
4. [Project Structure](#4-project-structure)
5. [Component Specifications](#5-component-specifications)
6. [API Specification](#6-api-specification)
7. [Batch Mode](#7-batch-mode)
8. [Token Cost Estimation](#8-token-cost-estimation)
9. [Frontend](#9-frontend)
10. [Configuration](#10-configuration)
11. [Infrastructure & Deployment](#11-infrastructure--deployment)
12. [Implementation Phases](#12-implementation-phases)

---

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Streamlit UI                         │
│          (Single text / Batch CSV / Cost estimate)          │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP
┌────────────────────────▼────────────────────────────────────┐
│                      FastAPI Layer                          │
│         /classify  /batch  /cost-estimate  /health          │
│              (Auth middleware hook — future)                 │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   Orchestrator Agent                        │
│  Cache → Vector Search → Web Search → Extract → Validate   │
└──┬──────────────┬──────────────┬──────────────┬────────────┘
   │              │              │              │
┌──▼───┐    ┌────▼────┐   ┌─────▼─────┐  ┌────▼────────┐
│Cache │    │ Vector  │   │  LLM      │  │  LLM        │
│Check │    │ Search  │   │  Pass 1   │  │  Pass 2     │
│(DB)  │    │ (FAISS) │   │ Extraction│  │ Validation  │
└──────┘    └─────────┘   └─────┬─────┘  └─────────────┘
                                │
                         ┌──────▼──────┐
                         │ Web Search  │
                         │ Google ADK  │
                         │ / SerpAPI   │
                         └─────────────┘
┌─────────────────────────────────────────────────────────────┐
│                     Data Layer                              │
│   SQLite (→ PostgreSQL)    FAISS index (→ pluggable)        │
│   brands / enquiries / categories tables                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Database Schema

### 2.1 Relational Database (SQLite → PostgreSQL)

```sql
-- Categories (pre-seeded, 100+ entries)
CREATE TABLE categories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    level_1     TEXT NOT NULL,           -- e.g. "Shopping"
    level_2     TEXT NOT NULL UNIQUE     -- e.g. "Fashion & Apparel"
);
INSERT INTO categories (level_1, level_2) VALUES ('Uncategorized', 'Uncategorized');

-- Brands / Merchants
CREATE TABLE brands (
    id               TEXT PRIMARY KEY,   -- UUID
    name             TEXT NOT NULL,      -- Normalised title case, e.g. "Amazon Marketplace"
    brand_group      TEXT,               -- Parent brand, e.g. "Amazon" (nullable)
    category_id      INTEGER NOT NULL REFERENCES categories(id),
    aliases          TEXT NOT NULL DEFAULT '[]',   -- JSON array of known raw strings
    website_url      TEXT,
    confidence_score REAL,               -- 0.0–1.0, confidence when entry was created
    status           TEXT NOT NULL DEFAULT 'pending_review',  -- pending_review | verified
    created_at       TEXT NOT NULL,      -- ISO 8601
    updated_at       TEXT NOT NULL
);

-- Enquiries (audit log + cache)
CREATE TABLE enquiries (
    id               TEXT PRIMARY KEY,   -- UUID
    input_text       TEXT NOT NULL,
    brand_id         TEXT REFERENCES brands(id),   -- NULL if unresolved
    confidence_score REAL,
    source           TEXT NOT NULL,      -- 'api' | 'batch'
    status           TEXT NOT NULL,      -- 'success' | 'unresolved' | 'error'
    error_detail     TEXT,               -- NULL unless status=error
    created_at       TEXT NOT NULL
);
CREATE UNIQUE INDEX idx_enquiries_input_text ON enquiries(input_text);  -- cache lookup
```

### 2.2 Vector Index (FAISS)

- **Index on:** `brands.name` + each entry in `brands.aliases[]`
- **Stored alongside index:** mapping of vector ID → `brand.id`
- **Embedding model:** `text-embedding-004` (Google) by default, configurable
- **Storage:** `./data/faiss.index` + `./data/faiss_map.json`
- **Interface:** `VectorStore` abstract class — swap to Pinecone, Vertex AI Vector Search, or Weaviate via config

---

## 3. Agent Flow

```
INPUT: raw text string (e.g. "AMZN*MARKETPLACE GB 90")

Step 1 — Cache Check
  └─ Query: SELECT * FROM enquiries WHERE input_text = :text
  └─ HIT  → return stored result immediately (source labelled 'cache')
  └─ MISS → continue

Step 2 — Vector Search (Table 1)
  └─ Embed input text
  └─ Search FAISS index (top 3 candidates with scores)
  └─ If top score ≥ VECTOR_CONFIDENCE_THRESHOLD (default 0.88) → match found
       └─ Return brand from Table 1 → skip web search
  └─ Else → continue

Step 3 — Web Search
  └─ Query construction (configurable):
       Option A: raw input text + system hint "this text is from a bank transaction statement"
       Option B: LLM generates optimised search query from raw text
  └─ Provider: Google ADK built-in (default) | SerpAPI (fallback / non-Google)
  └─ Fetch top 3 results (configurable: WEB_SEARCH_TOP_K)
  └─ Truncate each result snippet to 500 characters

Step 4 — LLM Pass 1: Extraction
  └─ Prompt includes:
       - Raw input text
       - Web search results (truncated snippets)
       - Top 15 candidate categories from vector search on category names (configurable)
         └─ Expand to top 30 if initial category confidence < 0.6, hard cap 50
       - Top 3 brand candidates from vector search (with scores)
       - Instruction: return JSON { brand_name, brand_group, category_id, website_url,
                                    confidence_score, is_new_brand, reasoning }
  └─ Model: EXTRACTION_MODEL (default: gemini-2.0-flash-latest)

Step 5 — Rule-Based Validation (cheap, always runs)
  └─ Check 1: category_id exists in categories table
  └─ Check 2: extracted brand name appears in at least one search result snippet or URL
  └─ Check 3: if is_new_brand=false, vector similarity score is above match threshold
  └─ Any failure → flag for Pass 2 re-evaluation

Step 6 — LLM Pass 2: Validation (always runs — accuracy priority)
  └─ Prompt includes:
       - Original input text
       - Web search results
       - Pass 1 answer (brand, category, reasoning)
       - Instruction: critically evaluate Pass 1 answer, return
                      { agree: bool, corrected_answer?: {...}, reasoning }
  └─ Model: VALIDATION_MODEL (default: gemini-2.0-flash-latest)
  └─ If agree=false → result is UNRESOLVED

Step 7 — Write Result
  └─ RESOLVED:
       └─ If new brand: INSERT into brands (status=pending_review), update FAISS index
       └─ INSERT into enquiries (status=success)
       └─ Return { brand_name, brand_group, category_level_1, category_level_2,
                   confidence_score, is_new_brand, brand_id }
  └─ UNRESOLVED (disagreement or rule failure):
       └─ INSERT into enquiries (status=unresolved, brand_id=NULL)
       └─ Return { brand_name: null, category: "Uncategorized",
                   confidence_score: 0, status: "unresolved",
                   audit_id: <enquiry.id> }
```

---

## 4. Project Structure

```
merchant-agent/
├── app/
│   ├── main.py                    # FastAPI app entry point
│   ├── api/
│   │   ├── routes/
│   │   │   ├── classify.py        # POST /classify
│   │   │   ├── batch.py           # POST /batch
│   │   │   └── cost.py            # POST /cost-estimate
│   │   └── middleware/
│   │       └── auth.py            # Hook point — no-op at launch
│   ├── agent/
│   │   ├── orchestrator.py        # Main agent flow (Steps 1–7)
│   │   ├── cache.py               # Table 2 exact-match cache
│   │   ├── extractor.py           # LLM Pass 1
│   │   ├── validator.py           # LLM Pass 2 + rule-based checks
│   │   └── query_builder.py       # Web search query construction
│   ├── providers/
│   │   ├── base.py                # Abstract LLMProvider, SearchProvider
│   │   ├── google_adk.py          # Google ADK implementation
│   │   ├── openai_provider.py     # OpenAI implementation (future)
│   │   └── serpapi.py             # SerpAPI search implementation
│   ├── vector/
│   │   ├── base.py                # Abstract VectorStore
│   │   ├── faiss_store.py         # FAISS implementation
│   │   └── pinecone_store.py      # Pinecone stub (future)
│   ├── db/
│   │   ├── base.py                # Abstract Repository
│   │   ├── sqlite_repo.py         # SQLite implementation
│   │   ├── postgres_repo.py       # PostgreSQL stub (future)
│   │   └── migrations/
│   │       └── 001_initial.sql
│   ├── cost/
│   │   └── estimator.py           # Token cost estimation
│   └── config.py                  # Centralised config via env vars / YAML
├── frontend/
│   └── app.py                     # Streamlit UI
├── data/
│   ├── faiss.index                # FAISS vector index (gitignored)
│   ├── faiss_map.json             # Vector ID → brand ID mapping
│   ├── merchant.db                # SQLite DB (gitignored)
│   └── categories.csv             # Seed data for categories table
├── config/
│   ├── default.yaml               # Default configuration
│   └── pricing.yaml               # Model pricing per 1K tokens
├── Dockerfile
├── docker-compose.yml             # Local dev: API + frontend
├── requirements.txt
├── .env.example
└── README.md
```

---

## 5. Component Specifications

### 5.1 LLM Provider Abstraction

```python
# app/providers/base.py
class LLMProvider(ABC):
    @abstractmethod
    async def complete(self, prompt: str, system: str) -> LLMResponse: ...

class SearchProvider(ABC):
    @abstractmethod
    async def search(self, query: str, top_k: int) -> list[SearchResult]: ...

class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed(self, text: str) -> list[float]: ...
```

Provider resolved at startup from `LLM_PROVIDER` config value: `google_adk` | `openai`.

### 5.2 Vector Store Abstraction

```python
# app/vector/base.py
class VectorStore(ABC):
    @abstractmethod
    def search(self, query_vector: list[float], top_k: int) -> list[VectorMatch]: ...

    @abstractmethod
    def upsert(self, id: str, vector: list[float], metadata: dict) -> None: ...

    @abstractmethod
    def persist(self) -> None: ...
```

FAISS index is loaded at startup, persisted to disk after every `upsert`.

### 5.3 Repository Abstraction

```python
# app/db/base.py
class BrandRepository(ABC):
    @abstractmethod
    def get_by_id(self, id: str) -> Brand | None: ...

    @abstractmethod
    def find_by_name(self, name: str) -> Brand | None: ...

    @abstractmethod
    def create(self, brand: Brand) -> Brand: ...

class EnquiryRepository(ABC):
    @abstractmethod
    def find_by_text(self, text: str) -> Enquiry | None: ...  # cache lookup

    @abstractmethod
    def create(self, enquiry: Enquiry) -> Enquiry: ...

class CategoryRepository(ABC):
    @abstractmethod
    def get_all(self) -> list[Category]: ...

    @abstractmethod
    def get_by_id(self, id: int) -> Category | None: ...

    @abstractmethod
    def search(self, query_vector: list[float], top_k: int) -> list[Category]: ...
```

### 5.4 Extraction Prompt Template

```
System: You are a merchant classification assistant. You analyse bank transaction
statement text and identify the merchant brand and shopping category.

User:
Transaction text: "{input_text}"
Note: This text is from a bank transaction statement and may contain noise
such as country codes, amounts, or location abbreviations.

Web search results:
{search_results}  ← top 3 snippets, 500 chars each

Known brand candidates from database (with similarity scores):
{vector_candidates}  ← top 3 matches

Available categories (shortlisted):
{categories}  ← top 15–30 as "id: level_1 > level_2"

Return JSON only:
{
  "brand_name": string | null,
  "brand_group": string | null,
  "category_id": integer | null,
  "website_url": string | null,
  "confidence_score": float (0.0–1.0),
  "is_new_brand": boolean,
  "reasoning": string
}
```

### 5.5 Validation Prompt Template

```
System: You are a critical reviewer. Evaluate the following merchant classification
and determine if it is correct based solely on the evidence provided.

User:
Transaction text: "{input_text}"

Web search results:
{search_results}

Proposed classification:
- Brand: {brand_name}
- Category: {category_level_1} > {category_level_2}
- Reasoning: {reasoning}

Do you agree? Reply JSON only:
{
  "agree": boolean,
  "corrected_answer": { ...same schema as extraction... } | null,
  "reasoning": string
}
```

---

## 6. API Specification

### POST /classify
```json
// Request
{ "text": "AMZN*MARKETPLACE GB 90" }

// Response 200
{
  "brand_id": "uuid",
  "brand_name": "Amazon Marketplace",
  "brand_group": "Amazon",
  "category_level_1": "Shopping",
  "category_level_2": "Online Marketplace",
  "confidence_score": 0.94,
  "is_new_brand": false,
  "status": "success",
  "audit_id": "uuid",
  "cached": false
}

// Response 200 (unresolved)
{
  "brand_id": null,
  "brand_name": null,
  "brand_group": null,
  "category_level_1": "Uncategorized",
  "category_level_2": "Uncategorized",
  "confidence_score": 0.0,
  "status": "unresolved",
  "audit_id": "uuid",
  "cached": false
}
```

### POST /batch
```json
// Request
{
  "texts": ["AMZN*MARKETPLACE GB 90", "STARBUCKS #1234 LONDON"],
  "batch_size": 5   // optional, overrides config default
}

// Response 200
{
  "results": [
    { "text": "AMZN*MARKETPLACE GB 90", "brand_name": "Amazon Marketplace", ... },
    { "text": "STARBUCKS #1234 LONDON", "brand_name": "Starbucks", ... }
  ],
  "summary": {
    "total": 2,
    "resolved": 2,
    "unresolved": 0,
    "errors": 0
  },
  "token_actuals": {
    "input_tokens": 1420,
    "output_tokens": 310,
    "estimated_cost_usd": 0.0021
  }
}
```

### POST /cost-estimate
```json
// Request
{
  "num_texts": 100,
  "batch_size": 5,
  "enable_web_search": true,
  "enable_validation_pass": true,
  "extraction_model": "gemini-2.0-flash-latest",
  "validation_model": "gemini-2.0-flash-latest"
}

// Response 200
{
  "estimated_input_tokens": 142000,
  "estimated_output_tokens": 31000,
  "estimated_cost_usd": 0.21,
  "breakdown": {
    "extraction_pass": { "input_tokens": 95000, "output_tokens": 20000, "cost_usd": 0.14 },
    "validation_pass": { "input_tokens": 47000, "output_tokens": 11000, "cost_usd": 0.07 }
  },
  "num_llm_calls": 20,
  "assumptions": "Estimates based on avg 500 chars/search result, 15 categories in shortlist"
}
```

### GET /health
```json
{ "status": "ok", "db": "ok", "vector_store": "ok" }
```

---

## 7. Batch Mode

### Processing Strategy
- Input: `texts[]` array (from API) or CSV file upload (from frontend)
- Texts are grouped into batches of `batch_size` (default: 1, configurable, max: 10)
- Each batch is sent as a **single LLM prompt** listing all N texts
- LLM returns a JSON array of N results

### Batch Prompt Structure
```
Transaction texts to classify (return a JSON array with one result per item,
in the same order):
1. "AMZN*MARKETPLACE GB 90"
2. "STARBUCKS #1234 LONDON"
...N. "..."

Available categories: {shortlisted_categories}
...
```

### Failure Handling
- Per-item status in response: `success | unresolved | error`
- Failed items are written to Table 2 with `status=error` + `error_detail`
- Partial results returned — batch does not fail atomically
- Failed items can be resubmitted individually or as a new batch

### CSV Upload (Frontend)
- Expected columns: `text` (required), any additional columns are passed through untouched
- Output CSV adds columns: `brand_id, brand_name, brand_group, category_level_1, category_level_2, confidence_score, status, audit_id`

---

## 8. Token Cost Estimation

```python
# app/cost/estimator.py

@dataclass
class CostEstimate:
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost_usd: float
    breakdown: dict
    num_llm_calls: int
    assumptions: str

class TokenCostEstimator:
    # Loaded from config/pricing.yaml
    PRICING: dict[str, ModelPricing]  # model_id → { input_per_1k, output_per_1k }

    def estimate(self, config: EstimateRequest) -> CostEstimate:
        """Pre-flight cost estimate before job runs."""
        batches = ceil(config.num_texts / config.batch_size)

        # Avg tokens per batch (empirically tuned, overridable in config)
        avg_search_tokens   = 500 * 3 * config.batch_size   # 3 results × 500 chars
        avg_category_tokens = 15 * 20                        # 15 categories × ~20 tokens each
        avg_input_overhead  = 200                            # system prompt + structure

        extraction_input  = batches * (avg_input_overhead + avg_search_tokens + avg_category_tokens)
        extraction_output = batches * config.batch_size * 80  # ~80 tokens per result

        validation_input  = extraction_input + extraction_output  # includes Pass 1 answer
        validation_output = extraction_output

        total_input  = extraction_input + (validation_input  if config.enable_validation_pass else 0)
        total_output = extraction_output + (validation_output if config.enable_validation_pass else 0)

        cost = self._calc_cost(config.extraction_model, extraction_input, extraction_output)
        if config.enable_validation_pass:
            cost += self._calc_cost(config.validation_model, validation_input, validation_output)

        return CostEstimate(
            estimated_input_tokens=total_input,
            estimated_output_tokens=total_output,
            estimated_cost_usd=round(cost, 4),
            breakdown={...},
            num_llm_calls=batches * (2 if config.enable_validation_pass else 1),
            assumptions="Estimates based on avg 500 chars/search result, 15 categories"
        )

    def record_actuals(self, response_metadata: dict) -> ActualCost:
        """Post-run: record real token usage from API response."""
        ...
```

### config/pricing.yaml
```yaml
models:
  gemini-2.0-flash-latest:
    input_per_1k_tokens: 0.000075
    output_per_1k_tokens: 0.0003
  gemini-1.5-pro:
    input_per_1k_tokens: 0.00125
    output_per_1k_tokens: 0.005
  gpt-4o:
    input_per_1k_tokens: 0.0025
    output_per_1k_tokens: 0.01
```

---

## 9. Frontend (Streamlit)

### Pages / Sections

**1. Single Classification**
- Text input box
- Submit button → calls `POST /classify`
- Result card: brand name, brand group, category (level 1 > level 2), confidence score badge, status pill
- Shows if result was served from cache

**2. Batch Classification**
- CSV file uploader
- Batch size slider (1–10)
- Cost estimate section:
  - "Estimate Cost" button → calls `POST /cost-estimate` → shows breakdown table
- "Run Batch" button → calls `POST /batch`
- Results table (sortable by status, confidence)
- Download results as CSV button

**3. Sidebar Config**
- API base URL (configurable)
- Model selection (extraction / validation)
- Web search toggle
- Validation pass toggle

---

## 10. Configuration

All config via environment variables (override) or `config/default.yaml`:

```yaml
# config/default.yaml

# LLM
llm_provider: google_adk             # google_adk | openai
extraction_model: gemini-2.0-flash-latest
validation_model: gemini-2.0-flash-latest
embedding_model: text-embedding-004

# Search
search_provider: google_adk          # google_adk | serpapi
search_top_k: 3
search_snippet_max_chars: 500
use_llm_query_builder: false         # true = LLM generates search query

# Vector store
vector_store: faiss                  # faiss | pinecone | vertex_ai
vector_store_path: ./data/faiss.index
vector_confidence_threshold: 0.88

# Database
db_provider: sqlite                  # sqlite | postgresql
sqlite_path: ./data/merchant.db
# postgresql_url: postgresql://...

# Agent
category_shortlist_size: 15
category_shortlist_expand_size: 30
category_shortlist_max: 50
category_confidence_threshold: 0.6

# Batch
batch_size: 1
batch_size_max: 10

# API
api_host: 0.0.0.0
api_port: 8080

# Secrets (set via env vars, not yaml)
# GOOGLE_API_KEY=...
# SERPAPI_KEY=...
# GOOGLE_APPLICATION_CREDENTIALS=...
```

---

## 11. Infrastructure & Deployment

### Dockerfile
```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY config/ ./config/
COPY data/categories.csv ./data/

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

### docker-compose.yml (local dev)
```yaml
version: "3.9"
services:
  api:
    build: .
    ports: ["8080:8080"]
    env_file: .env
    volumes:
      - ./data:/app/data   # persist FAISS + SQLite

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports: ["8501:8501"]
    environment:
      - API_URL=http://api:8080
```

### Cloud Run Deployment
```bash
# Build and push
gcloud builds submit --tag gcr.io/$PROJECT_ID/merchant-agent

# Deploy API
gcloud run deploy merchant-agent \
  --image gcr.io/$PROJECT_ID/merchant-agent \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --set-env-vars GOOGLE_API_KEY=$GOOGLE_API_KEY \
  --allow-unauthenticated   # remove when auth is added

# Deploy frontend (separate service)
gcloud run deploy merchant-agent-ui \
  --image gcr.io/$PROJECT_ID/merchant-agent-ui \
  --platform managed \
  --region us-central1 \
  --set-env-vars API_URL=https://merchant-agent-xxxxx-uc.a.run.app
```

### Auth Middleware Hook (future)
```python
# app/api/middleware/auth.py
async def auth_middleware(request: Request, call_next):
    # No-op at launch. Replace with API key / OAuth check here.
    return await call_next(request)
```

---

## 12. Implementation Phases

### Phase 1 — Core Agent (no API, no frontend)
- [ ] DB schema + SQLite repo
- [ ] Category seed data (`categories.csv` → DB)
- [ ] FAISS vector store + embedding
- [ ] Google ADK provider (LLM + search)
- [ ] Orchestrator: Steps 1–7 (cache, vector, web search, extract, validate, write)
- [ ] Unit tests for each step with mocked providers

### Phase 2 — API Layer
- [ ] FastAPI app skeleton
- [ ] `POST /classify` endpoint
- [ ] `GET /health` endpoint
- [ ] Auth middleware hook (no-op)
- [ ] Dockerfile + docker-compose

### Phase 3 — Batch Mode
- [ ] Batch orchestrator (group N texts per LLM call)
- [ ] `POST /batch` endpoint
- [ ] CSV parsing
- [ ] Per-item error handling + partial results

### Phase 4 — Cost Estimation
- [ ] `TokenCostEstimator.estimate()` (pre-flight)
- [ ] `TokenCostEstimator.record_actuals()` (post-run)
- [ ] `pricing.yaml` with initial model prices
- [ ] `POST /cost-estimate` endpoint

### Phase 5 — Frontend
- [ ] Streamlit app: single classification view
- [ ] Streamlit app: batch upload + cost estimate view
- [ ] Dockerfile.frontend
- [ ] Cloud Run deployment for frontend

### Phase 6 — Hardening
- [ ] SerpAPI provider (search fallback)
- [ ] OpenAI provider stub
- [ ] Pinecone VectorStore stub
- [ ] PostgreSQL repo stub
- [ ] Integration tests (real DB, real FAISS, mocked LLM)
- [ ] Logging + structured error responses
- [ ] README with setup instructions

---

## Key Design Decisions & Rationale

| Decision | Choice | Rationale |
|---|---|---|
| Validation pass frequency | Always | Accuracy over cost |
| Batch default N | 1 | Safe default; cost savings opt-in |
| Batch max N | 10 | Beyond 10, JSON parsing reliability degrades |
| Category shortlist | 15 → 30 → 50 | Balances token cost vs recall |
| New brand write | Immediate, status=pending_review | Enables caching; human reviews asynchronously |
| Unresolved action | null brand + Uncategorized + audit log | No hallucinated output ever reaches downstream |
| FAISS index update | On every new brand write | Index stays current; disk persist on every upsert |
| SQLite → PostgreSQL | Repository pattern | Zero application code change to swap |
| Auth | No-op middleware | One file to change when needed |

---

*Generated: 2026-03-30*
