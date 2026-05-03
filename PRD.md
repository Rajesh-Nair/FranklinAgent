# PRD: FranklinAgent — AI Merchant Classification Agent

## Background & Context

Financial transaction data produced by banks and payment processors is typically raw and machine-generated. Strings like "AMZN*MARKETPLACE GB 90" or "SQ *COFFEE SHOP LONDON" carry no structured metadata about the merchant, brand group, or spending category. Fintech teams that build budgeting tools, spending analytics, fraud detection, or customer insights pipelines must enrich this data before it becomes useful.

FranklinAgent is an internal AI agent system built by a fintech team to solve this enrichment problem at scale. It accepts raw transaction statement text, identifies the underlying merchant or brand via a combination of vector similarity search and live web search, classifies it against a predefined category taxonomy, and caches results so subsequent lookups on the same text are instant. The system is designed from day one to handle millions of requests in batch mode, with a FastAPI REST API, a Streamlit demo frontend, and a deployment target of Google Cloud Run.

The architecture is intentionally modular and provider-agnostic so that the underlying LLM, search engine, vector store, and database can each be swapped independently via configuration without changing application logic.

---

## Problem Statement

A fintech team needs to enrich raw bank transaction statement strings with structured merchant and category metadata at scale. The core challenges are:

- **Raw text is noisy and inconsistent.** Transaction strings contain country codes, amount fragments, location abbreviations, and payment-processor prefixes (e.g., "SQ *", "AMZN*") that obscure the actual merchant identity.
- **No static lookup table is sufficient.** The long tail of merchant variants is too large and constantly changing to maintain manually. New merchants appear regularly.
- **Accuracy is non-negotiable.** A classification that hallucinates a merchant name or assigns a wrong category corrupts downstream analytics. Unresolved is always preferable to wrong.
- **Scale is a hard requirement.** The system must support batch processing of millions of transactions. Individual classification latency is not a concern; batch throughput and cost efficiency are.
- **Results must be auditable.** Every classification — successful or not — must be logged with enough context to allow human review.
- **New merchant discovery must be safe.** When the system encounters a brand it has never seen, it must write a provisional record (pending human review) rather than silently discarding the result or treating it as verified.

---

## Initial Ideas & Alternatives Considered

**Two-pass LLM design (chosen):** Use a dedicated extraction LLM call (Pass 1) followed by a mandatory validation LLM call (Pass 2) that critically reviews the first answer. This prioritises accuracy over cost. Pass 2 always runs on net-new classifications (cache misses). A single Pass 2 disagreement marks the result as unresolved — there is no retry loop.

**Single-pass LLM (considered, rejected):** A single LLM call could reduce cost and latency, but the team decided accuracy over cost is the guiding principle. The dual-pass design was retained.

**Skipping Pass 2 on high-confidence results (considered, rejected):** Conditionally skipping Pass 2 when confidence is very high would save tokens. Rejected — LLM Pass 2 always runs on cache misses to maintain a consistent accuracy guarantee.

**Vector search as the only classifier (considered, rejected):** If FAISS similarity exceeds the threshold (0.88), the system can match a known brand without any LLM call. However, this only works for known brands in the database and provides no category reasoning. Web search + LLM is still needed for new merchants.

**Trusting cached results without re-validation (chosen):** Cache hits return stored values without re-running any LLM call. The assumption is that a result was validated when first classified and the stored value is trustworthy. The cache is a toggleable feature — it can be disabled if needed in the future.

**Google ADK as the primary provider (chosen):** The system uses the real Google Agent Development Kit (`google.adk.agents`, `google.adk.tools`) rather than calling the Gemini API directly. This gives access to native ADK tools including `google_search`, and keeps the agent orchestration aligned with the ADK programming model.

**SerpAPI as search fallback (planned):** SerpAPI is a configurable alternative search provider for environments where Google ADK search is unavailable or for non-Google deployments.

**FAISS as the vector store (chosen for launch):** FAISS is fast, dependency-light, and sufficient for local and single-instance Cloud Run deployments. Pinecone and Vertex AI Vector Search are documented as future swap targets via the VectorStore abstract interface.

**SQLite → PostgreSQL migration path (chosen):** SQLite is used for development and initial deployment. The Repository pattern abstracts all database access so PostgreSQL can be swapped in with zero application code changes.

---

## Solution

FranklinAgent is a modular AI agent system exposed as a FastAPI REST API, with a Streamlit frontend for demo and testing. It accepts raw bank transaction text, classifies the underlying merchant and category, caches results, and returns structured JSON.

### High-Level Architecture

```
Streamlit UI  →  FastAPI Layer  →  Orchestrator Agent  →  Data Layer
                                         │
                     ┌───────────────────┼────────────────────┐
                  Cache Check      Vector Search         Web Search
                  (DB lookup)      (FAISS)               (Google ADK)
                                         │
                               LLM Pass 1 (Extraction)
                               LLM Pass 2 (Validation)
                                         │
                                   Write Result (DB + FAISS)
```

### Agent Flow (7 Steps)

**Step 1 — Cache Check**
Query the `enquiries` table for an exact match on `input_text`. On a hit, return the stored result immediately with `cached: true`. On a miss, continue.

**Step 2 — Vector Search**
Embed the input text and search the FAISS index for the top 3 brand candidates with similarity scores. If the top score is ≥ 0.88 (configurable), treat it as a match and skip web search.

**Step 3 — Web Search**
Construct a search query (either raw text + system hint, or LLM-generated query if `use_llm_query_builder: true`). Fetch the top 3 results (configurable), truncate each snippet to 500 characters.

**Step 4 — LLM Pass 1: Extraction**
Send to the extraction model: input text, web search snippets, top 15–30 category candidates from vector search, top 3 brand candidates. Return structured JSON: `{ brand_name, brand_group, category_id, website_url, confidence_score, is_new_brand, reasoning }`.

**Step 5 — Rule-Based Validation**
Three cheap checks: (1) `category_id` exists in the categories table; (2) extracted brand name appears in at least one search result or URL; (3) if `is_new_brand=false`, vector similarity score is above match threshold. Any failure flags for Pass 2 re-evaluation.

**Step 6 — LLM Pass 2: Validation**
Always runs for cache misses. Send the original input, web results, and Pass 1 answer to the validation model. Return `{ agree: bool, corrected_answer?, reasoning }`. If `agree=false`, the result is immediately marked `unresolved`. No retry loop — disagreement is terminal.

**Step 7 — Write Result**
- **Resolved:** If new brand, insert into `brands` with `status=pending_review` and update FAISS index. Insert into `enquiries` with `status=success`. Return full classification.
- **Unresolved:** Insert into `enquiries` with `status=unresolved`, `brand_id=NULL`. Return null brand, `Uncategorized` category, `confidence_score=0`.

### Pre/Post LLM Call Hooks

Stub hooks are implemented at the call boundary of every LLM and search provider call. These are no-ops at launch but provide integration points for future data desensitisation workflows (e.g., stripping PII from transaction text before it leaves the infrastructure, or post-processing LLM responses). The hooks are defined as overridable methods on the provider base classes.

### API Endpoints

- `POST /classify` — single text classification
- `POST /batch` — batch of texts (grouped into LLM prompt batches of configurable size, max 10)
- `POST /cost-estimate` — pre-flight token cost estimate
- `GET /health` — health check for DB and vector store

### Batch Mode

Input texts are grouped into batches of N (default 1, max 10, configurable). Each batch is sent as a single LLM prompt listing all N texts. The LLM returns a JSON array of N results. Per-item status is tracked (`success | unresolved | error`). Partial results are returned — a batch does not fail atomically. The frontend supports CSV upload and CSV download of results.

### Category Taxonomy

A pre-supplied CSV of 100+ categories (two levels: `level_1`, `level_2`) is seeded into the database at startup. The taxonomy is treated as slow-changing stable data. Future extensions are possible via database update — no application code change required.

### New Brand Workflow

When a net-new brand is classified with sufficient confidence, it is immediately written to the `brands` table with `status=pending_review`. This enables caching of the result on the next lookup. A human reviews pending brands asynchronously. At scale (millions of transactions), bulk approval workflows are anticipated as a future operational need.

### Cost Estimation

A `TokenCostEstimator` class provides pre-flight estimates before a batch runs and records actual token usage from API responses post-run. Model pricing is loaded from `config/pricing.yaml`.

---

## User Stories

1. As a fintech data engineer, I want to submit a raw bank transaction string to the API and receive a structured merchant name, brand group, and category, so that I can enrich transaction records without manual lookup.
2. As a fintech data engineer, I want identical transaction strings to return cached results instantly, so that repeated lookups do not incur redundant LLM costs.
3. As a fintech data engineer, I want to toggle the cache off via configuration, so that I can force re-classification when needed.
4. As a fintech data engineer, I want to upload a CSV of transaction strings and receive an enriched CSV back, so that I can process bulk transaction exports without writing custom code.
5. As a fintech data engineer, I want to configure the batch size for LLM calls, so that I can tune throughput and cost trade-offs.
6. As a fintech data engineer, I want a pre-flight cost estimate before running a large batch, so that I can approve token spend before committing to the job.
7. As a fintech data engineer, I want actual token usage recorded after each batch, so that I can reconcile estimated vs. real LLM costs.
8. As a fintech data engineer, I want unresolved classifications returned with a null brand and an `unresolved` status rather than a hallucinated guess, so that my downstream pipelines can handle unknowns safely.
9. As a fintech data engineer, I want every classification — successful or unresolved — to be written to an audit log with a unique `audit_id`, so that I can trace any result back to its origin.
10. As a fintech data engineer, I want to know whether a result came from the cache or was freshly classified, so that I can assess data freshness.
11. As a fintech data engineer, I want each item in a batch to have its own status (`success | unresolved | error`), so that partial failures do not block the rest of the batch.
12. As a fintech data engineer, I want failed batch items to be retryable individually or as a new batch, so that transient errors do not cause permanent data loss.
13. As a data analyst, I want classification results to include a confidence score between 0 and 1, so that I can filter high-confidence results from uncertain ones.
14. As a data analyst, I want two-level category labels (`category_level_1`, `category_level_2`) on every result, so that I can build both coarse and fine-grained spending breakdowns.
15. As a data analyst, I want to use the Streamlit UI to test single transaction strings interactively, so that I can validate classification quality without writing API calls.
16. As a data analyst, I want the Streamlit UI to show whether a result was served from cache, so that I can understand when a fresh classification was made.
17. As an operations engineer, I want a `/health` endpoint that reports the status of the database and vector store, so that I can monitor system readiness in Cloud Run.
18. As an operations engineer, I want the system deployed as a Docker container on Google Cloud Run, so that it scales automatically with batch demand.
19. As an operations engineer, I want all secrets (API keys) passed via environment variables, never hardcoded or checked into source control, so that credentials are managed securely.
20. As an operations engineer, I want a `docker-compose.yml` for local development that runs the API and frontend together, so that developers can test the full stack without Cloud Run.
21. As a merchant data reviewer, I want new brands written to the database with `status=pending_review`, so that I can identify which merchant records require human verification.
22. As a merchant data reviewer, I want new brands to be cached immediately despite pending status, so that subsequent lookups on the same transaction text are served from cache without re-running the full agent.
23. As a system architect, I want the LLM, search, embedding, vector store, and database providers to each be swappable via configuration, so that no single external dependency is a permanent lock-in.
24. As a system architect, I want pre-call and post-call hooks defined on all LLM and search provider invocations, so that future data desensitisation or audit logging can be added without refactoring provider code.
25. As a system architect, I want the validation pass (LLM Pass 2) to always run on net-new classifications, so that accuracy is consistently guaranteed regardless of Pass 1 confidence score.
26. As a system architect, I want a single Pass 2 disagreement to mark a result as unresolved with no retry, so that the system never enters an infinite classification loop.
27. As a system architect, I want the FAISS vector index to be updated immediately when a new brand is written, so that the index stays current without a separate sync job.
28. As a system architect, I want the category shortlist passed to the LLM to expand dynamically (15 → 30 → 50) when category confidence is low, so that the model has enough candidates to make a correct selection without always paying for a large context.
29. As a developer, I want every module to have pytest tests tied to its specific requirements, so that each piece of code has a clear, requirement-driven test contract.
30. As a developer, I want all configuration driven by `config/default.yaml` with environment variable overrides, so that behaviour can be changed at deploy time without code changes.
31. As a developer, I want a `.env.example` file documenting all required secrets, so that new developers can set up the environment without reading source code.
32. As a product manager, I want the Streamlit frontend to expose model selection, web search toggle, and validation pass toggle in a sidebar, so that the demo audience can observe the cost and accuracy trade-offs live.
33. As a product manager, I want a cost estimate breakdown (extraction pass vs. validation pass, input vs. output tokens, USD) shown before a batch runs, so that stakeholders can make informed decisions about batch processing costs.

---

## Constraints & Assumptions

- **Batch throughput over single-call latency.** The system is optimised for millions of requests in batch mode. Individual `/classify` call latency is not a design constraint.
- **Accuracy over cost.** LLM Pass 2 always runs on new classifications. Token cost is a secondary concern to classification correctness.
- **No hallucinated output.** An unresolved result is always preferable to a confident wrong answer. Downstream systems must handle `null` brand values.
- **Category taxonomy is pre-supplied.** The 100+ category list is provided by the team and seeded at startup. It is slow-changing and treated as stable. Future extensions are possible via database update.
- **Google ADK is the primary provider.** The implementation uses `google.adk.agents` and `google.adk.tools.google_search` from the real ADK framework.
- **SQLite for Phase 1–2.** SQLite is sufficient for development and initial deployment. PostgreSQL is the production target but not required at launch.
- **FAISS for Phase 1.** FAISS is used for local vector search. Pinecone and Vertex AI are future stubs.
- **Data desensitisation is future work.** Pre/post LLM call hooks are stubbed but contain no PII logic at launch.
- **Auth is a no-op at launch.** The auth middleware hook is implemented as a pass-through. No delivery commitment on real auth.
- **OpenAI and Pinecone are placeholder stubs.** No delivery commitment; present as documented intent only.
- **Cache is trusted.** Cache hits return stored results without re-validation. The assumption is that stored classifications were correctly validated at first write.
- **Cache is toggleable.** The cache can be disabled via configuration for force-reclassification scenarios.
- **`ralph/` tooling is out of scope.** The `ralph/` automation scripts in the repo are developer scaffolding, not part of FranklinAgent.
- **Batch max N = 10.** Beyond 10 texts per LLM prompt, JSON parsing reliability degrades. This is a hard cap.
- **New brands write immediately.** New merchant records are written with `status=pending_review` immediately — no hold queue. Human review is asynchronous.
- **Single Pass 2 disagreement is terminal.** There is no retry on unresolved. No third LLM call, no alternative search query.

---

## Implementation Decisions

### Modules to Build

**`app/agent/orchestrator.py`** — Main agent flow coordinator. Implements the 7-step pipeline: cache check → vector search → web search → LLM extraction → rule-based validation → LLM validation → write result. Calls into cache, vector store, search provider, LLM provider, and repository layers. Does not hold business logic itself — delegates to dedicated sub-modules.

**`app/agent/cache.py`** — Exact-match cache. Queries the `enquiries` table by `input_text`. Returns a cached result or signals a miss. Cache behaviour is toggled via config.

**`app/agent/extractor.py`** — LLM Pass 1 wrapper. Constructs the extraction prompt, calls the LLM provider, parses and returns structured extraction output.

**`app/agent/validator.py`** — Rule-based validation (three checks) + LLM Pass 2 wrapper. Constructs the validation prompt, calls the LLM provider, returns agree/disagree with optional corrected answer.

**`app/agent/query_builder.py`** — Search query construction. Either returns raw input text with a system hint, or calls the LLM to generate an optimised search query (controlled by `use_llm_query_builder` config flag).

**`app/providers/base.py`** — Abstract interfaces: `LLMProvider`, `SearchProvider`, `EmbeddingProvider`. Each defines the contract the rest of the system depends on. All providers implement these interfaces.

**`app/providers/google_adk.py`** — Google ADK implementation of `LLMProvider`, `SearchProvider`, and `EmbeddingProvider`. Uses `google.adk.agents.Agent`, `google.adk.tools.google_search`, and the `text-embedding-004` model. Includes no-op pre-call and post-call hook stubs on every LLM and search invocation.

**`app/providers/serpapi.py`** — SerpAPI implementation of `SearchProvider`. Fallback search provider, configurable via `search_provider` config value.

**`app/providers/openai_provider.py`** — OpenAI stub. Placeholder implementation of `LLMProvider`. No delivery commitment.

**`app/vector/base.py`** — Abstract `VectorStore` interface: `search`, `upsert`, `persist`.

**`app/vector/faiss_store.py`** — FAISS implementation. Loaded from disk at startup (`./data/faiss.index` + `./data/faiss_map.json`). Updated and persisted to disk on every `upsert` (i.e., every new brand write). Indexes `brands.name` and all entries in `brands.aliases`.

**`app/vector/pinecone_store.py`** — Pinecone stub. Placeholder. No delivery commitment.

**`app/db/base.py`** — Abstract repository interfaces: `BrandRepository`, `EnquiryRepository`, `CategoryRepository`. All database access goes through these interfaces.

**`app/db/sqlite_repo.py`** — SQLite implementation of all three repository interfaces. Uses the schema defined in `001_initial.sql`. Runs schema migration at startup.

**`app/db/postgres_repo.py`** — PostgreSQL stub. Placeholder. No delivery commitment.

**`app/db/migrations/001_initial.sql`** — DDL for `categories`, `brands`, `enquiries` tables and the `idx_enquiries_input_text` unique index.

**`app/cost/estimator.py`** — `TokenCostEstimator` class. `estimate()` for pre-flight estimates; `record_actuals()` for post-run token tracking. Loads pricing from `config/pricing.yaml`.

**`app/api/routes/classify.py`** — `POST /classify` endpoint. Single text input, full agent pipeline invocation.

**`app/api/routes/batch.py`** — `POST /batch` endpoint. Accepts `texts[]` array and optional `batch_size`. Groups texts into LLM batches, collects per-item results, returns summary and optional token actuals.

**`app/api/routes/cost.py`** — `POST /cost-estimate` endpoint. Delegates to `TokenCostEstimator.estimate()`.

**`app/api/middleware/auth.py`** — No-op auth middleware. Pass-through at launch.

**`app/main.py`** — FastAPI application entry point. Registers routers and middleware.

**`app/config.py`** — Centralised configuration. Loads `config/default.yaml`, applies environment variable overrides. Exposes a typed config object consumed by all modules.

**`frontend/app.py`** — Streamlit application. Three sections: single classification, batch upload/download, sidebar config. Calls the FastAPI layer via HTTP.

### Database Schema

```sql
CREATE TABLE categories (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    level_1 TEXT NOT NULL,
    level_2 TEXT NOT NULL UNIQUE
);
INSERT INTO categories (level_1, level_2) VALUES ('Uncategorized', 'Uncategorized');

CREATE TABLE brands (
    id               TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    brand_group      TEXT,
    category_id      INTEGER NOT NULL REFERENCES categories(id),
    aliases          TEXT NOT NULL DEFAULT '[]',
    website_url      TEXT,
    confidence_score REAL,
    status           TEXT NOT NULL DEFAULT 'pending_review',
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);

CREATE TABLE enquiries (
    id               TEXT PRIMARY KEY,
    input_text       TEXT NOT NULL,
    brand_id         TEXT REFERENCES brands(id),
    confidence_score REAL,
    source           TEXT NOT NULL,
    status           TEXT NOT NULL,
    error_detail     TEXT,
    created_at       TEXT NOT NULL
);
CREATE UNIQUE INDEX idx_enquiries_input_text ON enquiries(input_text);
```

### API Contracts

**POST /classify**
- Request: `{ "text": string }`
- Response 200 (success): `{ brand_id, brand_name, brand_group, category_level_1, category_level_2, confidence_score, is_new_brand, status: "success", audit_id, cached: bool }`
- Response 200 (unresolved): `{ brand_id: null, brand_name: null, brand_group: null, category_level_1: "Uncategorized", category_level_2: "Uncategorized", confidence_score: 0.0, status: "unresolved", audit_id, cached: false }`

**POST /batch**
- Request: `{ "texts": string[], "batch_size"?: int }`
- Response 200: `{ results: [...], summary: { total, resolved, unresolved, errors }, token_actuals: { input_tokens, output_tokens, estimated_cost_usd } }`

**POST /cost-estimate**
- Request: `{ num_texts, batch_size, enable_web_search, enable_validation_pass, extraction_model, validation_model }`
- Response 200: `{ estimated_input_tokens, estimated_output_tokens, estimated_cost_usd, breakdown, num_llm_calls, assumptions }`

**GET /health**
- Response 200: `{ "status": "ok", "db": "ok", "vector_store": "ok" }`

### Configuration Keys

All keys settable via `config/default.yaml` or environment variable override:
- `llm_provider`, `extraction_model`, `validation_model`, `embedding_model`
- `search_provider`, `search_top_k`, `search_snippet_max_chars`, `use_llm_query_builder`
- `vector_store`, `vector_store_path`, `vector_confidence_threshold`
- `db_provider`, `sqlite_path`
- `category_shortlist_size` (15), `category_shortlist_expand_size` (30), `category_shortlist_max` (50), `category_confidence_threshold` (0.6)
- `batch_size` (default 1), `batch_size_max` (10)
- `api_host`, `api_port`
- `cache_enabled` (bool, default true)
- Secrets via env vars only: `GOOGLE_API_KEY`, `SERPAPI_KEY`, `GOOGLE_APPLICATION_CREDENTIALS`

### LLM Provider Hook Points

Every LLM provider call site exposes two no-op stubs:
- `_pre_llm_call(prompt, system)` — called before the API request is sent
- `_post_llm_call(response)` — called after the API response is received

Every search provider call site exposes:
- `_pre_search_call(query)` — called before the search request
- `_post_search_call(results)` — called after results are returned

These are overridable in subclasses for future desensitisation, logging, or audit workflows.

### Batch Processing Logic

- Input: `texts[]` array
- Group into batches of `batch_size` (default 1, max 10)
- Each batch → single LLM prompt listing all N texts → JSON array of N results
- Per-item status: `success | unresolved | error`
- On item error: write to `enquiries` with `status=error` and `error_detail`
- Partial results returned — batch does not fail atomically
- Failed items can be resubmitted individually or as a new batch

### Category Shortlist Expansion

- Default shortlist: top 15 categories by vector similarity to the input text
- If category confidence < 0.6 after Pass 1: expand to 30
- Hard cap: 50 categories maximum per prompt

### Vector Index Management

- FAISS index loaded from disk at API startup
- Index covers `brands.name` + all entries in `brands.aliases[]`
- Vector ID → `brand.id` mapping stored in `faiss_map.json`
- On every new brand write: `upsert` into FAISS index + `persist()` to disk immediately
- Embedding model: `text-embedding-004` (Google), configurable

---

## Dependencies

- **Google ADK** (`google-adk`) — LLM provider, search tool, embedding model
- **FAISS** (`faiss-cpu`) — vector similarity search
- **FastAPI** + **Uvicorn** — REST API layer
- **Streamlit** — demo frontend
- **SQLite** (stdlib) — development database; **PostgreSQL** (future, via psycopg2 or asyncpg)
- **SerpAPI** (`google-search-results`) — fallback search provider
- **Pydantic** — request/response validation
- **PyYAML** — configuration loading
- **pytest** — test framework
- **Docker** + **Google Cloud Run** — containerisation and deployment
- **Google Cloud Build** — CI/CD pipeline for container builds

---

## Testing Decisions

- **Framework:** pytest
- **Principle:** Tests are tied to each development issue. Every module is tested based on its specific requirement — no global test coverage targets.
- **What makes a good test:** Tests verify external behaviour (inputs and outputs of a module's public interface), not internal implementation details. Tests should be independent, deterministic, and fast.
- **Phase 1 testing:** Unit tests for each agent step (cache, vector search, extraction, validation, write) using mocked providers. Each mock replaces the external dependency at the provider interface boundary.
- **Phase 6 testing:** Integration tests using a real SQLite database, a real FAISS index, and mocked LLM/search providers. These tests verify the full orchestrator pipeline end-to-end.
- **Modules with tests in Phase 1:**
  - `orchestrator.py` — full pipeline with all providers mocked
  - `cache.py` — hit and miss paths
  - `extractor.py` — prompt construction and response parsing
  - `validator.py` — rule checks + Pass 2 agree/disagree branches
  - `faiss_store.py` — search and upsert
  - `sqlite_repo.py` — CRUD operations per repository interface
  - `estimator.py` — cost calculation correctness

---

## Open Questions

- **Bulk approval UI for `pending_review` brands:** At million-record scale, a human needs tooling to review and bulk-approve new brand records. The scope and form of this tooling (admin API endpoint, separate internal tool, direct DB access) is undefined and deferred.
- **Cache invalidation strategy:** If a brand's category changes (e.g., a reclassification or data correction), existing `enquiries` cache entries remain stale. No invalidation mechanism is currently defined.
- **Concurrent FAISS writes:** FAISS does not support concurrent writes. If multiple API instances run on Cloud Run and both attempt to write a new brand simultaneously, there is a risk of index corruption or lost writes. This is unresolved for multi-instance deployments.
- **CSV column name contract:** The batch CSV upload expects a `text` column. Behaviour on missing or differently-named columns should be specified precisely.
- **Rate limits and quota management:** No strategy is defined for handling Google ADK or SerpAPI rate limits during large batch runs (e.g., exponential backoff, request queuing, quota monitoring).
- **FAISS index initialisation:** Behaviour when the system starts with no existing FAISS index file (first-time deployment) needs to be specified — create an empty index vs. fail with a clear error.

---

## Out of Scope

- **Real auth middleware.** The hook is a no-op. No API key, OAuth, or JWT implementation is planned.
- **Admin UI for brand review.** Human review of `pending_review` brands is done outside the system.
- **Data desensitisation logic.** Pre/post LLM call hooks are stubbed but contain no PII handling.
- **OpenAI provider.** Stub only — no working implementation.
- **Pinecone vector store.** Stub only — no working implementation.
- **PostgreSQL repository.** Stub only — no working implementation.
- **`ralph/` automation scripts.** Developer scaffolding, not part of FranklinAgent.
- **Retry logic on unresolved classifications.** A single Pass 2 disagreement is always terminal.
- **Real-time streaming responses.** All API responses are synchronous JSON.
- **Multi-tenancy.** Single-tenant internal tool only.

---

## Further Notes

### Key Design Decisions (from plan)

| Decision | Choice | Rationale |
|---|---|---|
| Validation pass frequency | Always (on cache misses) | Accuracy over cost |
| Batch default N | 1 | Safe default; cost savings opt-in |
| Batch max N | 10 | Beyond 10, JSON parsing reliability degrades |
| Category shortlist | 15 → 30 → 50 | Balances token cost vs recall |
| New brand write | Immediate, status=pending_review | Enables caching; human reviews asynchronously |
| Unresolved action | null brand + Uncategorized + audit log | No hallucinated output ever reaches downstream |
| FAISS index update | On every new brand write | Index stays current; disk persist on every upsert |
| SQLite → PostgreSQL | Repository pattern | Zero application code change to swap |
| Auth | No-op middleware | One file to change when needed |
| Cache | Trusted on hit, toggleable | Avoids redundant LLM cost; force-reclassify via config |

### Extraction Prompt Structure

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

### Validation Prompt Structure

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

### Model Pricing Configuration (`config/pricing.yaml`)

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

### Token Cost Estimation Logic

- `batches = ceil(num_texts / batch_size)`
- `avg_search_tokens = 500 × 3 × batch_size`
- `avg_category_tokens = 15 × 20`
- `avg_input_overhead = 200`
- `extraction_input = batches × (overhead + search + category)`
- `extraction_output = batches × batch_size × 80`
- `validation_input = extraction_input + extraction_output`
- `validation_output = extraction_output`

### Deployment

- **Dockerfile:** Python 3.12-slim, exposes port 8080, runs Uvicorn
- **docker-compose.yml:** `api` service (8080) + `frontend` service (8501), shared `./data` volume
- **Cloud Run:** 2Gi memory, managed platform, `us-central1`, secrets via env vars
- **Separate Cloud Run service** for Streamlit frontend with `API_URL` env var pointing to the API service

### Implementation Phases

- **Phase 1 — Core Agent:** DB schema, SQLite repo, category seed, FAISS store, Google ADK provider, orchestrator Steps 1–7, unit tests
- **Phase 2 — API Layer:** FastAPI skeleton, `/classify`, `/health`, auth hook, Dockerfile, docker-compose
- **Phase 3 — Batch Mode:** Batch orchestrator, `/batch`, CSV parsing, per-item error handling
- **Phase 4 — Cost Estimation:** `TokenCostEstimator`, `pricing.yaml`, `/cost-estimate`
- **Phase 5 — Frontend:** Streamlit single + batch views, `Dockerfile.frontend`, Cloud Run deployment
- **Phase 6 — Hardening:** SerpAPI provider, OpenAI stub, Pinecone stub, PostgreSQL stub, integration tests, structured logging, README

---

*Generated: 2026-05-03*
