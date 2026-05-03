# Issues: FranklinAgent — AI Merchant Classification Agent

PRD: #15

## Dependency graph

```mermaid
flowchart TD
    16["#16: Project scaffold & config"]:::afk
    17["#17: DB schema & repositories"]:::afk
    18["#18: FAISS vector store"]:::afk
    19["#19: Google ADK LLM & search"]:::afk
    20["#20: Core orchestrator pipeline"]:::afk
    21["#21: POST /classify + Docker"]:::afk
    22["#22: POST /batch endpoint"]:::afk
    23["#23: POST /cost-estimate"]:::afk
    24["#24: Streamlit frontend"]:::afk
    25["#25: Hardening & integration tests"]:::afk

    16 --> 17
    17 --> 18
    17 --> 19
    18 --> 20
    19 --> 20
    20 --> 21
    21 --> 22
    21 --> 23
    21 --> 25
    22 --> 24
    23 --> 24

    classDef afk fill:#4a90d9,color:#fff,stroke:#2c5f8a
    classDef hitl fill:#ffd700,color:#333,stroke:#b8960c
```

**Legend:** Blue = AFK (no human required) · Yellow = HITL (human decision needed)

## Issue index

| # | Title | Type | Blocked by |
|---|-------|------|------------|
| #16 | Project scaffold & configuration system | AFK | — |
| #17 | Database schema, seed data & repositories | AFK | #16 |
| #18 | FAISS vector store & embedding provider | AFK | #17 |
| #19 | Google ADK LLM & search provider | AFK | #17 |
| #20 | Core 7-step orchestrator pipeline | AFK | #17, #18, #19 |
| #21 | POST /classify + GET /health FastAPI endpoints & Docker | AFK | #20 |
| #22 | Batch mode — POST /batch endpoint | AFK | #21 |
| #23 | Cost estimation — POST /cost-estimate endpoint | AFK | #21 |
| #24 | Streamlit demo frontend | AFK | #21, #22, #23 |
| #25 | Hardening: alternate provider stubs, integration tests & logging | AFK | #21 |
```
