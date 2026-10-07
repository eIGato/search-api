# Nevis Search API

A search API for wealth advisors across their clients and the clients' documents.

* **Clients** are found by name, email and description, including partial and misspelled input:
  `NevisWealth` → `john.doe@neviswealth.com`, `nevis welth` → the same client.
* **Documents** are found by keyword *and* by meaning: `address proof` → a document about a
  `utility bill`, `inheritance for grandchildren` → a last will.
* **Summaries** of documents come from Claude when an API key is configured, otherwise from an
  extractive summarizer. Both paths work without any setup.

Stack: Python 3.13, FastAPI, PostgreSQL 18 with `pgvector` and `pg_trgm`, local ONNX embeddings
(`fastembed`, `all-MiniLM-L6-v2`), optional Claude (`claude-haiku-4-5`).

## Quick start

Requirements: Docker with Compose.

```bash
cp .env.example .env                  # optional: API key, Claude key, ports... (gitignored)
docker compose up -d --build          # API on http://localhost:8000, Postgres on localhost:5433
python3 scripts/seed_demo.py          # optional: 3 demo clients with 8 documents (stdlib only)
curl 'http://localhost:8000/search?q=NevisWealth'
```

* Swagger UI: http://localhost:8000/docs (ReDoc: `/redoc`). A static copy of the spec is in
  [`docs/openapi.json`](docs/openapi.json).
* The first build downloads the embedding model (~90 MB) into the image. After that the service
  needs no network access.
* Port 8000 taken? Set `API_PORT` in `.env` (or `API_PORT=8010 docker compose up -d`).
* Configuration lives in `.env`, which is optional and gitignored. Every setting is documented
  in [`.env.example`](.env.example). Compose passes the file to the API container, and local
  runs (`uv run ...`) read it too. The most relevant settings:

| Variable | Default | Effect |
|---|---|---|
| `API_KEY` | unset | When set, every endpoint except `/health` requires the `X-API-Key` header. |
| `ANTHROPIC_API_KEY` | unset | Enables LLM summaries and LLM query expansion. |
| `LLM_MODEL` | `claude-haiku-4-5` | Claude model for those features. |
| `LLM_QUERY_EXPANSION` | `true` | Set to `false` to keep LLM summaries but skip per-query LLM calls. |
| `WEB_CONCURRENCY` | `1` | Worker processes. Each loads its own copy of the model (see [Concurrency](#concurrency-and-performance)). |
| `EMBEDDING_THREADS` | one per physical core | ONNX Runtime threads per embedding call. |
| `DB_POOL_SIZE`, `DB_MAX_OVERFLOW` | `10`, `10` | Connection pool per worker. |
| `MIN_SEMANTIC_SIMILARITY` | `0.3` | Cosine similarity below which a semantic-only match counts as noise. |
| `CLIENT_FUZZY_THRESHOLD` | `0.4` | `pg_trgm` strict word similarity required for a fuzzy client match. |

Defaults and validation are in [`src/search_api/config.py`](src/search_api/config.py). With
`ANTHROPIC_API_KEY` set on an instance others can reach, also set `API_KEY`: every new search
query may trigger a billed LLM call.

## API

| Method | Path | Success | Errors |
|---|---|---|---|
| `POST` | `/clients` | `201` + `Location` header | `409` email already used (case-insensitive), `422` |
| `GET` | `/clients/{client_id}` | `200` | `404` |
| `POST` | `/clients/{client_id}/documents` | `201` + `Location` header | `404` unknown client, `422` |
| `GET` | `/clients/{client_id}/documents` | `200`, newest first | `404` |
| `GET` | `/clients/{client_id}/documents/{document_id}` | `200` | `404` |
| `GET` | `/clients/{client_id}/documents/{document_id}/summary` | `200` | `404` |
| `GET` | `/search?q=…[&type=client\|document][&client_id=…][&limit=20]` | `200` | `422`, also for `client_id` with `type=client` |
| `GET` | `/health` | `200` | `500` if the database is unreachable |

With `API_KEY` set, a missing or wrong key returns `401`. Errors have the shape
`{"detail": "..."}` (validation errors: FastAPI's list of field errors).

Additions to the model in the assignment:

* `Client.created_at`.
* Read endpoints for clients and documents. They also back the `Location` headers.
* A summary endpoint.
* A typed result shape for `/search`, described next.

### Search results

`/search` returns a single list of clients and documents sorted by relevance. Each item has a
`type` discriminator, a `score` between 0 and 1 and a `matched_by` list that explains the match:

* `matched_by` values for clients: `name`, `email`, `description`.
* `matched_by` values for documents: `keyword`, `semantic`.
* Document results also include a `snippet`: the matching passage, with keywords in `**bold**`.

The `X-Search-Expansions` response header lists the related phrases the query was expanded with
(JSON array).

## Example queries and responses

The responses below are real output from the demo data (`scripts/seed_demo.py`), shortened
where marked.

**Create a client**

```bash
curl -X POST localhost:8000/clients -H 'Content-Type: application/json' -d '{
  "first_name": "John", "last_name": "Doe", "email": "john.doe@neviswealth.com",
  "description": "Tech founder, interested in sustainable investing.",
  "social_links": ["https://www.linkedin.com/in/johndoe"]}'
```
```json
{
  "id": "8052de38-218c-4e3d-b982-c87ea83b693d",
  "first_name": "John",
  "last_name": "Doe",
  "email": "john.doe@neviswealth.com",
  "description": "Tech founder, interested in sustainable investing.",
  "social_links": ["https://www.linkedin.com/in/johndoe"],
  "created_at": "2026-10-06T17:08:09.338810Z"
}
```

**Add a document**

```bash
curl -X POST localhost:8000/clients/8052de38-218c-4e3d-b982-c87ea83b693d/documents \
  -H 'Content-Type: application/json' -d '{"title": "Utility bill - March 2026",
  "content": "Electricity utility bill issued to John Doe, 12 Baker Street, London. Billing period March 2026. Amount due: 84.20 GBP."}'
```
```json
{
  "id": "c53395b5-8002-4842-ab8b-078b3b027e2a",
  "client_id": "8052de38-218c-4e3d-b982-c87ea83b693d",
  "title": "Utility bill - March 2026",
  "content": "Electricity utility bill issued to John Doe, 12 Baker Street, London. Billing period March 2026. Amount due: 84.20 GBP.",
  "created_at": "2026-10-06T17:08:09.438249Z"
}
```

**Find a client by a part of their email:** `GET /search?q=NevisWealth`

```json
[
  {
    "type": "client",
    "score": 0.9,
    "matched_by": ["email"],
    "client": {
      "id": "8052de38-218c-4e3d-b982-c87ea83b693d",
      "first_name": "John",
      "last_name": "Doe",
      "email": "john.doe@neviswealth.com",
      "description": "Tech founder, interested in sustainable investing.",
      "social_links": ["https://www.linkedin.com/in/johndoe"],
      "created_at": "2026-10-06T17:08:09.338810Z"
    }
  }
]
```

**Find documents by meaning:** `GET /search?q=address proof&limit=2`

```
X-Search-Expansions: ["utility bill", "electricity bill", "gas bill", "water bill", "council tax bill", "bank statement", ...]
```
```json
[
  {
    "type": "document",
    "score": 1.0,
    "matched_by": ["keyword", "semantic"],
    "snippet": "Residential **tenancy** **agreement** between the landlord and tenant Maria Rossi for the apartment at Via Roma 5, Milan, starting January",
    "document": {
      "id": "f4cacf4f-9989-4440-8eba-3773993e7594",
      "client_id": "92bc4815-b1c2-46c6-84b0-e8aebed095b4",
      "title": "Tenancy agreement",
      "content": "Residential tenancy agreement between the landlord and tenant Maria Rossi for the apartment at Via Roma 5, Milan, starting January 2025.",
      "created_at": "2026-10-06T17:08:09.652121Z"
    }
  },
  {
    "type": "document",
    "score": 0.9761,
    "matched_by": ["keyword", "semantic"],
    "snippet": "**Electricity** **utility** **bill** issued to John Doe, 12 Baker Street, London. **Billing** period March 2026. Amount",
    "document": {
      "id": "c53395b5-8002-4842-ab8b-078b3b027e2a",
      "client_id": "8052de38-218c-4e3d-b982-c87ea83b693d",
      "title": "Utility bill - March 2026",
      "content": "Electricity utility bill issued to John Doe, 12 Baker Street, London. Billing period March 2026. Amount due: 84.20 GBP.",
      "created_at": "2026-10-06T17:08:09.438249Z"
    }
  }
]
```

More queries against the demo data (type, score, `matched_by`, title or email):

| Query | Results |
|---|---|
| `nevis welth` (typo) | client 0.35 `email` john.doe@neviswealth.com |
| `Rossi` | client 0.95 `name,email` maria@rossi-family.it; document 1.0 `keyword,semantic` "Last will and testament"; document 0.49 `keyword` "Tenancy agreement" |
| `inheritance for grandchildren` | document 0.5 `semantic` "Last will and testament" (no shared keywords) |
| `risk appetite` | document 1.0 "Investment policy statement" (says "risk tolerance"); client 0.5 `description` maria@rossi-family.it ("conservative risk profile") |
| `chocolate cake recipe` | `[]` |

**Summary:** `GET /clients/{client_id}/documents/{document_id}/summary`

```json
{
  "document_id": "58e2b263-6e48-48c7-816b-7c6703d74c50",
  "summary": "I, Maria Rossi, leave my property in Tuscany to my grandchildren in equal shares. Executor: Luca Rossi.",
  "method": "extractive"
}
```

With `ANTHROPIC_API_KEY` set, `method` is `llm` and the summary is a 2-3 sentence abstract.

**Errors**

```text
GET  /clients/nope                                   → 404 {"detail": "Client not found"}
POST /clients {..."email": "JOHN.DOE@neviswealth.com"} → 409 {"detail": "A client with this email already exists"}
GET  /search?q=%20                                   → 422 (q must contain a non-space character)
```

## How search works

### Clients: lexical matching

Clients are short structured records. An advisor types a name, a fragment of an email or a
company domain, often misspelled. Postgres maintains normalized copies of the fields as
generated columns, and each match type gets a fixed score tier:

| Match | Example | Score |
|---|---|---|
| Exact email, or full name in either order | `Doe John` | 1.0 |
| Exact first or last name, any length | `Li` | 0.95 |
| Substring of the name or email after stripping non-alphanumerics (3+ characters) | `NevisWealth`, `nevis-wealth` → `johndoeneviswealthcom` | 0.9 |
| Fuzzy: `pg_trgm` strict word similarity ≥ 0.4, scaled by the similarity | `Jane Smiht`, `marya rosi` | ≤ 0.85 |
| Description: full-text match | `surgeon` | 0.6 |
| Description: full-text match through a query expansion | `risk appetite` → "risk profile" | 0.5 |

All of these are backed by GIN indexes (trigram and `tsvector`). Scores are computed in SQL,
so when more clients match than the candidate pool holds, the weakest matches are dropped: `smith`
keeps the client named Smith even among many Smithsons.

### Documents: hybrid retrieval with query expansion

1. **Query expansion.** A curated thesaurus of wealth-management concepts
   ([`thesaurus.py`](src/search_api/thesaurus.py)) maps concepts to the documents that satisfy
   them, and back: "proof of address" ↔ utility bill, bank statement, tenancy agreement; also KYC,
   identity, source of wealth, risk profile, tax, estate planning, retirement and ESG.

   When `ANTHROPIC_API_KEY` is set, Claude suggests up to 5 more phrases. A call costs about
   $0.0005 and adds 0.7-2 s of latency (measured), so it is made only when it can help:
   * the thesaurus does not cover the query;
   * the query does not name a client directly (names, emails and domains have no related
     terms; the model returns an empty list for them anyway);
   * the query is not an email address;
   * the search includes documents.

   Results, including empty ones, are cached in Postgres per query and model. The cache is
   shared by all workers and survives restarts. Failed calls are not cached. Identical
   concurrent queries in one process share one call. LLM phrases expand the document search
   only; client descriptions use the thesaurus.
2. **Keyword retriever.** Postgres full-text search (English stemming, title weighted above
   content) for the query OR any expansion phrase.
3. **Semantic retriever.** The query and each expansion are embedded. Documents are stored as
   overlapping ~120-word chunks, each prefixed with the title, so long documents are not
   truncated by the model. Matching uses an HNSW cosine index; within one client (`client_id`)
   it is exact instead, because an approximate scan filtered by client only sees the chunks
   nearest across all clients and can miss that client's documents. Each document keeps its best
   chunk similarity, with expansions weighted 0.9. Matches below `MIN_SEMANTIC_SIMILARITY` are
   dropped, so unrelated queries return nothing instead of the "closest" noise.
4. **Fusion.** Reciprocal Rank Fusion (k = 60) of the two ranked lists, normalized so that rank 1
   in both lists scores 1.0 and rank 1 in one list scores 0.5.

### Ranking clients and documents together

* Clients the query names directly (score ≥ 0.9) come first: if you type a client's name, you
  most likely want the client, not every document that mentions them.
* Everything else is sorted by score, with clients first on ties.
* Scores of the two types come from different methods. They are meaningful within a type and
  only roughly comparable across types. Use `type=` to search one kind only.

## Design decisions and tradeoffs

* **One Postgres instead of Postgres + Elasticsearch + a vector DB.** Trigram, full-text and
  vector search all live next to the data. That means one container, transactional consistency
  (a document is searchable the moment `POST` returns) and no sync jobs. A dedicated search
  engine would become worth it at millions of documents, or for features like faceting and
  per-field analyzers. Until then, more moving parts buy nothing.
* **Local embeddings instead of an embeddings API.** Client documents in wealth management are
  personal financial data. Sending them to a third party is a compliance decision, not a coding
  one. A local model also means reviewers need no keys, behaviour is reproducible, and there are
  no per-call costs or network latency. The cost: a small model is less capable.
* **Why `all-MiniLM-L6-v2`.** I benchmarked 9 fastembed models (from 67 MB up to 1.2 GB) on
  wealth-management queries. MiniLM separated relevant from unrelated documents best: for
  "risk appetite", the investment policy scored 0.34 and the next document 0.11. It is also tiny
  and fast. Bigger models did not help where it matters.
* **The thesaurus exists because embeddings alone fail the assignment's own example.** No model
  I tested, including 1 GB ones, reliably ranks a utility bill first for "address proof". The
  link is domain knowledge ("a utility bill is accepted as proof of address"), not language
  similarity. The set of document types an advisor handles is finite and well known, so a
  curated concept list is deterministic, explainable (`X-Search-Expansions`), free and easy to
  extend. Embeddings still handle everything the thesaurus does not cover ("inheritance for
  grandchildren" → will). The optional LLM expansion generalizes further ("college fund for
  kids" → "529 plan"), at the cost of 0.7-2 s and ~$0.0005 per new query. That is why it is
  skipped where it cannot help, and cached.
* **Rank fusion instead of score blending.** `ts_rank` and cosine similarity live on different,
  query-dependent scales. RRF uses only ranks, so no per-model calibration is needed. Its
  weakness: a document found by both retrievers on generic words can outrank a strong
  single-retriever match.
* **Embedding at write time, synchronously.** Simple, and the document is immediately searchable.
  Embedding runs on the CPU: a one-page (~500-word) document takes about 0.5 s to `POST` on a
  laptop. Very large documents (or PDFs needing OCR) would call for an async job queue, with
  the document marked "indexing" until done.
* **Summaries: on demand, cached, with a safe fallback.**
  * Generating them on write would make the LLM a dependency of ingestion and spend tokens on
    documents nobody reads.
  * The extractive fallback (the sentences closest to the document's centroid embedding) never
    invents facts. That matters for financial documents.
  * LLM summaries are cached. Fallback summaries are cached only when no LLM is configured, so
    a transient LLM outage does not stick.
  * Exactly one request generates a document's summary, even when many arrive at once in
    different workers. The first one takes a short lease on the document row; the others poll
    for its result. The lease expires if its holder dies. Measured with Claude Haiku 4.5: about
    1 s and $0.0004-0.0011 per one-page document.
* **Malformed IDs return 404, not 422.** The contract defines IDs as opaque strings, so a
  malformed ID is just an ID that does not exist.
* **Emails are unique (case-insensitive).** Duplicate clients would make search results
  ambiguous for advisors, so a duplicate returns `409`.

## Concurrency and performance

* **The service is async end to end.** CPU-bound embedding runs in a thread pool; ONNX Runtime
  releases the GIL and uses all cores.
* **No request holds a database connection during slow work.** Each request runs short SQL
  phases and returns its connection to the pool before embedding or calling the LLM. A small
  pool therefore serves many concurrent requests, and overload shows up as latency instead of
  pool timeouts. A test with a single-connection pool and a slow LLM checks this.
* **Races.**
  * Duplicate emails: a unique index, so concurrent identical `POST /clients` give exactly one
    `201` and the rest `409`.
  * Summaries: a lease (see above).
  * Expansion cache inserts: `ON CONFLICT DO NOTHING`.
* **Throughput is bound by CPU.** Measured on an 8-thread laptop with demo data:

  | Load | Throughput | p50 / p95 latency |
  |---|---|---|
  | 1 search at a time | ~50/s | 20 / 41 ms |
  | 20 concurrent searches | ~60/s | ~0.3 / 0.6-0.9 s |
  | 100 concurrent searches | ~60/s | ~1.2-1.5 / 3 s |
  | 20 concurrent `POST` documents (~300 words) | ~17/s | ~1.1 / 1.7 s |

  A thesaurus-expanded query such as "address proof" costs ~24 ms of CPU for 15 embeddings and
  ~29 ms for 15 vector lookups. More workers (`WEB_CONCURRENCY` 2-8, with `EMBEDDING_THREADS`
  split accordingly) did not raise throughput beyond noise, because ONNX Runtime already
  uses all cores. The default is therefore one worker per container. To scale, add CPUs or
  replicas; the database is not the bottleneck (~60% of one core at peak).

## Testing

```bash
uv sync
uv run pytest                 # all tests; starts a throwaway pgvector container (Docker needed)
uv run pytest -m "not model"  # skip the tests that need the ~90 MB embedding model
TEST_DATABASE_URL=postgresql+asyncpg://search:search@localhost:5433/search uv run pytest  # reuse a DB
uv run ruff check . && uv run mypy src tests
```

Tests ignore `.env`, so local settings and keys cannot change their outcome or trigger billed
calls.

* `tests/test_units.py`: chunking, thesaurus matching, rank fusion, extractive summaries,
  LLM failure fallbacks (timeouts, connection errors, rate limits) and sharing of identical
  in-flight expansion calls.
* `tests/test_api.py`: every endpoint against real Postgres with a deterministic bag-of-words
  embedder. Covers:
  * validation, `404` / `409` / `401` responses and the `Location` headers;
  * the matching tiers above;
  * LIKE/SQL special characters (`%`, `_`, `' OR 1=1 --`);
  * short-query noise, long-document chunking and type/client filters;
  * candidate limits: the strongest client matches survive them, and search within one client
    is exhaustive even where the planner would use the approximate vector index;
  * the LLM paths, using a fake LLM: fallbacks, when the LLM is (not) asked, the expansion
    cache across app instances, one LLM call for concurrent summary requests;
  * no connection held during slow work (single-connection pool, slow LLM).
* `tests/test_semantic.py` (marker `model`): golden queries on a realistic corpus with the real
  model.
  * Recall: the right documents rank first, including the assignment's examples.
  * Precision: unrelated queries return no documents.

## Local development

```bash
cp .env.example .env                          # optional; DATABASE_URL already points at compose's db
docker compose up -d db                       # Postgres on localhost:5433
uv sync
uv run alembic upgrade head
uv run search-api --reload --port 8000        # DATABASE_URL defaults to localhost:5433
```

## Deployment (single server)

The compose file runs as is on any Linux server with Docker. Measured with demo data, the
stack needs:

| Resource | Recommended | Why |
|---|---|---|
| CPU | 2 vCPU (1 works) | Search p50 18 ms on 2 vCPU, 57 ms on 1; embedding is the only CPU-heavy work. |
| RAM | 2 GB | API ~250 MB idle (mostly the model), Postgres 40-170 MB. Embedding a maximum-size document (500k characters) peaks at ~1.3 GB. |
| Disk | 20 GB | Images ~1.4 GB, data tens of MB. |

Build the image locally (the server then needs neither the sources nor network access to
the model) and copy it over with the compose file:

```bash
docker compose build api
docker save search-api:latest | gzip | ssh user@server 'gunzip | docker load'
ssh user@server mkdir -p search-api && scp docker-compose.yml user@server:search-api/
```

On the server, create `.env` before the first start (Postgres takes its password only when it
initializes the volume), then start the stack from the loaded image:

```bash
cd search-api
printf 'API_PORT=80\nAPI_KEY=%s\nPOSTGRES_PASSWORD=%s\n' "$(openssl rand -hex 24)" "$(openssl rand -hex 24)" > .env
docker compose up -d --no-build
grep API_KEY .env             # the key to hand out
```

Load the demo data from your machine: `python3 scripts/seed_demo.py http://<server> <API_KEY>`.
To update, repeat the `docker save` step and run `docker compose up -d --no-build` again.

* Postgres is published on `127.0.0.1` only. Docker bypasses host firewalls such as `ufw`,
  so a port published on all interfaces would be reachable from the internet.
* Both containers restart automatically after a crash or a reboot.
* Without a domain and TLS, the API key travels in plain text: use a dedicated key and rotate
  it afterwards. Put a reverse proxy with TLS (e.g. Caddy) in front for anything longer-lived.
* `ANTHROPIC_API_KEY` is optional. If you set it on a public server, also set a spend limit
  in the Anthropic console.

## Project layout

```
src/search_api/
  api.py          HTTP endpoints
  search.py       client matching, document retrievers, rank fusion, merging
  thesaurus.py    domain concepts for query expansion
  embeddings.py   embedder interface + fastembed implementation
  chunking.py     sentence-aware overlapping chunks
  llm.py          Claude summaries and query expansion, with graceful fallbacks
  summary.py      summaries: extractive, and LLM with a generation lease
  models.py       SQLAlchemy models (generated search columns, vector chunks)
  schemas.py      request/response models
  config.py       settings (environment variables)
migrations/       Alembic migrations
scripts/          demo data loader
tests/
```

## With more time / in production

* **Access control.** Advisors should only see their own clients. Add an advisor/tenant ID to
  every table and filter on it in every query, ideally with Postgres row-level security.
* **Evaluation.** A larger labelled query set with recall@k and MRR, to tune thresholds, RRF
  weights and the model choice on real advisor queries rather than intuition.
* **Model upgrades.** Changing the embedding model needs a migration (the vector dimension is
  checked at startup) and a re-embedding job.
* **Other features.**
  * Highlighting for semantic-only matches.
  * Pagination cursors.
  * Async ingestion for large files (PDF/OCR).
  * One SQL statement for all per-expansion vector lookups (they take about half of the time
    of an expanded query).
  * Metrics on zero-result queries, to grow the thesaurus.
