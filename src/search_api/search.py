"""Search across clients (lexical/fuzzy) and documents (hybrid keyword + semantic).

Clients are short structured records looked up by name or email, often partially or with typos
("NevisWealth" -> john.doe@neviswealth.com), so they are matched lexically with pg_trgm and
full-text search on the description.

Documents are free text where the wording of the query and of the document often differ
("address proof" vs "utility bill"), so two retrievers run and are fused:
- keyword: Postgres full-text search over title + content;
- semantic: cosine similarity between the query embedding and document chunk embeddings.
Both use the query plus its expansions (domain thesaurus, optionally an LLM).
"""

import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from sqlalchemy import FromClause, RowMapping, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from search_api import models, schemas
from search_api.config import Settings
from search_api.db import release_connection
from search_api.embeddings import Embedder, embed_queries
from search_api.llm import LLMClient, normalize_query
from search_api.thesaurus import expand_query

SearchType = Literal["client", "document"]

# Client score tiers. Scores are comparable within a result type; across types they only
# express "strong" vs "weak" matches (see README, "Ranking").
EXACT_MATCH_SCORE = 1.0
NAME_PART_MATCH_SCORE = 0.95
SUBSTRING_MATCH_SCORE = 0.9
FUZZY_MATCH_WEIGHT = 0.85
DESCRIPTION_MATCH_SCORE = 0.6
DESCRIPTION_RELATED_SCORE = 0.5  # matched only through a query expansion
# Client matches at least this strong (exact, name part, substring) mean the query names the
# client directly; they are listed before documents that merely mention the client.
DIRECT_CLIENT_MATCH_SCORE = SUBSTRING_MATCH_SCORE

SNIPPET_CHARS = 300
MAX_EXPANSIONS = 20


@dataclass
class ClientHit:
    client_id: uuid.UUID
    score: float
    matched_by: list[schemas.ClientMatch]


@dataclass
class DocumentHit:
    document_id: uuid.UUID
    score: float
    matched_by: list[schemas.DocumentMatch]
    snippet: str


@dataclass
class _Candidate:
    keyword_rank: int | None = None
    semantic_rank: int | None = None
    headline: str | None = None
    best_chunk: str | None = None
    similarity: float = 0.0
    matched_by: list[schemas.DocumentMatch] = field(default_factory=list)


def compact(value: str) -> str:
    """Lowercase alphanumerics only; mirrors clients.compact_text in the database."""
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


# --- Clients -----------------------------------------------------------------------------------


def _tsquery_sql(n_expansions: int) -> str:
    """The query (web-search syntax) OR'ed with each expansion phrase."""
    return " || ".join(
        ["websearch_to_tsquery('english', :q)"]
        + [f"phraseto_tsquery('english', :e{i})" for i in range(n_expansions)]
    )


def _tsquery_params(query: str, expansions: Sequence[str]) -> dict[str, object]:
    return {"q": query, **{f"e{i}": e for i, e in enumerate(expansions)}}


def _client_sql(n_expansions: int) -> str:
    return f"""
        SELECT c.id,
               lower(c.email) = :ql AS email_exact,
               :ql IN (lower(c.first_name || ' ' || c.last_name),
                       lower(c.last_name || ' ' || c.first_name)) AS name_exact,
               :ql IN (lower(c.first_name), lower(c.last_name)) AS name_part_exact,
               (:qc_ok AND split_part(c.compact_text, '|', 1) LIKE :qc_like) AS name_substring,
               (:qc_ok AND split_part(c.compact_text, '|', 2) LIKE :qc_like) AS email_substring,
               word_similarity(:ql, lower(c.first_name || ' ' || c.last_name)) AS name_similarity,
               word_similarity(:ql, lower(c.email)) AS email_similarity,
               c.description_tsv @@ websearch_to_tsquery('english', :q) AS description_match,
               c.description_tsv @@ t.tsq AS description_related
        FROM clients c, (SELECT {_tsquery_sql(n_expansions)} AS tsq) t
        WHERE (:qc_ok AND c.compact_text LIKE :qc_like)
           OR :ql <% c.name_email_text
           OR lower(c.first_name) = :ql
           OR lower(c.last_name) = :ql
           OR c.description_tsv @@ t.tsq
        LIMIT :pool
    """


def _client_match_scores(
    row: RowMapping, fuzzy_threshold: float
) -> dict[schemas.ClientMatch, float]:
    """Best score per matched field for one candidate row of _client_sql()."""
    signals: list[tuple[bool, schemas.ClientMatch, float]] = [
        (row["email_exact"], "email", EXACT_MATCH_SCORE),
        (row["name_exact"], "name", EXACT_MATCH_SCORE),
        (row["name_part_exact"], "name", NAME_PART_MATCH_SCORE),
        (row["name_substring"], "name", SUBSTRING_MATCH_SCORE),
        (row["email_substring"], "email", SUBSTRING_MATCH_SCORE),
        (
            row["name_similarity"] >= fuzzy_threshold,
            "name",
            FUZZY_MATCH_WEIGHT * row["name_similarity"],
        ),
        (
            row["email_similarity"] >= fuzzy_threshold,
            "email",
            FUZZY_MATCH_WEIGHT * row["email_similarity"],
        ),
        (row["description_match"], "description", DESCRIPTION_MATCH_SCORE),
        (row["description_related"], "description", DESCRIPTION_RELATED_SCORE),
    ]
    scores: dict[schemas.ClientMatch, float] = {}
    for matched, field_name, score in signals:
        if matched:
            scores[field_name] = max(scores.get(field_name, 0.0), score)
    return scores


async def search_clients(
    session: AsyncSession, query: str, expansions: Sequence[str], settings: Settings
) -> list[ClientHit]:
    ql = " ".join(query.lower().split())
    qc = compact(query)
    # Substring matching on 1-2 characters would match almost everyone.
    qc_ok = len(qc) >= 3

    await session.execute(
        text("SELECT set_config('pg_trgm.word_similarity_threshold', :t, true)"),
        {"t": str(settings.client_fuzzy_threshold)},
    )
    rows = (
        await session.execute(
            text(_client_sql(len(expansions))),
            {
                **_tsquery_params(query, expansions),
                "ql": ql,
                "qc_ok": qc_ok,
                "qc_like": f"%{_escape_like(qc)}%",
                "pool": settings.candidate_pool,
            },
        )
    ).mappings()

    hits: list[ClientHit] = []
    for row in rows:
        scores = _client_match_scores(row, settings.client_fuzzy_threshold)
        if scores:
            matched_by = sorted(scores, key=lambda m: -scores[m])
            hits.append(ClientHit(row["id"], round(max(scores.values()), 4), matched_by))

    hits.sort(key=lambda h: -h.score)
    return hits


# --- Documents ---------------------------------------------------------------------------------


def _keyword_sql(n_expansions: int, client_filter: bool) -> str:
    where_client = "AND d.client_id = :client_id" if client_filter else ""
    return f"""
        SELECT d.id,
               ts_headline('english', d.content, t.tsq,
                           'MaxFragments=1, MaxWords=40, MinWords=15, StartSel=**, StopSel=**')
                   AS headline
        FROM documents d, (SELECT {_tsquery_sql(n_expansions)} AS tsq) t
        WHERE d.search_tsv @@ t.tsq {where_client}
        ORDER BY ts_rank_cd(d.search_tsv, t.tsq, 32) DESC, d.created_at DESC
        LIMIT :pool
    """


async def _keyword_candidates(
    session: AsyncSession,
    query: str,
    expansions: Sequence[str],
    client_id: uuid.UUID | None,
    pool: int,
) -> list[tuple[uuid.UUID, str]]:
    params = {**_tsquery_params(query, expansions), "pool": pool}
    if client_id is not None:
        params["client_id"] = client_id
    rows = await session.execute(text(_keyword_sql(len(expansions), client_id is not None)), params)
    return [(row.id, row.headline) for row in rows]


async def _semantic_candidates(
    session: AsyncSession,
    vectors: Sequence[list[float]],
    weights: Sequence[float],
    client_id: uuid.UUID | None,
    settings: Settings,
) -> list[tuple[uuid.UUID, float, str]]:
    """Best weighted chunk similarity per document, over the query and its expansions."""
    # HNSW returns at most ef_search rows per scan; make sure it covers the candidate pool.
    await session.execute(
        text("SELECT set_config('hnsw.ef_search', :v, true)"),
        {"v": str(max(40, settings.candidate_pool * 2))},
    )

    chunk = models.DocumentChunk
    chunks: FromClause = (
        chunk.__table__
        if client_id is None
        # Exact search within one client. An HNSW scan filtered by client only sees the
        # ef_search chunks nearest overall, so a client whose chunks are not among them would
        # get no semantic results at all. A client has few chunks, so exact search is cheap;
        # MATERIALIZED keeps the planner from turning it back into a filtered index scan.
        else select(chunk.document_id, chunk.content, chunk.embedding)
        .join(models.Document)
        .where(models.Document.client_id == client_id)
        .cte("client_chunks")
        .prefix_with("MATERIALIZED")
    )

    best: dict[uuid.UUID, tuple[float, str]] = {}
    for vector, weight in zip(vectors, weights, strict=True):
        distance = chunks.c.embedding.cosine_distance(vector)
        stmt = (
            select(chunks.c.document_id, chunks.c.content, distance.label("distance"))
            .order_by(distance)
            .limit(settings.candidate_pool)
        )
        for document_id, content, dist in await session.execute(stmt):
            similarity = weight * (1.0 - float(dist))
            if (
                similarity >= settings.min_semantic_similarity
                and similarity > best.get(document_id, (0.0, ""))[0]
            ):
                best[document_id] = (similarity, content)

    ranked = sorted(best.items(), key=lambda item: -item[1][0])
    return [(doc_id, sim, content) for doc_id, (sim, content) in ranked]


def _truncate(value: str, limit: int = SNIPPET_CHARS) -> str:
    if len(value) <= limit:
        return value
    cut = value[:limit].rsplit(" ", 1)[0]
    return f"{cut}…"


def fuse(
    keyword: Sequence[tuple[uuid.UUID, str]],
    semantic: Sequence[tuple[uuid.UUID, float, str]],
    rrf_k: int,
) -> list[DocumentHit]:
    """Reciprocal Rank Fusion of the two ranked lists.

    RRF only looks at ranks, so the incomparable scales of ts_rank and cosine similarity do not
    matter. The score is normalized so that a document ranked first by both retrievers gets 1.0
    and one ranked first by a single retriever gets 0.5.
    """
    candidates: dict[uuid.UUID, _Candidate] = {}
    for rank, (doc_id, headline) in enumerate(keyword, start=1):
        c = candidates.setdefault(doc_id, _Candidate())
        c.keyword_rank, c.headline = rank, headline
        c.matched_by.append("keyword")
    for rank, (doc_id, similarity, chunk_text) in enumerate(semantic, start=1):
        c = candidates.setdefault(doc_id, _Candidate())
        c.semantic_rank, c.similarity, c.best_chunk = rank, similarity, chunk_text
        c.matched_by.append("semantic")

    max_score = 2 / (rrf_k + 1)
    hits = []
    for doc_id, c in candidates.items():
        raw = sum(1 / (rrf_k + r) for r in (c.keyword_rank, c.semantic_rank) if r is not None)
        snippet = c.headline if c.headline else _truncate(c.best_chunk or "")
        hits.append(DocumentHit(doc_id, round(raw / max_score, 4), c.matched_by, snippet))
    hits.sort(key=lambda h: -h.score)
    return hits


async def search_documents(
    session: AsyncSession,
    query: str,
    expansions: Sequence[str],
    vectors: Sequence[list[float]],
    settings: Settings,
    client_id: uuid.UUID | None = None,
) -> list[DocumentHit]:
    """`vectors` are the embeddings of [query, *expansions], computed beforehand."""
    weights = [1.0] + [settings.expansion_weight] * len(expansions)
    keyword = await _keyword_candidates(
        session, query, expansions, client_id, settings.candidate_pool
    )
    semantic = await _semantic_candidates(session, vectors, weights, client_id, settings)
    return fuse(keyword, semantic, settings.rrf_k)


# --- LLM query expansion -------------------------------------------------------------------------


def should_ask_llm(
    query: str, thesaurus_expansions: Sequence[str], client_hits: Sequence[ClientHit]
) -> bool:
    """Whether an LLM call is likely to improve document results for this query.

    Each call costs about $0.0005 and 1-2 s of latency, so it is skipped when the query
    - is covered by the domain thesaurus already;
    - names a client directly (names, emails, domains have no meaningful related terms);
    - looks like an email address.
    """
    if thesaurus_expansions or "@" in query:
        return False
    return not any(hit.score >= DIRECT_CLIENT_MATCH_SCORE for hit in client_hits)


async def llm_expansions(session: AsyncSession, llm: LLMClient, query: str) -> list[str]:
    """LLM expansions for `query`, cached in the database across workers and restarts.

    Failures are not cached, so a transient outage does not stick.
    """
    key = normalize_query(query)
    cached = await session.scalar(
        select(models.QueryExpansion.phrases).where(
            models.QueryExpansion.query_key == key, models.QueryExpansion.model == llm.model
        )
    )
    await release_connection(session)  # the LLM call can take seconds
    if cached is not None:
        return list(cached)

    phrases = await llm.expand_query(query)
    if phrases is None:
        return []
    await session.execute(
        insert(models.QueryExpansion)
        .values(query_key=key, model=llm.model, phrases=phrases)
        .on_conflict_do_nothing()
    )
    await session.commit()
    return phrases


# --- Combined ----------------------------------------------------------------------------------


@dataclass
class SearchOutcome:
    results: list[schemas.SearchResult]
    expansions: list[str]


async def search(
    session: AsyncSession,
    embedder: Embedder,
    settings: Settings,
    query: str,
    types: set[SearchType],
    limit: int,
    client_id: uuid.UUID | None = None,
    llm: LLMClient | None = None,
) -> SearchOutcome:
    """Run a search in phases so that no database connection is held during slow work.

    1. Clients (one short transaction). Also run for document-only searches, because a direct
       client match decides whether the LLM is worth asking.
    2. No connection held: LLM expansion (if useful, cached) and query embeddings.
    3. Documents (one short transaction), then loading the result rows.
    """
    expansions = expand_query(query)[:MAX_EXPANSIONS]

    client_hits = (
        await search_clients(session, query, expansions, settings) if client_id is None else []
    )
    await release_connection(session)

    document_hits: list[DocumentHit] = []
    if "document" in types:
        if (
            llm is not None
            and settings.llm_query_expansion
            and should_ask_llm(query, expansions, client_hits)
        ):
            known = {normalize_query(e) for e in expansions} | {normalize_query(query)}
            extra = await llm_expansions(session, llm, query)
            expansions += [p for p in extra if normalize_query(p) not in known]
            expansions = expansions[:MAX_EXPANSIONS]
        vectors = await embed_queries(embedder, [query, *expansions])
        document_hits = await search_documents(
            session, query, expansions, vectors, settings, client_id
        )
    if "client" not in types:
        client_hits = []

    # Clients the query names directly come first; everything else is merged by score, with
    # clients first on ties (an entity match is the more specific answer). Only the top `limit`
    # results are loaded from the database.
    def sort_key(hit: ClientHit | DocumentHit) -> tuple[int, float, int]:
        is_client = isinstance(hit, ClientHit)
        direct = is_client and hit.score >= DIRECT_CLIENT_MATCH_SCORE
        return (0 if direct else 1, -hit.score, 0 if is_client else 1)

    hits: list[ClientHit | DocumentHit] = [*client_hits, *document_hits]
    hits.sort(key=sort_key)
    merged = hits[:limit]

    client_ids = [h.client_id for h in merged if isinstance(h, ClientHit)]
    document_ids = [h.document_id for h in merged if isinstance(h, DocumentHit)]
    clients: dict[uuid.UUID, models.Client] = {}
    documents: dict[uuid.UUID, models.Document] = {}
    if client_ids:
        stmt = select(models.Client).where(models.Client.id.in_(client_ids))
        clients = {c.id: c for c in await session.scalars(stmt)}
    if document_ids:
        doc_stmt = select(models.Document).where(models.Document.id.in_(document_ids))
        documents = {d.id: d for d in await session.scalars(doc_stmt)}

    results: list[schemas.SearchResult] = []
    for hit in merged:
        if isinstance(hit, ClientHit):
            results.append(
                schemas.ClientSearchResult(
                    score=hit.score,
                    matched_by=hit.matched_by,
                    client=schemas.Client.model_validate(clients[hit.client_id]),
                )
            )
        else:
            results.append(
                schemas.DocumentSearchResult(
                    score=hit.score,
                    matched_by=hit.matched_by,
                    snippet=hit.snippet,
                    document=schemas.Document.model_validate(documents[hit.document_id]),
                )
            )
    return SearchOutcome(results, expansions)
