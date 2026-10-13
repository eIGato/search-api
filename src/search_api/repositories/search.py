"""Search queries: lexical client matching and the two document retrievers.

Each method expects to run inside a transaction: the per-query planner settings are set with
`set_config(..., is_local => true)` and apply to that transaction only.
"""

import uuid
from collections.abc import Sequence

from sqlalchemy import FromClause, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from search_api import models
from search_api.ranking import (
    DESCRIPTION_MATCH_SCORE,
    DESCRIPTION_RELATED_SCORE,
    EXACT_MATCH_SCORE,
    FUZZY_MATCH_WEIGHT,
    NAME_PART_MATCH_SCORE,
    SUBSTRING_MATCH_SCORE,
    ClientHit,
    KeywordCandidate,
    SemanticCandidate,
    client_hit,
    compact,
)


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _tsquery_sql(n_expansions: int) -> str:
    """The query (web-search syntax) OR'ed with each expansion phrase."""
    return " || ".join(
        ["websearch_to_tsquery('english', :q)"]
        + [f"phraseto_tsquery('english', :e{i})" for i in range(n_expansions)]
    )


def _tsquery_params(query: str, expansions: Sequence[str]) -> dict[str, object]:
    return {"q": query, **{f"e{i}": e for i, e in enumerate(expansions)}}


def _client_sql(n_expansions: int) -> str:
    """Best score per field (0 = no match) for the strongest `:pool` candidate clients.

    Scored in SQL so that the candidate limit drops the weakest matches, not arbitrary rows:
    "smith" must keep the client named Smith even when many "Smithson"s match too.
    Each CASE takes the first matching tier, which is the best one because tiers are listed in
    descending score order (fuzzy scores are at most FUZZY_MATCH_WEIGHT).
    """
    return f"""
        WITH signals AS (
            SELECT c.id,
                   lower(c.email) = :ql AS email_exact,
                   :ql IN (lower(c.first_name || ' ' || c.last_name),
                           lower(c.last_name || ' ' || c.first_name)) AS name_exact,
                   :ql IN (lower(c.first_name), lower(c.last_name)) AS name_part_exact,
                   (:qc_ok AND split_part(c.compact_text, '|', 1) LIKE :qc_like)
                       AS name_substring,
                   (:qc_ok AND split_part(c.compact_text, '|', 2) LIKE :qc_like)
                       AS email_substring,
                   strict_word_similarity(:ql, lower(c.first_name || ' ' || c.last_name))
                       AS name_similarity,
                   strict_word_similarity(:ql, lower(c.email)) AS email_similarity,
                   c.description_tsv @@ websearch_to_tsquery('english', :q) AS description_match,
                   c.description_tsv @@ t.tsq AS description_related
            FROM clients c, (SELECT {_tsquery_sql(n_expansions)} AS tsq) t
            WHERE (:qc_ok AND c.compact_text LIKE :qc_like)
               OR :ql <<% c.name_email_text
               OR lower(c.first_name) = :ql
               OR lower(c.last_name) = :ql
               OR c.description_tsv @@ t.tsq
        ),
        scored AS (
            SELECT id,
                   CASE WHEN name_exact THEN CAST(:exact AS float8)
                        WHEN name_part_exact THEN CAST(:name_part AS float8)
                        WHEN name_substring THEN CAST(:substring AS float8)
                        WHEN name_similarity >= CAST(:fuzzy_threshold AS float8)
                            THEN CAST(:fuzzy_weight AS float8) * name_similarity
                        ELSE 0 END AS name_score,
                   CASE WHEN email_exact THEN CAST(:exact AS float8)
                        WHEN email_substring THEN CAST(:substring AS float8)
                        WHEN email_similarity >= CAST(:fuzzy_threshold AS float8)
                            THEN CAST(:fuzzy_weight AS float8) * email_similarity
                        ELSE 0 END AS email_score,
                   CASE WHEN description_match THEN CAST(:description AS float8)
                        WHEN description_related THEN CAST(:description_related AS float8)
                        ELSE 0 END AS description_score
            FROM signals
        )
        SELECT id, name_score, email_score, description_score
        FROM scored
        WHERE greatest(name_score, email_score, description_score) > 0
        ORDER BY greatest(name_score, email_score, description_score) DESC, id
        LIMIT :pool
    """


_CLIENT_SCORE_PARAMS = {
    "exact": EXACT_MATCH_SCORE,
    "name_part": NAME_PART_MATCH_SCORE,
    "substring": SUBSTRING_MATCH_SCORE,
    "fuzzy_weight": FUZZY_MATCH_WEIGHT,
    "description": DESCRIPTION_MATCH_SCORE,
    "description_related": DESCRIPTION_RELATED_SCORE,
}


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


class SearchRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def search_clients(
        self, query: str, expansions: Sequence[str], fuzzy_threshold: float, pool: int
    ) -> list[ClientHit]:
        """The strongest `pool` client matches, best first."""
        ql = " ".join(query.lower().split())
        qc = compact(query)
        # Substring matching on 1-2 characters would match almost everyone.
        qc_ok = len(qc) >= 3

        await self._session.execute(
            text("SELECT set_config('pg_trgm.strict_word_similarity_threshold', :t, true)"),
            {"t": str(fuzzy_threshold)},
        )
        rows = (
            await self._session.execute(
                text(_client_sql(len(expansions))),
                {
                    **_tsquery_params(query, expansions),
                    "ql": ql,
                    "qc_ok": qc_ok,
                    "qc_like": f"%{_escape_like(qc)}%",
                    **_CLIENT_SCORE_PARAMS,
                    "fuzzy_threshold": fuzzy_threshold,
                    "pool": pool,
                },
            )
        ).mappings()
        return [
            client_hit(
                row["id"],
                {
                    "name": row["name_score"],
                    "email": row["email_score"],
                    "description": row["description_score"],
                },
            )
            for row in rows
        ]

    async def keyword_candidates(
        self,
        query: str,
        expansions: Sequence[str],
        client_id: uuid.UUID | None,
        pool: int,
    ) -> list[KeywordCandidate]:
        """Full-text matches over title + content, best first, with a highlighted headline."""
        params = {**_tsquery_params(query, expansions), "pool": pool}
        if client_id is not None:
            params["client_id"] = client_id
        rows = await self._session.execute(
            text(_keyword_sql(len(expansions), client_id is not None)), params
        )
        return [KeywordCandidate(row.id, row.headline) for row in rows]

    async def semantic_candidates(
        self,
        vectors: Sequence[list[float]],
        weights: Sequence[float],
        client_id: uuid.UUID | None,
        pool: int,
        min_similarity: float,
    ) -> list[SemanticCandidate]:
        """Best weighted chunk similarity per document over all vectors, best first."""
        # HNSW returns at most ef_search rows per scan; make sure it covers the candidate pool.
        await self._session.execute(
            text("SELECT set_config('hnsw.ef_search', :v, true)"), {"v": str(max(40, pool * 2))}
        )

        chunk = models.DocumentChunk
        chunks: FromClause = (
            chunk.__table__
            if client_id is None
            # Exact search within one client. An HNSW scan filtered by client only sees the
            # ef_search chunks nearest overall, so a client whose chunks are not among them
            # would get no semantic results at all. A client has few chunks, so exact search is
            # cheap; MATERIALIZED keeps the planner from turning it back into a filtered index
            # scan.
            else select(chunk.document_id, chunk.content, chunk.embedding)
            .join(models.Document)
            .where(models.Document.client_id == client_id)
            .cte("client_chunks")
            .prefix_with("MATERIALIZED")
        )

        best: dict[uuid.UUID, SemanticCandidate] = {}
        for vector, weight in zip(vectors, weights, strict=True):
            distance = chunks.c.embedding.cosine_distance(vector)
            stmt = (
                select(chunks.c.document_id, chunks.c.content, distance.label("distance"))
                .order_by(distance)
                .limit(pool)
            )
            for document_id, content, dist in await self._session.execute(stmt):
                similarity = weight * (1.0 - float(dist))
                current = best.get(document_id)
                if similarity >= min_similarity and (
                    current is None or similarity > current.similarity
                ):
                    best[document_id] = SemanticCandidate(document_id, similarity, content)

        return sorted(best.values(), key=lambda c: -c.similarity)
