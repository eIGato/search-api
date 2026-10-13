"""Search ranking rules: client score tiers, rank fusion for documents, merging both types.

Pure functions without I/O. The client tiers are applied in SQL (see
repositories.search.SearchRepository.search_clients) so that the candidate limit keeps the
strongest matches, but the values and their order are defined here.
"""

import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field

from search_api.domain import ClientMatch, DocumentMatch

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


@dataclass(frozen=True, slots=True)
class ClientHit:
    client_id: uuid.UUID
    score: float
    matched_by: list[ClientMatch]


@dataclass(frozen=True, slots=True)
class DocumentHit:
    document_id: uuid.UUID
    score: float
    matched_by: list[DocumentMatch]
    snippet: str


@dataclass(frozen=True, slots=True)
class KeywordCandidate:
    document_id: uuid.UUID
    headline: str


@dataclass(frozen=True, slots=True)
class SemanticCandidate:
    document_id: uuid.UUID
    similarity: float
    chunk: str


@dataclass
class _Candidate:
    keyword_rank: int | None = None
    semantic_rank: int | None = None
    headline: str | None = None
    best_chunk: str | None = None
    matched_by: list[DocumentMatch] = field(default_factory=list)


def compact(value: str) -> str:
    """Lowercase alphanumerics only; mirrors clients.compact_text in the database."""
    return re.sub(r"[^a-z0-9]", "", value.lower())


def client_hit(client_id: uuid.UUID, scores: dict[ClientMatch, float]) -> ClientHit:
    """A client's hit from its per-field scores (0 = the field did not match)."""
    matched = {name: score for name, score in scores.items() if score > 0}
    matched_by = sorted(matched, key=lambda name: -matched[name])
    return ClientHit(client_id, round(max(matched.values()), 4), matched_by)


def names_client_directly(hits: Sequence[ClientHit]) -> bool:
    return any(hit.score >= DIRECT_CLIENT_MATCH_SCORE for hit in hits)


def _truncate(value: str, limit: int = SNIPPET_CHARS) -> str:
    if len(value) <= limit:
        return value
    cut = value[:limit].rsplit(" ", 1)[0]
    return f"{cut}…"


def fuse(
    keyword: Sequence[KeywordCandidate], semantic: Sequence[SemanticCandidate], rrf_k: int
) -> list[DocumentHit]:
    """Reciprocal Rank Fusion of the two ranked lists.

    RRF only looks at ranks, so the incomparable scales of ts_rank and cosine similarity do not
    matter. The score is normalized so that a document ranked first by both retrievers gets 1.0
    and one ranked first by a single retriever gets 0.5.
    """
    candidates: dict[uuid.UUID, _Candidate] = {}
    for rank, kc in enumerate(keyword, start=1):
        c = candidates.setdefault(kc.document_id, _Candidate())
        c.keyword_rank, c.headline = rank, kc.headline
        c.matched_by.append("keyword")
    for rank, sc in enumerate(semantic, start=1):
        c = candidates.setdefault(sc.document_id, _Candidate())
        c.semantic_rank, c.best_chunk = rank, sc.chunk
        c.matched_by.append("semantic")

    max_score = 2 / (rrf_k + 1)
    hits = []
    for doc_id, c in candidates.items():
        raw = sum(1 / (rrf_k + r) for r in (c.keyword_rank, c.semantic_rank) if r is not None)
        snippet = c.headline if c.headline else _truncate(c.best_chunk or "")
        hits.append(DocumentHit(doc_id, round(raw / max_score, 4), c.matched_by, snippet))
    hits.sort(key=lambda h: -h.score)
    return hits


def merge(
    client_hits: Sequence[ClientHit], document_hits: Sequence[DocumentHit], limit: int
) -> list[ClientHit | DocumentHit]:
    """The top `limit` hits of both types in one list.

    Clients the query names directly come first; everything else is merged by score, with
    clients first on ties (an entity match is the more specific answer).
    """

    def sort_key(hit: ClientHit | DocumentHit) -> tuple[int, float, int]:
        is_client = isinstance(hit, ClientHit)
        direct = is_client and hit.score >= DIRECT_CLIENT_MATCH_SCORE
        return (0 if direct else 1, -hit.score, 0 if is_client else 1)

    hits: list[ClientHit | DocumentHit] = [*client_hits, *document_hits]
    return sorted(hits, key=sort_key)[:limit]
