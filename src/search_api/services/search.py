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

import uuid

from search_api.config import Settings
from search_api.db import Database
from search_api.domain import ClientResult, DocumentResult, SearchResults, SearchType
from search_api.embeddings import Embedder, embed_queries
from search_api.errors import InvalidRequestError
from search_api.llm import LanguageModel, normalize_query
from search_api.ranking import ClientHit, DocumentHit, fuse, merge, names_client_directly
from search_api.thesaurus import expand_query

MAX_EXPANSIONS = 20


class SearchService:
    def __init__(
        self,
        database: Database,
        embedder: Embedder,
        llm: LanguageModel | None,
        settings: Settings,
    ) -> None:
        self._db = database
        self._embedder = embedder
        self._llm = llm if settings.llm_query_expansion else None
        self._settings = settings

    async def search(
        self,
        text: str,
        *,
        result_type: SearchType | None = None,
        client_id: uuid.UUID | None = None,
        limit: int,
    ) -> SearchResults:
        """Search in phases, so that no transaction is open during slow work:

        1. Expansions: the thesaurus, then the LLM if it is likely to help (one short read,
           the LLM call, one short cache write).
        2. Query embeddings, without a transaction.
        3. Ranking and loading the results, in one read-only transaction: results are ranked
           and loaded from the same snapshot, so every hit is found when it is loaded.
        """
        query = " ".join(text.split())
        types: set[SearchType] = {result_type} if result_type else {"client", "document"}
        if client_id is not None:
            if result_type == "client":
                raise InvalidRequestError(
                    "client_id restricts the search to documents; it cannot be combined with "
                    "type=client"
                )
            types = {"document"}

        expansions = expand_query(query)[:MAX_EXPANSIONS]
        vectors: list[list[float]] = []
        if "document" in types:
            expansions = await self._add_llm_expansions(query, expansions)
            vectors = await embed_queries(self._embedder, [query, *expansions])

        async with self._db.transaction(read_only=True) as uow:
            client_hits: list[ClientHit] = []
            if "client" in types:
                client_hits = await uow.search.search_clients(
                    query,
                    expansions,
                    self._settings.client_fuzzy_threshold,
                    self._settings.candidate_pool,
                )
            document_hits: list[DocumentHit] = []
            if "document" in types:
                weights = [1.0] + [self._settings.expansion_weight] * len(expansions)
                keyword = await uow.search.keyword_candidates(
                    query, expansions, client_id, self._settings.candidate_pool
                )
                semantic = await uow.search.semantic_candidates(
                    vectors,
                    weights,
                    client_id,
                    self._settings.candidate_pool,
                    self._settings.min_semantic_similarity,
                )
                document_hits = fuse(keyword, semantic, self._settings.rrf_k)

            top = merge(client_hits, document_hits, limit)
            clients = await uow.clients.get_many(
                [h.client_id for h in top if isinstance(h, ClientHit)]
            )
            documents = await uow.documents.get_many(
                [h.document_id for h in top if isinstance(h, DocumentHit)]
            )

        results: list[ClientResult | DocumentResult] = [
            ClientResult(hit.score, hit.matched_by, clients[hit.client_id])
            if isinstance(hit, ClientHit)
            else DocumentResult(hit.score, hit.matched_by, hit.snippet, documents[hit.document_id])
            for hit in top
        ]
        return SearchResults(results, expansions)

    async def _add_llm_expansions(self, query: str, expansions: list[str]) -> list[str]:
        """Add LLM expansions when they are likely to improve document results.

        Each LLM call costs about $0.0005 and 1-2 s of latency, so it is skipped when the query
        - is covered by the domain thesaurus already;
        - looks like an email address;
        - names a client directly (names, emails, domains have no meaningful related terms).
        Expansions are cached in the database across workers and restarts; failures are not
        cached, so a transient outage does not stick.
        """
        llm = self._llm
        if llm is None or expansions or "@" in query:
            return expansions

        key = normalize_query(query)
        async with self._db.transaction(read_only=True) as uow:
            client_hits = await uow.search.search_clients(
                query, [], self._settings.client_fuzzy_threshold, self._settings.candidate_pool
            )
            if names_client_directly(client_hits):
                return expansions
            phrases = await uow.query_expansions.get(key, llm.model)

        if phrases is None:
            phrases = await llm.expand_query(query)
            if phrases is None:
                return expansions
            # A cache entry, independent of the data searched: storing it in its own
            # transaction after the call cannot leave anything inconsistent.
            async with self._db.transaction() as uow:
                await uow.query_expansions.add(key, llm.model, phrases)

        known = {normalize_query(e) for e in expansions} | {key}
        new = [p for p in phrases if normalize_query(p) not in known]
        return [*expansions, *new][:MAX_EXPANSIONS]
