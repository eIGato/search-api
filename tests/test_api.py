"""API tests against a real Postgres (pgvector + pg_trgm) with a deterministic fake embedder."""

import asyncio
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import NullPool, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from search_api.config import Settings
from search_api.embeddings import Embedder
from search_api.main import create_app
from search_api.search import _semantic_candidates
from tests.conftest import Json, create_client, create_document

# Fixtures: test data by short name.
People = dict[str, Json]
Library = dict[str, Json]

# --- clients -----------------------------------------------------------------------------------


def test_create_client(client: TestClient) -> None:
    response = client.post(
        "/clients",
        json={
            "first_name": "  John ",
            "last_name": "Doe",
            "email": "john.doe@neviswealth.com",
            "social_links": ["https://www.linkedin.com/in/johndoe"],
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert uuid.UUID(body["id"])
    assert body["first_name"] == "John"  # whitespace stripped
    assert body["description"] is None
    assert body["social_links"] == ["https://www.linkedin.com/in/johndoe"]
    assert response.headers["Location"].endswith(f"/clients/{body['id']}")
    assert client.get(f"/clients/{body['id']}").json() == body


@pytest.mark.parametrize(
    "payload",
    [
        {"last_name": "Doe", "email": "a@b.com"},  # missing first_name
        {"first_name": "John", "last_name": "Doe"},  # missing email
        {"first_name": "John", "last_name": "Doe", "email": "not-an-email"},
        {"first_name": "   ", "last_name": "Doe", "email": "a@b.com"},  # blank after strip
        {"first_name": "John", "last_name": "Doe", "email": "a@b.com", "social_links": ["nope"]},
        {"first_name": "John", "last_name": "Doe", "email": "a@b.com", "social_links": "x"},
    ],
)
def test_create_client_validation(client: TestClient, payload: dict[str, object]) -> None:
    assert client.post("/clients", json=payload).status_code == 422


def test_duplicate_email_is_rejected_case_insensitively(client: TestClient) -> None:
    create_client(client, email="john.doe@neviswealth.com")
    response = client.post(
        "/clients",
        json={"first_name": "Johnny", "last_name": "D", "email": "John.Doe@NevisWealth.com"},
    )
    assert response.status_code == 409
    assert "already exists" in response.json()["detail"]


@pytest.mark.parametrize("client_id", [str(uuid.uuid4()), "not-a-uuid"])
def test_get_unknown_client(client: TestClient, client_id: str) -> None:
    response = client.get(f"/clients/{client_id}")
    assert response.status_code == 404
    assert response.json() == {"detail": "Client not found"}


# --- documents ---------------------------------------------------------------------------------


def test_create_and_get_document(client: TestClient) -> None:
    owner = create_client(client)
    response = client.post(
        f"/clients/{owner['id']}/documents",
        json={"title": "Utility bill", "content": "Electricity bill for March."},
    )

    assert response.status_code == 201
    document = response.json()
    assert document["client_id"] == owner["id"]
    assert document["title"] == "Utility bill"
    assert document["content"] == "Electricity bill for March."
    assert document["created_at"]
    assert response.headers["Location"].endswith(
        f"/clients/{owner['id']}/documents/{document['id']}"
    )
    assert client.get(f"/clients/{owner['id']}/documents/{document['id']}").json() == document
    assert client.get(f"/clients/{owner['id']}/documents").json() == [document]


def test_create_document_for_unknown_client(client: TestClient) -> None:
    response = client.post(
        f"/clients/{uuid.uuid4()}/documents", json={"title": "t", "content": "c"}
    )
    assert response.status_code == 404


@pytest.mark.parametrize(
    "payload",
    [
        {"title": "t"},
        {"content": "c"},
        {"title": "", "content": "c"},
        {"title": "t", "content": " "},
    ],
)
def test_create_document_validation(client: TestClient, payload: dict[str, str]) -> None:
    owner = create_client(client)
    assert client.post(f"/clients/{owner['id']}/documents", json=payload).status_code == 422


def test_document_is_not_reachable_through_another_client(client: TestClient) -> None:
    owner = create_client(client)
    other = create_client(client, email="other@example.com")
    document = create_document(client, owner["id"], "Passport", "Passport copy")

    response = client.get(f"/clients/{other['id']}/documents/{document['id']}")
    assert response.status_code == 404


def test_long_document_is_chunked_and_searchable_by_its_end(client: TestClient) -> None:
    owner = create_client(client)
    filler = " ".join(f"Paragraph {i} talks about general market conditions." for i in range(100))
    create_document(client, owner["id"], "Annual letter", f"{filler} Finally, zygomorphic.")

    results = client.get("/search", params={"q": "zygomorphic", "type": "document"}).json()
    assert [r["document"]["title"] for r in results] == ["Annual letter"]
    assert "zygomorphic" in results[0]["snippet"]


# --- summaries ---------------------------------------------------------------------------------


class FakeLLM:
    """Stands in for LLMClient: counts calls, can be slow, can fail (returns None)."""

    model = "fake-model"
    max_call_seconds = 2.0

    def __init__(
        self,
        summary: str | None,
        expansions: dict[str, list[str] | None] | None = None,
        delay: float = 0.0,
    ) -> None:
        self.summary = summary
        self.expansions = expansions or {"special": ["llm suggested phrase"]}
        self.delay = delay
        self.calls = 0
        self.expansion_calls: list[str] = []

    async def summarize(self, title: str, content: str) -> str | None:
        self.calls += 1
        await asyncio.sleep(self.delay)
        return self.summary

    async def expand_query(self, query: str) -> list[str] | None:
        self.expansion_calls.append(query)
        await asyncio.sleep(self.delay)
        return self.expansions.get(query, [])


def _summary_url(client: TestClient) -> str:
    owner = create_client(client)
    content = (
        "The client holds a balanced portfolio. Equities returned five percent. "
        "Bonds returned two percent. The office was repainted. "
        "The portfolio will be rebalanced in January."
    )
    document = create_document(client, owner["id"], "Q3 review", content)
    return f"/clients/{owner['id']}/documents/{document['id']}/summary"


def test_extractive_summary_without_llm(client: TestClient) -> None:
    url = _summary_url(client)
    response = client.get(url)

    assert response.status_code == 200
    body = response.json()
    assert body["method"] == "extractive"
    assert 0 < len(body["summary"]) < 200
    assert client.get(url).json() == body  # cached


def test_llm_summary_is_used_and_cached(client: TestClient) -> None:
    url = _summary_url(client)
    llm = FakeLLM("A Q3 portfolio review.")
    client.app.state.llm = llm  # type: ignore[attr-defined]

    assert client.get(url).json()["summary"] == "A Q3 portfolio review."
    assert client.get(url).json()["method"] == "llm"
    assert llm.calls == 1


def test_llm_failure_falls_back_to_extractive_and_retries_later(client: TestClient) -> None:
    url = _summary_url(client)
    llm = FakeLLM(None)
    client.app.state.llm = llm  # type: ignore[attr-defined]

    assert client.get(url).json()["method"] == "extractive"
    llm.summary = "Recovered."
    assert client.get(url).json() == {
        "document_id": url.split("/")[-2],
        "summary": "Recovered.",
        "method": "llm",
    }


def test_concurrent_summary_requests_call_the_llm_once(client: TestClient) -> None:
    url = _summary_url(client)
    llm = FakeLLM("One summary.", delay=0.5)
    client.app.state.llm = llm  # type: ignore[attr-defined]

    with ThreadPoolExecutor(max_workers=5) as pool:
        responses = list(pool.map(lambda _: client.get(url), range(5)))

    assert [r.status_code for r in responses] == [200] * 5
    assert {r.json()["summary"] for r in responses} == {"One summary."}
    assert llm.calls == 1


def test_summary_of_unknown_document(client: TestClient) -> None:
    owner = create_client(client)
    response = client.get(f"/clients/{owner['id']}/documents/{uuid.uuid4()}/summary")
    assert response.status_code == 404


# --- search: clients ---------------------------------------------------------------------------


@pytest.fixture
def people(client: TestClient) -> People:
    return {
        "john": create_client(client),
        "jane": create_client(
            client,
            first_name="Jane",
            last_name="Smith",
            email="jane.smith@gmail.com",
            description="Retired surgeon with a conservative risk profile",
        ),
        "li": create_client(
            client, first_name="Wei", last_name="Li", email="wli@example.org", description=None
        ),
    }


def search(client: TestClient, q: str, **params: Any) -> list[Json]:
    response = client.get("/search", params={"q": q, **params})
    assert response.status_code == 200, response.text
    results: list[Json] = response.json()
    return results


def emails(results: list[Json]) -> list[object]:
    return [r["client"]["email"] for r in results if r["type"] == "client"]


def test_search_finds_client_by_email_domain(client: TestClient, people: People) -> None:
    # The example from the assignment.
    results = search(client, "NevisWealth")
    assert emails(results) == ["john.doe@neviswealth.com"]
    assert results[0]["matched_by"] == ["email"]
    assert results[0]["score"] == pytest.approx(0.9)


@pytest.mark.parametrize(
    ("query", "expected_email", "expected_score"),
    [
        ("john.doe@neviswealth.com", "john.doe@neviswealth.com", 1.0),  # exact email
        ("JOHN DOE", "john.doe@neviswealth.com", 1.0),  # exact full name, any case
        ("Doe John", "john.doe@neviswealth.com", 1.0),  # reversed full name
        ("Li", "wli@example.org", 0.95),  # short exact last name
        ("nevis wealth", "john.doe@neviswealth.com", 0.9),  # spacing differs
        ("smit", "jane.smith@gmail.com", 0.9),  # prefix
    ],
)
def test_search_client_exact_and_substring(
    client: TestClient, people: People, query: str, expected_email: str, expected_score: float
) -> None:
    results = search(client, query, type="client")
    assert emails(results)[0] == expected_email
    assert results[0]["score"] == pytest.approx(expected_score)


def test_search_client_tolerates_typos(client: TestClient, people: People) -> None:
    results = search(client, "Jane Smiht", type="client")
    assert emails(results) == ["jane.smith@gmail.com"]
    assert 0.3 < results[0]["score"] < 0.9


def test_search_client_by_description(client: TestClient, people: People) -> None:
    results = search(client, "surgeon", type="client")
    assert emails(results) == ["jane.smith@gmail.com"]
    assert results[0]["matched_by"] == ["description"]


def test_search_client_description_through_expansion(client: TestClient, people: People) -> None:
    # "risk appetite" is not in the description, "risk profile" is (thesaurus concept).
    results = search(client, "risk appetite", type="client")
    assert emails(results) == ["jane.smith@gmail.com"]


@pytest.mark.parametrize("query", ["%", "_", "%%%", "\\", "' OR 1=1 --"])
def test_search_special_characters_are_literal(
    client: TestClient, people: People, query: str
) -> None:
    assert search(client, query) == []


def test_search_short_query_does_not_match_everyone(client: TestClient, people: People) -> None:
    assert search(client, "e", type="client") == []


def test_candidate_pool_keeps_the_strongest_client_matches(
    client: TestClient, settings: Settings, embedder: Embedder
) -> None:
    # Created first, so that an unordered scan would meet the weaker matches first.
    for i in range(5):
        create_client(
            client, first_name="Anna", last_name=f"Smithson{i}", email=f"a{i}@example.org"
        )
    create_client(client, first_name="Tom", last_name="Smith", email="tom@example.org")

    small_pool = settings.model_copy(update={"candidate_pool": 3})
    with TestClient(create_app(small_pool, embedder)) as api:
        results = search(api, "smith", type="client")

    assert len(results) == 3
    assert emails(results)[0] == "tom@example.org"
    assert results[0]["score"] == pytest.approx(0.95)


# --- search: documents -------------------------------------------------------------------------


@pytest.fixture
def library(client: TestClient, people: People) -> Library:
    john, jane = people["john"]["id"], people["jane"]["id"]
    return {
        "bill": create_document(
            client,
            john,
            "Utility bill - March",
            "Electricity utility bill for 12 Baker Street, London. Amount due 84 GBP.",
        ),
        "passport": create_document(client, john, "Passport", "Copy of British passport."),
        "lease": create_document(
            client, jane, "Tenancy agreement", "Residential tenancy agreement for a flat in Leeds."
        ),
    }


def titles(results: list[Json]) -> list[object]:
    return [r["document"]["title"] for r in results if r["type"] == "document"]


def test_search_documents_via_concept_expansion(client: TestClient, library: Library) -> None:
    # The example from the assignment: "address proof" finds the utility bill.
    response = client.get("/search", params={"q": "address proof"})
    found = titles(response.json())

    assert set(found[:2]) == {"Utility bill - March", "Tenancy agreement"}
    assert "utility bill" in json.loads(response.headers["X-Search-Expansions"])


def test_search_document_result_shape(client: TestClient, library: Library) -> None:
    [result] = search(client, "electricity", type="document")
    assert result["type"] == "document"
    assert result["document"] == library["bill"]
    assert "keyword" in result["matched_by"]
    assert "**Electricity**" in result["snippet"]
    assert 0 < result["score"] <= 1


def test_search_type_filter(client: TestClient, library: Library) -> None:
    assert {r["type"] for r in search(client, "john", type="client")} == {"client"}
    assert {r["type"] for r in search(client, "passport", type="document")} == {"document"}


def test_search_within_one_clients_documents(
    client: TestClient, people: People, library: Library
) -> None:
    results = search(client, "address proof", client_id=people["jane"]["id"])
    assert titles(results) == ["Tenancy agreement"]
    assert all(r["type"] == "document" for r in results)


def test_semantic_search_within_one_client_is_exhaustive(
    client: TestClient, settings: Settings, embedder: Embedder, database_url: str
) -> None:
    """A client's documents are found even when other clients' documents are all closer.

    Filtering an approximate (HNSW) scan by client would only see the ef_search chunks nearest
    overall. The planner picks that plan only on large tables, so sorting is disabled here to
    force it wherever it is possible.
    """
    others = create_client(client, email="others@example.org")
    target = create_client(client, first_name="Tom", last_name="Smith", email="tom@example.org")
    for i in range(60):
        create_document(client, others["id"], "Investigation", f"Investigation {i}.")
    memo = create_document(client, target["id"], "Memo", "Investment notes.")

    async def run() -> list[tuple[uuid.UUID, float, str]]:
        engine = create_async_engine(database_url, poolclass=NullPool)
        try:
            async with AsyncSession(engine) as session:
                await session.execute(text("SET LOCAL enable_sort = off"))
                return await _semantic_candidates(
                    session,
                    embedder.embed_queries(["investigation"]),
                    [1.0],
                    uuid.UUID(target["id"]),
                    settings.model_copy(update={"candidate_pool": 10}),
                )
        finally:
            await engine.dispose()

    assert [str(doc_id) for doc_id, _, _ in asyncio.run(run())] == [memo["id"]]


def test_client_named_in_query_ranks_above_documents_mentioning_them(
    client: TestClient, people: People
) -> None:
    create_document(client, people["jane"]["id"], "Note", "Call Jane Smith about Jane Smith's IRA.")
    results = search(client, "Jane Smith")
    assert [r["type"] for r in results][:2] == ["client", "document"]


def test_search_limit(client: TestClient, people: People) -> None:
    for i in range(5):
        create_document(client, people["john"]["id"], f"Statement {i}", "Monthly bank statement.")
    assert len(search(client, "bank statement", limit=3)) == 3


def test_llm_expansions_are_used_when_configured(client: TestClient, people: People) -> None:
    create_document(client, people["john"]["id"], "Memo", "An llm suggested phrase appears here.")
    client.app.state.llm = FakeLLM(None)  # type: ignore[attr-defined]

    response = client.get("/search", params={"q": "special", "type": "document"})
    assert titles(response.json()) == ["Memo"]
    assert json.loads(response.headers["X-Search-Expansions"]) == ["llm suggested phrase"]


@pytest.mark.parametrize(
    ("query", "params"),
    [
        ("address proof", {}),  # covered by the thesaurus
        ("Jane Smith", {}),  # names a client directly
        ("Jane Smith", {"type": "document"}),  # ... even when only documents are requested
        ("someone@example.com", {}),  # an email
        ("special", {"type": "client"}),  # no document search, nothing to expand for
    ],
)
def test_llm_is_not_asked_when_it_cannot_help(
    client: TestClient, people: People, query: str, params: dict[str, str]
) -> None:
    llm = FakeLLM(None)
    client.app.state.llm = llm  # type: ignore[attr-defined]
    assert client.get("/search", params={"q": query, **params}).status_code == 200
    assert llm.expansion_calls == []


def test_llm_expansions_are_cached_in_the_database(
    client: TestClient, settings: Settings, embedder: Embedder, people: People
) -> None:
    llm = FakeLLM(None, expansions={"special": ["llm suggested phrase"], "flaky": None})
    client.app.state.llm = llm  # type: ignore[attr-defined]

    search(client, "special")
    search(client, "Special ")  # same cache key
    # A restarted (or another) worker shares the cache.
    with TestClient(create_app(settings, embedder)) as other_worker:
        other_worker.app.state.llm = llm  # type: ignore[attr-defined]
        response = other_worker.get("/search", params={"q": "special"})
        assert json.loads(response.headers["X-Search-Expansions"]) == ["llm suggested phrase"]
    assert llm.expansion_calls == ["special"]

    # Failures are not cached.
    search(client, "flaky")
    search(client, "flaky")
    assert llm.expansion_calls == ["special", "flaky", "flaky"]


def test_no_connection_is_held_during_slow_work(
    settings: Settings, embedder: Embedder, client: TestClient, people: People
) -> None:
    """With a single pooled connection, requests that each wait on a slow LLM still all succeed:
    if the connection were held during the LLM call they would queue behind each other and run
    into the pool timeout."""
    one_connection = settings.model_copy(
        update={"db_pool_size": 1, "db_max_overflow": 0, "db_pool_timeout_seconds": 1.0}
    )
    document = create_document(client, people["john"]["id"], "Memo", "Quarterly review notes.")
    summary_url = f"/clients/{people['john']['id']}/documents/{document['id']}/summary"
    llm = FakeLLM("Summary.", delay=0.6)

    with TestClient(create_app(one_connection, embedder)) as api:
        api.app.state.llm = llm  # type: ignore[attr-defined]
        requests = [("/search", {"q": f"uncovered query {i}"}) for i in range(4)]
        requests.append((summary_url, {}))
        with ThreadPoolExecutor(max_workers=len(requests)) as pool:
            responses = list(pool.map(lambda r: api.get(r[0], params=r[1]), requests))

    assert [r.status_code for r in responses] == [200] * len(requests)
    assert len(llm.expansion_calls) == 4


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"q": ""},
        {"q": "   "},
        {"q": "x" * 501},
        {"q": "a", "limit": 0},
        {"q": "a", "limit": 101},
        {"q": "a", "type": "invoice"},
        {"q": "a", "client_id": "nope"},
        {"q": "a", "client_id": str(uuid.uuid4()), "type": "client"},
    ],
)
def test_search_validation(client: TestClient, params: dict[str, str | int]) -> None:
    assert client.get("/search", params=params).status_code == 422


def test_search_with_no_data(client: TestClient) -> None:
    assert search(client, "anything") == []


# --- auth & health -----------------------------------------------------------------------------


def test_api_key_is_enforced_when_configured(
    settings: Settings, embedder: Embedder, client: TestClient
) -> None:
    secured = settings.model_copy(update={"api_key": SecretStr("s3cret")})
    with TestClient(create_app(secured, embedder)) as api:
        assert api.get("/search", params={"q": "x"}).status_code == 401
        assert (
            api.get("/search", params={"q": "x"}, headers={"X-API-Key": "wrong"}).status_code == 401
        )
        assert (
            api.get("/search", params={"q": "x"}, headers={"X-API-Key": "s3cret"}).status_code
            == 200
        )
        assert api.get("/health").status_code == 200  # health stays open for probes


def test_openapi_documents_endpoints(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/clients", "/clients/{client_id}/documents", "/search"} <= set(paths)
