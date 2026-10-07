"""Search quality on a small realistic corpus with the real embedding model.

These are the "golden" queries the design is judged by. Deselect with `-m "not model"` if the
model cannot be downloaded.
"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from search_api.config import Settings
from search_api.embeddings import FastEmbedEmbedder
from search_api.main import create_app
from tests.conftest import _truncate, create_client, create_document

pytestmark = pytest.mark.model

CORPUS = {
    "john": [
        (
            "Utility bill - March 2026",
            "Electricity utility bill issued to John Doe, 12 Baker Street, London. "
            "Billing period March 2026. Amount due: 84.20 GBP.",
        ),
        (
            "Passport copy",
            "Scanned copy of passport. Nationality: British. Date of birth 01.02.1980. "
            "Passport number 123456789, expires 2031.",
        ),
        (
            "Investment policy statement",
            "Moderate risk tolerance, ten year horizon, 60/40 allocation between equities and "
            "bonds. The client accepts drawdowns of up to 15 percent.",
        ),
        (
            "ESG preferences",
            "The client wants to exclude fossil fuel producers and increase exposure to "
            "renewable energy and green bonds.",
        ),
    ],
    "maria": [
        (
            "Tenancy agreement",
            "Residential tenancy agreement between the landlord and tenant Maria Rossi for the "
            "apartment at Via Roma 5, Milan, starting January 2025.",
        ),
        (
            "Last will and testament",
            "I, Maria Rossi, leave my property in Tuscany to my grandchildren in equal shares. "
            "Executor: Luca Rossi.",
        ),
        (
            "Tax return 2025",
            "Annual income tax return with capital gains of 12,000 EUR and dividend income of "
            "3,400 EUR. Total tax paid: 5,100 EUR.",
        ),
        (
            "Pension statement",
            "State pension and private annuity payments of 2,300 EUR per month since "
            "retiring from the hospital in 2022.",
        ),
    ],
    "ahmed": [
        (
            "Bank statement",
            "Current account statement for September 2026 sent to 7 Rue de Rivoli, Paris. "
            "Closing balance 12,400 EUR.",
        ),
        (
            "Meeting notes",
            "Discussed funding university education for two children and buying a holiday "
            "home in Portugal within five years.",
        ),
        (
            "Business sale agreement",
            "Share purchase agreement for the sale of Al-Sayed Logistics Ltd for 4.5 million "
            "EUR, the main source of the client's wealth.",
        ),
    ],
}


@pytest.fixture(scope="module")
def api(database_url: str) -> Iterator[TestClient]:
    settings = Settings(
        _env_file=None, database_url=database_url, api_key=None, anthropic_api_key=None
    )
    embedder = FastEmbedEmbedder(settings.embedding_model, settings.embedding_cache_dir)
    _truncate(database_url)
    with TestClient(create_app(settings, embedder)) as client:
        owners = {
            "john": create_client(client),
            "maria": create_client(
                client, first_name="Maria", last_name="Rossi", email="maria@rossi.it"
            ),
            "ahmed": create_client(
                client, first_name="Ahmed", last_name="Al-Sayed", email="ahmed@alsayed.fr"
            ),
        }
        for owner, documents in CORPUS.items():
            for title, content in documents:
                create_document(client, owners[owner]["id"], title, content)
        yield client
    _truncate(database_url)


def document_titles(api: TestClient, query: str) -> list[str]:
    response = api.get("/search", params={"q": query, "type": "document"})
    assert response.status_code == 200
    return [r["document"]["title"] for r in response.json()]


@pytest.mark.parametrize(
    ("query", "expected_top"),
    [
        # The example from the assignment.
        ("address proof", {"Utility bill - March 2026", "Tenancy agreement", "Bank statement"}),
        ("proof of identity", {"Passport copy"}),
        ("how much tax did she pay", {"Tax return 2025"}),
        ("risk appetite", {"Investment policy statement"}),
        ("retirement income", {"Pension statement"}),
        ("source of wealth", {"Business sale agreement"}),
        # MiniLM alone ranks the generic investment policy first here; the thesaurus fixes it.
        ("sustainable investing", {"ESG preferences"}),
        # Purely semantic: no shared words, no thesaurus entry.
        ("inheritance for grandchildren", {"Last will and testament"}),
        ("saving for kids' college", {"Meeting notes"}),
    ],
)
def test_relevant_documents_rank_first(api: TestClient, query: str, expected_top: set[str]) -> None:
    titles = document_titles(api, query)
    assert set(titles[: len(expected_top)]) == expected_top, titles


def test_address_proof_does_not_rank_identity_documents_above_address_documents(
    api: TestClient,
) -> None:
    titles = document_titles(api, "address proof")
    if "Passport copy" in titles:
        assert titles.index("Passport copy") >= 3


@pytest.mark.parametrize("query", ["chocolate cake recipe", "football results"])
def test_unrelated_queries_return_nothing(api: TestClient, query: str) -> None:
    assert document_titles(api, query) == []
