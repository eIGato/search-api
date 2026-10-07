import asyncio
import hashlib
import os
import re
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import asyncpg
import numpy as np
import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient

from search_api.config import Settings
from search_api.embeddings import Embedder
from search_api.main import create_app
from search_api.models import EMBEDDING_DIM

ROOT = Path(__file__).resolve().parent.parent

# A JSON object from an API response.
Json = dict[str, Any]


class HashingEmbedder:
    """Deterministic bag-of-words embedder: cosine similarity ~ shared (crudely stemmed) words.

    Fast and model-free, so API tests do not depend on a downloaded model. Semantic quality is
    covered separately by tests marked `model`.
    """

    dim = EMBEDDING_DIM

    def _embed(self, text: str) -> list[float]:
        vector = np.zeros(self.dim, dtype=np.float32)
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            stem = word[:6]
            digest = hashlib.blake2b(stem.encode(), digest_size=4).digest()
            vector[int.from_bytes(digest, "little") % self.dim] += 1.0
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            vector[0] = 1.0
            norm = 1.0
        normalized: list[float] = (vector / norm).tolist()
        return normalized

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]


@pytest.fixture(scope="session")
def database_url() -> Iterator[str]:
    """A migrated database: TEST_DATABASE_URL if set, otherwise a throwaway container."""
    url = os.environ.get("TEST_DATABASE_URL")
    container = None
    if url is None:
        from testcontainers.community.postgres import PostgresContainer

        container = PostgresContainer("pgvector/pgvector:pg18", driver="asyncpg")
        container.start()
        url = container.get_connection_url()

    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "migrations"))
    config.attributes["database_url"] = url
    config.attributes["configure_logger"] = False
    command.upgrade(config, "head")
    try:
        yield url
    finally:
        if container is not None:
            container.stop()


def _truncate(database_url: str) -> None:
    async def run() -> None:
        conn = await asyncpg.connect(database_url.replace("postgresql+asyncpg", "postgresql"))
        try:
            await conn.execute(
                "TRUNCATE clients, documents, document_chunks, query_expansions CASCADE"
            )
        finally:
            await conn.close()

    asyncio.run(run())


@pytest.fixture
def settings(database_url: str) -> Settings:
    # Tests ignore the developer's .env, and the keys are explicitly unset, so that local
    # settings and secrets do not leak into tests.
    return Settings(_env_file=None, database_url=database_url, api_key=None, anthropic_api_key=None)


@pytest.fixture
def embedder() -> Embedder:
    return HashingEmbedder()


@pytest.fixture
def client(settings: Settings, embedder: Embedder, database_url: str) -> Iterator[TestClient]:
    _truncate(database_url)
    with TestClient(create_app(settings, embedder)) as test_client:
        yield test_client


def create_client(api: TestClient, **overrides: object) -> Json:
    payload: dict[str, object] = {
        "first_name": "John",
        "last_name": "Doe",
        "email": "john.doe@neviswealth.com",
        "description": "Tech founder interested in sustainable investing",
    }
    payload.update(overrides)
    response = api.post("/clients", json=payload)
    assert response.status_code == 201, response.text
    body: Json = response.json()
    return body


def create_document(api: TestClient, client_id: object, title: str, content: str) -> Json:
    response = api.post(
        f"/clients/{client_id}/documents", json={"title": title, "content": content}
    )
    assert response.status_code == 201, response.text
    body: Json = response.json()
    return body
