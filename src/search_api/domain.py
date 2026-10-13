"""Domain objects passed between the layers.

Repositories build them from database rows, services work with them, and the API layer turns
them into response schemas. They are plain immutable values, not tied to a database session, so
they stay valid after the transaction that loaded them has ended.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

SearchType = Literal["client", "document"]
ClientMatch = Literal["name", "email", "description"]
DocumentMatch = Literal["keyword", "semantic"]
SummaryMethod = Literal["llm", "extractive"]


@dataclass(frozen=True, slots=True)
class NewClient:
    first_name: str
    last_name: str
    email: str
    description: str | None = None
    social_links: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Client:
    id: uuid.UUID
    first_name: str
    last_name: str
    email: str
    description: str | None
    social_links: list[str]
    created_at: datetime


@dataclass(frozen=True, slots=True)
class NewDocument:
    title: str
    content: str


@dataclass(frozen=True, slots=True)
class Document:
    id: uuid.UUID
    client_id: uuid.UUID
    title: str
    content: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class Summary:
    document_id: uuid.UUID
    text: str
    method: SummaryMethod


@dataclass(frozen=True, slots=True)
class ClientResult:
    score: float
    matched_by: list[ClientMatch]
    client: Client


@dataclass(frozen=True, slots=True)
class DocumentResult:
    score: float
    matched_by: list[DocumentMatch]
    snippet: str
    document: Document


@dataclass(frozen=True, slots=True)
class SearchResults:
    results: list[ClientResult | DocumentResult]
    # Phrases the query was expanded with (thesaurus, then LLM).
    expansions: list[str]
