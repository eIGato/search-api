import json
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from search_api import models, schemas
from search_api.auth import require_api_key
from search_api.chunking import chunk_text
from search_api.config import Settings
from search_api.db import get_session, release_connection
from search_api.embeddings import Embedder, embed_passages
from search_api.llm import LLMClient
from search_api.search import SearchType, search
from search_api.summary import get_summary

Session = Annotated[AsyncSession, Depends(get_session)]


def get_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_embedder(request: Request) -> Embedder:
    embedder: Embedder = request.app.state.embedder
    return embedder


def get_llm(request: Request) -> LLMClient | None:
    llm: LLMClient | None = request.app.state.llm
    return llm


AppSettings = Annotated[Settings, Depends(get_settings)]
AppEmbedder = Annotated[Embedder, Depends(get_embedder)]
AppLLM = Annotated[LLMClient | None, Depends(get_llm)]

ResponsesDoc = dict[int | str, dict[str, Any]]
NOT_FOUND: ResponsesDoc = {status.HTTP_404_NOT_FOUND: {"model": schemas.ErrorResponse}}
UNAUTHORIZED: ResponsesDoc = {status.HTTP_401_UNAUTHORIZED: {"model": schemas.ErrorResponse}}

router = APIRouter(dependencies=[Depends(require_api_key)], responses=UNAUTHORIZED)
health_router = APIRouter()


def _parse_id(value: str, entity: str) -> uuid.UUID:
    # IDs are opaque strings in the API contract: a malformed ID is just an ID that does not
    # exist, so it is a 404 rather than a validation error.
    try:
        return uuid.UUID(value)
    except ValueError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"{entity} not found") from None


async def _get_client(session: AsyncSession, client_id: str) -> models.Client:
    client = await session.get(models.Client, _parse_id(client_id, "Client"))
    if client is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Client not found")
    return client


async def _get_document(session: AsyncSession, client_id: str, document_id: str) -> models.Document:
    document = await session.get(models.Document, _parse_id(document_id, "Document"))
    if document is None or document.client_id != _parse_id(client_id, "Client"):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    return document


# --- Clients -----------------------------------------------------------------------------------


@router.post(
    "/clients",
    status_code=status.HTTP_201_CREATED,
    response_model=schemas.Client,
    responses={status.HTTP_409_CONFLICT: {"model": schemas.ErrorResponse}},
    tags=["clients"],
    summary="Create a client",
)
async def create_client(
    payload: schemas.ClientCreate, session: Session, request: Request, response: Response
) -> models.Client:
    client = models.Client(**payload.model_dump())
    session.add(client)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise HTTPException(
            status.HTTP_409_CONFLICT, "A client with this email already exists"
        ) from None
    await session.refresh(client)
    response.headers["Location"] = str(request.url_for("get_client", client_id=str(client.id)))
    return client


@router.get(
    "/clients/{client_id}",
    response_model=schemas.Client,
    responses=NOT_FOUND,
    tags=["clients"],
    summary="Get a client",
)
async def get_client(client_id: str, session: Session) -> models.Client:
    return await _get_client(session, client_id)


# --- Documents ---------------------------------------------------------------------------------


@router.post(
    "/clients/{client_id}/documents",
    status_code=status.HTTP_201_CREATED,
    response_model=schemas.Document,
    responses=NOT_FOUND,
    tags=["documents"],
    summary="Add a document to a client",
    description="The document is chunked and embedded synchronously, so it is searchable as "
    "soon as this request returns.",
)
async def create_document(
    client_id: str,
    payload: schemas.DocumentCreate,
    session: Session,
    settings: AppSettings,
    embedder: AppEmbedder,
    request: Request,
    response: Response,
) -> models.Document:
    client = await _get_client(session, client_id)
    await release_connection(session)  # embedding is CPU-bound and can take a while

    chunks = chunk_text(payload.content, settings.chunk_words, settings.chunk_overlap_words)
    # The title is prepended to every chunk so that each one carries the document's context.
    vectors = await embed_passages(embedder, [f"{payload.title}\n{c}" for c in chunks])

    document = models.Document(client_id=client.id, title=payload.title, content=payload.content)
    document.chunks = [
        models.DocumentChunk(chunk_index=i, content=chunk, embedding=vector)
        for i, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
    ]
    session.add(document)
    await session.commit()
    await session.refresh(document)
    response.headers["Location"] = str(
        request.url_for("get_document", client_id=client_id, document_id=str(document.id))
    )
    return document


@router.get(
    "/clients/{client_id}/documents",
    response_model=list[schemas.Document],
    responses=NOT_FOUND,
    tags=["documents"],
    summary="List a client's documents",
)
async def list_documents(client_id: str, session: Session) -> list[models.Document]:
    client = await _get_client(session, client_id)
    stmt = (
        select(models.Document)
        .where(models.Document.client_id == client.id)
        .order_by(models.Document.created_at.desc())
    )
    return list(await session.scalars(stmt))


@router.get(
    "/clients/{client_id}/documents/{document_id}",
    response_model=schemas.Document,
    responses=NOT_FOUND,
    tags=["documents"],
    summary="Get a document",
)
async def get_document(client_id: str, document_id: str, session: Session) -> models.Document:
    return await _get_document(session, client_id, document_id)


@router.get(
    "/clients/{client_id}/documents/{document_id}/summary",
    response_model=schemas.DocumentSummary,
    responses=NOT_FOUND,
    tags=["documents"],
    summary="Get a short summary of a document",
    description="Generated on first request and cached. Uses the configured LLM when "
    "ANTHROPIC_API_KEY is set, otherwise (or if the LLM call fails) picks the document's most "
    "representative sentences.",
)
async def get_document_summary(
    client_id: str,
    document_id: str,
    session: Session,
    embedder: AppEmbedder,
    llm: AppLLM,
) -> schemas.DocumentSummary:
    document = await _get_document(session, client_id, document_id)
    summary, method = await get_summary(session, document, embedder, llm)
    return schemas.DocumentSummary(document_id=document.id, summary=summary, method=method)


# --- Search ------------------------------------------------------------------------------------


@router.get(
    "/search",
    response_model=list[schemas.SearchResult],
    tags=["search"],
    summary="Search clients and documents",
    description="""
Searches clients by name, email and description (exact, substring and fuzzy matches), and
documents by keyword and meaning (hybrid full-text + embedding search with query expansion).

Results of both types are returned in one list sorted by `score` (0-1, higher is better).
Each result has a `type` discriminator (`client` or `document`) and `matched_by`, which explains
why it matched. The phrases the query was expanded with are returned in the
`X-Search-Expansions` response header (JSON array).
""",
)
async def search_endpoint(
    session: Session,
    settings: AppSettings,
    embedder: AppEmbedder,
    llm: AppLLM,
    response: Response,
    q: Annotated[
        str,
        Query(min_length=1, max_length=500, pattern=r"\S", description="Free-text query."),
    ],
    type: Annotated[SearchType | None, Query(description="Restrict results to one type.")] = None,
    client_id: Annotated[
        uuid.UUID | None,
        Query(
            description="Only search this client's documents (implies type=document; "
            "cannot be combined with type=client)."
        ),
    ] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list[schemas.SearchResult]:
    query = " ".join(q.split())
    types: set[SearchType] = {type} if type else {"client", "document"}
    if client_id is not None:
        if type == "client":
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                "client_id restricts the search to documents; it cannot be combined with "
                "type=client",
            )
        types = {"document"}

    outcome = await search(session, embedder, settings, query, types, limit, client_id, llm)
    response.headers["X-Search-Expansions"] = json.dumps(outcome.expansions)  # ASCII-escaped
    return outcome.results


# --- Health ------------------------------------------------------------------------------------


@health_router.get("/health", tags=["health"], summary="Liveness and database check")
async def health(session: Session) -> dict[str, str]:
    await session.execute(text("SELECT 1"))
    return {"status": "ok"}
