from fastapi import APIRouter, Request, Response, status

from search_api import schemas
from search_api.api.dependencies import Documents, Summaries, parse_id
from search_api.api.errors import NOT_FOUND
from search_api.domain import NewDocument

router = APIRouter(tags=["documents"])


@router.post(
    "/clients/{client_id}/documents",
    status_code=status.HTTP_201_CREATED,
    response_model=schemas.Document,
    responses=NOT_FOUND,
    summary="Add a document to a client",
    description="The document is chunked and embedded synchronously, so it is searchable as "
    "soon as this request returns.",
)
async def create_document(
    client_id: str,
    payload: schemas.DocumentCreate,
    documents: Documents,
    request: Request,
    response: Response,
) -> schemas.Document:
    document = await documents.create(
        parse_id(client_id, "Client"), NewDocument(**payload.model_dump())
    )
    response.headers["Location"] = str(
        request.url_for("get_document", client_id=client_id, document_id=str(document.id))
    )
    return schemas.Document.model_validate(document)


@router.get(
    "/clients/{client_id}/documents",
    response_model=list[schemas.Document],
    responses=NOT_FOUND,
    summary="List a client's documents",
)
async def list_documents(client_id: str, documents: Documents) -> list[schemas.Document]:
    found = await documents.list_for_client(parse_id(client_id, "Client"))
    return [schemas.Document.model_validate(document) for document in found]


@router.get(
    "/clients/{client_id}/documents/{document_id}",
    response_model=schemas.Document,
    responses=NOT_FOUND,
    summary="Get a document",
)
async def get_document(client_id: str, document_id: str, documents: Documents) -> schemas.Document:
    document = await documents.get(
        parse_id(client_id, "Document"), parse_id(document_id, "Document")
    )
    return schemas.Document.model_validate(document)


@router.get(
    "/clients/{client_id}/documents/{document_id}/summary",
    response_model=schemas.DocumentSummary,
    responses=NOT_FOUND,
    summary="Get a short summary of a document",
    description="Generated on first request and cached. Uses the configured LLM when "
    "ANTHROPIC_API_KEY is set, otherwise (or if the LLM call fails) picks the document's most "
    "representative sentences.",
)
async def get_document_summary(
    client_id: str, document_id: str, summaries: Summaries
) -> schemas.DocumentSummary:
    summary = await summaries.get(
        parse_id(client_id, "Document"), parse_id(document_id, "Document")
    )
    return schemas.DocumentSummary(
        document_id=summary.document_id, summary=summary.text, method=summary.method
    )
