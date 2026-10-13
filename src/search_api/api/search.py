import json
import uuid
from typing import Annotated

from fastapi import APIRouter, Query, Response

from search_api import schemas
from search_api.api.dependencies import Search
from search_api.domain import ClientResult, DocumentResult, SearchType

router = APIRouter(tags=["search"])


def _to_schema(result: ClientResult | DocumentResult) -> schemas.SearchResult:
    if isinstance(result, ClientResult):
        return schemas.ClientSearchResult(
            score=result.score,
            matched_by=result.matched_by,
            client=schemas.Client.model_validate(result.client),
        )
    return schemas.DocumentSearchResult(
        score=result.score,
        matched_by=result.matched_by,
        snippet=result.snippet,
        document=schemas.Document.model_validate(result.document),
    )


@router.get(
    "/search",
    response_model=list[schemas.SearchResult],
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
    search: Search,
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
    outcome = await search.search(q, result_type=type, client_id=client_id, limit=limit)
    response.headers["X-Search-Expansions"] = json.dumps(outcome.expansions)  # ASCII-escaped
    return [_to_schema(result) for result in outcome.results]
