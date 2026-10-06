import secrets

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from search_api.config import Settings

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="Required only when the server is started with API_KEY set.",
)


async def require_api_key(request: Request, api_key: str | None = Security(api_key_header)) -> None:
    settings: Settings = request.app.state.settings
    if settings.api_key is None:
        return
    expected = settings.api_key.get_secret_value().encode()
    if api_key is None or not secrets.compare_digest(api_key.encode(), expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key",
            headers={"WWW-Authenticate": "APIKey"},
        )
