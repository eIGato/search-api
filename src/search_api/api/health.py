from fastapi import APIRouter

from search_api.api.dependencies import AppContainer

router = APIRouter(tags=["health"])


@router.get("/health", summary="Liveness and database check")
async def health(container: AppContainer) -> dict[str, str]:
    await container.database.ping()
    return {"status": "ok"}
