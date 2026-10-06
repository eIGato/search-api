from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://search:search@localhost:5433/search"
    # Per worker process. Requests hold a connection only for their short SQL phases (never
    # while embedding or calling the LLM), so a small pool serves many concurrent requests.
    db_pool_size: int = Field(default=10, ge=1)
    db_max_overflow: int = Field(default=10, ge=0)
    db_pool_timeout_seconds: float = Field(default=30, gt=0)

    # When set, every endpoint except /health and the docs requires an `X-API-Key` header.
    api_key: SecretStr | None = None

    # Embeddings. The vector column dimension is fixed by the migration, so changing the model to
    # one with a different dimension requires a new migration (checked at startup).
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    embedding_cache_dir: str | None = None
    # ONNX Runtime threads per embedding call (None: one per physical core).
    embedding_threads: int | None = Field(default=None, ge=1)

    # Chunking: documents are split into overlapping word windows so long documents do not get
    # truncated by the model's context window (256 word pieces for MiniLM).
    chunk_words: int = Field(default=120, ge=10)
    chunk_overlap_words: int = Field(default=30, ge=0)

    # Search tuning.
    # Cosine similarity below which a semantic-only hit is considered noise. Model-specific:
    # MiniLM scores relevant passages around 0.3-0.6 and unrelated ones around 0.0-0.2.
    min_semantic_similarity: float = 0.3
    # Candidates fetched per retriever before fusion.
    candidate_pool: int = 50
    # Reciprocal Rank Fusion constant.
    rrf_k: int = 60
    # Weight applied to similarities obtained through query expansion rather than the query itself.
    expansion_weight: float = 0.9
    # pg_trgm strict_word_similarity threshold for fuzzy (typo-tolerant) client matches.
    client_fuzzy_threshold: float = 0.4

    # Optional LLM features (document summaries, LLM query expansion). Without a key the service
    # falls back to extractive summaries and thesaurus-only query expansion.
    anthropic_api_key: SecretStr | None = None
    llm_model: str = "claude-haiku-4-5"
    llm_query_expansion: bool = True
    llm_timeout_seconds: float = 10.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
