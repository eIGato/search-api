import uuid

from search_api.chunking import chunk_text
from search_api.config import Settings
from search_api.db import Database
from search_api.domain import Document, NewDocument
from search_api.embeddings import Embedder, embed_passages
from search_api.errors import NotFoundError


class DocumentService:
    def __init__(self, database: Database, embedder: Embedder, settings: Settings) -> None:
        self._db = database
        self._embedder = embedder
        self._chunk_words = settings.chunk_words
        self._chunk_overlap_words = settings.chunk_overlap_words

    async def create(self, client_id: uuid.UUID, new: NewDocument) -> Document:
        """Chunk, embed and store a document; it is searchable once this returns.

        Embedding is slow, so it runs before the write transaction instead of inside it. The
        write transaction then checks the client again and locks it: the first check happened
        in another transaction, and the client must not disappear before the document and its
        chunks are committed together.
        """
        # Fail fast: do not spend the embedding work on a client that does not exist.
        async with self._db.transaction(read_only=True) as uow:
            if not await uow.clients.exists(client_id):
                raise NotFoundError("Client")

        chunks = chunk_text(new.content, self._chunk_words, self._chunk_overlap_words)
        # The title is prepended to every chunk so that each one carries the document's context.
        vectors = await embed_passages(self._embedder, [f"{new.title}\n{c}" for c in chunks])

        async with self._db.transaction() as uow:
            if not await uow.clients.lock_for_reference(client_id):
                raise NotFoundError("Client")
            return await uow.documents.add(client_id, new, list(zip(chunks, vectors, strict=True)))

    async def list_for_client(self, client_id: uuid.UUID) -> list[Document]:
        async with self._db.transaction(read_only=True) as uow:
            if not await uow.clients.exists(client_id):
                raise NotFoundError("Client")
            return await uow.documents.list_for_client(client_id)

    async def get(self, client_id: uuid.UUID, document_id: uuid.UUID) -> Document:
        async with self._db.transaction(read_only=True) as uow:
            document = await uow.documents.get(client_id, document_id)
        if document is None:
            raise NotFoundError("Document")
        return document
