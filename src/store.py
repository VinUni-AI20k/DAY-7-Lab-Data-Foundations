from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb  # noqa: F401

            # TODO: initialize chromadb client + collection
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        metadata = dict(doc.metadata) if doc.metadata is not None else {}
        # keep a reference to the original document id for deletes/filters
        metadata.setdefault('doc_id', doc.id)
        embedding = self._embedding_fn(doc.content)
        record = {
            'id': f"{self._collection_name}_{self._next_index}_{doc.id}",
            'content': doc.content,
            'metadata': metadata,
            'embedding': embedding,
        }
        return record

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if not records:
            return []
        q_vec = self._embedding_fn(query)
        scored: list[dict[str, Any]] = []
        for r in records:
            score = _dot(q_vec, r['embedding'])
            scored.append({
                'id': r.get('id'),
                'content': r.get('content'),
                'metadata': r.get('metadata', {}),
                'score': score,
            })
        scored.sort(key=lambda x: x['score'], reverse=True)
        return scored[:max(0, top_k)]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        for doc in docs:
            rec = self._make_record(doc)
            self._store.append(rec)
            self._next_index += 1

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if metadata_filter is None:
            return self.search(query, top_k=top_k)

        def match(rec_meta: dict, filt: dict) -> bool:
            for k, v in filt.items():
                if rec_meta.get(k) != v:
                    return False
            return True

        filtered = [r for r in self._store if match(r.get('metadata', {}), metadata_filter)]
        return self._search_records(query, filtered, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        before = len(self._store)
        self._store = [r for r in self._store if r.get('metadata', {}).get('doc_id') != doc_id]
        after = len(self._store)
        return after < before
