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
            import chromadb

            self._collection = chromadb.Client().get_or_create_collection(name=self._collection_name)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        content = doc.content if isinstance(doc.content, str) else str(doc.content)
        metadata = dict(doc.metadata) if isinstance(doc.metadata, dict) else {}
        metadata.setdefault("doc_id", doc.id)

        idx = self._next_index
        self._next_index = idx + 1
        record_id = f"{doc.id}:{idx}"

        embedding = self._embedding_fn(content)
        return {
            "id": record_id,
            "doc_id": doc.id,
            "content": content,
            "text": content,
            "document": content,
            "metadata": metadata,
            "embedding": embedding,
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if top_k <= 0 or not records:
            return []

        q_emb = self._embedding_fn(query)
        limit = min(top_k, len(records))

        scored: list[tuple[float, dict[str, Any]]] = []
        append_score = scored.append
        dot = _dot
        for rec in records:
            append_score((dot(q_emb, rec["embedding"]), rec))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [{**rec, "score": score} for score, rec in scored[:limit]]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        if not docs:
            return

        records = [self._make_record(doc) for doc in docs]
        self._store.extend(records)

        if self._use_chroma and self._collection is not None:
            try:
                self._collection.add(
                    ids=[r["id"] for r in records],
                    documents=[r["content"] for r in records],
                    embeddings=[r["embedding"] for r in records],
                    metadatas=[r["metadata"] for r in records],
                )
            except Exception:
                self._use_chroma = False

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        if top_k <= 0:
            return []

        if self._use_chroma and self._collection is not None:
            try:
                q_emb = self._embedding_fn(query)
                result = self._collection.query(
                    query_embeddings=[q_emb],
                    n_results=top_k,
                    include=["documents", "metadatas", "embeddings", "distances"],
                )

                ids = result.get("ids", [[]])[0]
                docs = result.get("documents", [[]])[0]
                metas = result.get("metadatas", [[]])[0]
                embs = result.get("embeddings", [[]])[0]
                dists = result.get("distances", [[]])[0]

                out: list[dict[str, Any]] = []
                for i, record_id in enumerate(ids):
                    emb = embs[i] if i < len(embs) else None
                    score = _dot(q_emb, emb) if emb is not None else -(dists[i] if i < len(dists) else 0.0)
                    content = docs[i] if i < len(docs) else ""
                    metadata = metas[i] if i < len(metas) and isinstance(metas[i], dict) else {}
                    out.append(
                        {
                            "id": record_id,
                            "doc_id": metadata.get("doc_id", str(record_id).split(":", 1)[0]),
                            "content": content,
                            "text": content,
                            "document": content,
                            "metadata": metadata,
                            "embedding": emb if emb is not None else [],
                            "score": score,
                        }
                    )

                out.sort(key=lambda item: item["score"], reverse=True)
                return out[:top_k]
            except Exception:
                self._use_chroma = False

        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma and self._collection is not None:
            try:
                return int(self._collection.count())
            except Exception:
                self._use_chroma = False
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if not metadata_filter:
            return self.search(query, top_k=top_k)

        if self._use_chroma and self._collection is not None:
            try:
                q_emb = self._embedding_fn(query)
                result = self._collection.query(
                    query_embeddings=[q_emb],
                    n_results=top_k,
                    where=metadata_filter,
                    include=["documents", "metadatas", "embeddings", "distances"],
                )

                ids = result.get("ids", [[]])[0]
                docs = result.get("documents", [[]])[0]
                metas = result.get("metadatas", [[]])[0]
                embs = result.get("embeddings", [[]])[0]
                dists = result.get("distances", [[]])[0]

                out: list[dict[str, Any]] = []
                for i, record_id in enumerate(ids):
                    emb = embs[i] if i < len(embs) else None
                    score = _dot(q_emb, emb) if emb is not None else -(dists[i] if i < len(dists) else 0.0)
                    content = docs[i] if i < len(docs) else ""
                    metadata = metas[i] if i < len(metas) and isinstance(metas[i], dict) else {}
                    out.append(
                        {
                            "id": record_id,
                            "doc_id": metadata.get("doc_id", str(record_id).split(":", 1)[0]),
                            "content": content,
                            "text": content,
                            "document": content,
                            "metadata": metadata,
                            "embedding": emb if emb is not None else [],
                            "score": score,
                        }
                    )
                out.sort(key=lambda item: item["score"], reverse=True)
                return out[:top_k]
            except Exception:
                self._use_chroma = False

        filtered = [
            rec
            for rec in self._store
            if all(rec.get("metadata", {}).get(key) == value for key, value in metadata_filter.items())
        ]
        return self._search_records(query, filtered, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        if not doc_id:
            return False

        before = len(self._store)
        self._store = [
            rec
            for rec in self._store
            if rec.get("doc_id") != doc_id and rec.get("metadata", {}).get("doc_id") != doc_id
        ]
        removed = len(self._store) < before

        if self._use_chroma and self._collection is not None:
            try:
                existing = self._collection.get(where={"doc_id": doc_id}, include=[])
                ids = existing.get("ids", [])
                if ids:
                    self._collection.delete(ids=ids)
                    removed = True
            except Exception:
                self._use_chroma = False

        return removed
