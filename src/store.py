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
            import os
            
            persist_dir = os.getenv("CHROMA_PERSIST_DIR")
            if persist_dir:
                self._client = chromadb.PersistentClient(path=persist_dir)
            else:
                self._client = chromadb.Client()
                
            if self._collection_name.startswith("test"):
                try:
                    self._client.delete_collection(self._collection_name)
                except Exception:
                    pass
            self._collection = self._client.get_or_create_collection(
                self._collection_name, metadata={"hnsw:space": "cosine"}
            )
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        self._next_index += 1
        meta = dict(doc.metadata) if doc.metadata else {}
        if "doc_id" not in meta:
            meta["doc_id"] = doc.id
        return {
            "id": f"chunk_{self._next_index}",
            "content": doc.content,
            "metadata": meta,
            "embedding": self._embedding_fn(doc.content)
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        from .chunking import compute_similarity
        query_emb = self._embedding_fn(query)
        scored = []
        for r in records:
            score = compute_similarity(query_emb, r["embedding"])
            scored.append({
                "id": r["id"],
                "content": r["content"],
                "metadata": r["metadata"],
                "score": score
            })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.
        """
        if not docs:
            return
        if self._use_chroma:
            ids, docs_content, metadatas, embeddings = [], [], [], []
            for doc in docs:
                rec = self._make_record(doc)
                ids.append(rec["id"])
                docs_content.append(rec["content"])
                metadatas.append(rec["metadata"])
                embeddings.append(rec["embedding"])
            self._collection.add(
                ids=ids, documents=docs_content, embeddings=embeddings, metadatas=metadatas
            )
        else:
            for doc in docs:
                self._store.append(self._make_record(doc))

    def search(self, query: str, top_k: int = 5, threshold: float = 0.0) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query, dropping any below the threshold.
        """
        # Tăng lượng query db ảo nếu đang dùng threshold để không bị sót
        if threshold > 0.0 and top_k < 20:
            top_k = 20
            
        if self._use_chroma:
            query_emb = self._embedding_fn(query)
            res = self._collection.query(query_embeddings=[query_emb], n_results=top_k)
            results = []
            if res and res.get("ids") and res["ids"][0]:
                for i in range(len(res["ids"][0])):
                    score = 1.0 - res["distances"][0][i]
                    if score >= threshold:
                        results.append({
                            "id": res["ids"][0][i],
                            "content": res["documents"][0][i],
                            "metadata": res["metadatas"][0][i],
                            "score": score
                        })
            results.sort(key=lambda x: x["score"], reverse=True)
            return results
        else:
            scored = self._search_records(query, self._store, top_k)
            return [r for r in scored if r["score"] >= threshold]

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.
        """
        if metadata_filter is None:
            metadata_filter = {}
            
        if self._use_chroma:
            if not metadata_filter:
                where = None
            elif len(metadata_filter) == 1:
                k, v = list(metadata_filter.items())[0]
                where = {k: v}
            else:
                where = {"$and": [{k: v} for k, v in metadata_filter.items()]}
                
            query_emb = self._embedding_fn(query)
            res = self._collection.query(
                query_embeddings=[query_emb], 
                n_results=top_k,
                where=where
            )
            results = []
            if res and res.get("ids") and res["ids"][0]:
                for i in range(len(res["ids"][0])):
                    results.append({
                        "id": res["ids"][0][i],
                        "content": res["documents"][0][i],
                        "metadata": res["metadatas"][0][i],
                        "score": 1.0 - res["distances"][0][i]
                    })
            results.sort(key=lambda x: x["score"], reverse=True)
            return results
        else:
            filtered = []
            for r in self._store:
                match = True
                for k, v in metadata_filter.items():
                    if r["metadata"].get(k) != v:
                        match = False
                        break
                if match:
                    filtered.append(r)
            return self._search_records(query, filtered, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.
        """
        if self._use_chroma:
            initial_count = self._collection.count()
            self._collection.delete(where={"doc_id": doc_id})
            return self._collection.count() < initial_count
        else:
            initial_count = len(self._store)
            self._store = [r for r in self._store if r["metadata"].get("doc_id") != doc_id]
            return len(self._store) < initial_count
