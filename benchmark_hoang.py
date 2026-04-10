from pathlib import Path
import glob
import os
import sys
from typing import List

from src.chunking import RecursiveChunker
from src.store import EmbeddingStore
from src.models import Document
from src.embeddings import MockEmbedder

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

def load_data(pattern: str) -> List[tuple[str, str]]:
    """Load files matching pattern and return list of (path, content)."""
    base = Path(__file__).parent
    matches = glob.glob(str(base / pattern))
    out: list[tuple[str, str]] = []
    for p in sorted(matches):
        try:
            with open(p, "r", encoding="utf-8") as fh:
                out.append((os.path.basename(p), fh.read()))
        except Exception:
            # skip unreadable files
            continue
    return out


def chunk_documents(docs, chunker, min_len=30):
    out = []
    seen = set()
    for fname, content in docs:
        for i, c in enumerate(chunker.chunk(content)):
            txt = c.strip()
            if len(txt) < min_len:
                continue
            if txt in seen:
                continue
            seen.add(txt)
            doc_id = f"{fname}__chunk__{i}"
            metadata = {"source": fname, "chunk_index": i}
            out.append(Document(id=doc_id, content=txt, metadata=metadata))
    return out


def main() -> None:
    docs = load_data("data/*.md")
    if not docs:
        print("No documents found at data/*.md — nothing to index.")
        return

    chunker = RecursiveChunker(chunk_size=300)
    chunked = chunk_documents(docs, chunker)

    embedder = MockEmbedder()
    store = EmbeddingStore(collection_name="my_strategy", embedding_fn=embedder)
    store.add_documents(chunked)

    QUERIES = [
        {"q": "Đối tác giao hàng thất bại làm gì?", "gold": "Chụp ảnh bằng chứng..."},
        {"q": "Làm sao để yêu cầu hoàn tiền?", "gold": "Gửi yêu cầu..."},
    ]

    print("=== MY STRATEGY RESULTS ===")
    for item in QUERIES:
        q = item.get("q", "")
        results = store.search(q, top_k=3)
        print(f"Query: {q}")
        print("Top-3:")
        for r in results:
            src = r.get("metadata", {}).get("source")
            score = r.get("score")
            snippet = (r.get("content") or "")[:120].replace("\n", " ")
            print(f" - source={src} score={score:.4f} snippet={snippet}")
        print(f"Gold: {item.get('gold')}")
        print()


if __name__ == "__main__":
    main()