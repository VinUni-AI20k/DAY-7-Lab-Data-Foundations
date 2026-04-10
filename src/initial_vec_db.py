import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=False)

# Ensure the root dir is in sys.path when running as a stand-alone script
sys.path.append(str(Path(__file__).parent.parent))

from src.models import Document
from src.store import EmbeddingStore
from src.chunking import MarkdownChunker, SentenceChunker, SemanticChunker, RecursiveChunker, FixedSizeChunker
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV, LOCAL_EMBEDDING_MODEL, OPENAI_EMBEDDING_MODEL,
    LocalEmbedder, OpenAIEmbedder, _mock_embed
)

def load_all_markdowns(dir_path: str) -> list[Document]:
    docs = []
    base_path = Path(dir_path)
    for path in base_path.rglob("*.md"):
        content = path.read_text(encoding="utf-8")
        docs.append(Document(
            id=path.stem,
            content=content,
            metadata={"source": str(path), "extension": ".md"}
        ))
    return docs

def init_db():
    print("=== Initializing Vector DB from mydata/ ===")
    load_dotenv(override=False)
    
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "local":
        try:
            embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    elif provider == "openai":
        try:
            embedder = OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    else:
        embedder = _mock_embed

    store = EmbeddingStore(collection_name=os.getenv("COLLECTION_NAME"), embedding_fn=embedder)
    
    print("Loading Markdown documents from 'mydata/'...")
    raw_docs = load_all_markdowns("mydata")
    if not raw_docs:
        print("No markdown files found in 'mydata/'. Exiting.")
        return
        
    print(f"Loaded {len(raw_docs)} raw documents.")
    
    # chunker = MarkdownChunker(chunk_size=600)
    # chunker = SentenceChunker(max_sentences_per_chunk=3)
    # chunker = SemanticChunker(embedder_fn=embedder, similarity_threshold=0.5)
    chunker = FixedSizeChunker()
    chunked_docs = []
    
    for doc in raw_docs:
        chunks = chunker.chunk(doc.content)
        for i, c in enumerate(chunks):
            # Create a separate chunk representation, preserving the overall doc ID and metadata
            chunked_docs.append(Document(
                id=doc.id, 
                content=c,
                metadata={**doc.metadata, "chunk_index": i}
            ))

    print(f"Chunked into {len(chunked_docs)} pieces. Ingesting into Vector DB...")
    store.add_documents(chunked_docs)
    print(f"Success! Vector store {os.getenv("COLLECTION_NAME")} now holds {store.get_collection_size()} chunks.")

if __name__ == "__main__":
    init_db()
