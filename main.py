from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

from src.agent import KnowledgeBaseAgent
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore

SAMPLE_FILES = [
    "data/python_intro.txt",
    "data/vector_store_notes.md",
    "data/rag_system_design.md",
    "data/customer_support_playbook.txt",
    "data/chunking_experiment_report.md",
    "data/vi_retrieval_notes.md",
]

load_dotenv(override=False)

def load_documents_from_files(file_paths: list[str]) -> list[Document]:
    """Load documents from file paths for the manual demo."""
    allowed_extensions = {".md", ".txt"}
    documents: list[Document] = []

    for raw_path in file_paths:
        path = Path(raw_path)

        if path.suffix.lower() not in allowed_extensions:
            print(f"Skipping unsupported file type: {path} (allowed: .md, .txt)")
            continue

        if not path.exists() or not path.is_file():
            print(f"Skipping missing file: {path}")
            continue

        content = path.read_text(encoding="utf-8")
        documents.append(
            Document(
                id=path.stem,
                content=content,
                metadata={"source": str(path), "extension": path.suffix.lower()},
            )
        )

    return documents


def demo_llm(prompt: str) -> str:
    """A simple mock LLM for manual RAG testing."""
    preview = prompt[:400].replace("\n", " ")
    return f"[DEMO LLM] Generated answer from prompt preview: {preview}..."


def get_llm_fn() -> Callable[[str], str]:
    """Return an LLM caller based on environment configs."""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        return demo_llm
        
    try:
        from openai import OpenAI
        base_url = os.getenv("DASHSCOPE_ENDPOINT", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
        model = os.getenv("QWEN_MODEL", "qwen3.5-27b")
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        def dashscope_llm(prompt: str) -> str:
            print(f"[LLM] Generating answer using Qwen model ({model})...")
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            return response.choices[0].message.content
            
        return dashscope_llm
    except ImportError:
        print("[LLM Warn] 'openai' package not installed, falling back to demo_llm")
        return demo_llm


def run_manual_demo(question: str | None = None, sample_files: list[str] | None = None) -> int:
    files = sample_files or SAMPLE_FILES
    query = question or "Summarize the key information from the loaded files."

    print("=== Manual File Test ===")
    
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

    print(f"\nEmbedding backend: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")

    store = EmbeddingStore(collection_name=os.getenv("COLLECTION_NAME"), embedding_fn=embedder)
    
    if store.get_collection_size() > 0:
        print(f"\n[Vector DB] Found {store.get_collection_size()} chunks in {os.getenv("COLLECTION_NAME")}. Skipping file ingestion.")
    else:
        print("Accepted file types: .md, .txt")
        print("Input file list:")
        for file_path in files:
            print(f"  - {file_path}")

        docs = load_documents_from_files(files)
        if not docs:
            print("\nNo valid input files were loaded.")
            print("Create files matching the sample paths above, then rerun:")
            print("  python3 main.py")
            return 1

        print(f"\nLoaded {len(docs)} documents")
        for doc in docs:
            print(f"  - {doc.id}: {doc.metadata['source']}")
            
        store.add_documents(docs)

    print(f"\nStored {store.get_collection_size()} documents in EmbeddingStore")
    print("\n=== EmbeddingStore Search Test ===")
    
    try:
        threshold = float(os.getenv("VEC_SIM_THREDHOLD", "0.0"))
    except ValueError:
        threshold = 0.0
        
    print(f"Query: {query} | Cosine Threshold: {threshold}")
    search_results = store.search(query, top_k=5, threshold=threshold)
    for index, result in enumerate(search_results, start=1):
        print(f"{index}. score={result['score']:.3f} source={result['metadata'].get('source')}")
        print(f"   content preview: {result['content'][:120].replace(chr(10), ' ')}...")

    print("\n=== KnowledgeBaseAgent Test ===")
    
    llm = get_llm_fn()
    agent = KnowledgeBaseAgent(store=store, llm_fn=llm)
    print(f"Question: {query}")
    print("Agent answer:")
    print(agent.answer(query, top_k=5, threshold=threshold))
    return 0


def main() -> int:
    question = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else None
    return run_manual_demo(question=question)


if __name__ == "__main__":
    raise SystemExit(main())
