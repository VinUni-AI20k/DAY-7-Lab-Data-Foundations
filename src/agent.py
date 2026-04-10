from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        # Retrieve top-k contexts
        candidates = self.store.search(question, top_k=top_k)
        contexts = []
        for c in candidates:
            content = c.get('content') or ''
            contexts.append(content)

        prompt_parts = ["Use the following retrieved context to answer the question.", "\nContext:"]
        prompt_parts.append("\n---\n".join(contexts) if contexts else "(no context)")
        prompt_parts.append(f"\n\nQuestion: {question}\nAnswer:")
        prompt = "\n".join(prompt_parts)

        if not callable(self.llm_fn):
            return ""
        return self.llm_fn(prompt)
