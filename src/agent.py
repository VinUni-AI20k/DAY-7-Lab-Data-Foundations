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
        results = self.store.search(question, top_k=top_k)
        if results:
            context_lines = []
            for index, result in enumerate(results, start=1):
                context_lines.append(f"[{index}] {result['content']}")
            context = "\n\n".join(context_lines)
        else:
            context = "No relevant context was retrieved from the knowledge base."

        prompt = (
            "You are a helpful assistant answering questions with retrieved knowledge.\n"
            "Use the context below to answer the question. If the context is insufficient, say so clearly.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        return self.llm_fn(prompt)
