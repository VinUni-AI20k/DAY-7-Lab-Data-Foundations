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

    def answer(self, question: str, top_k: int = 3, chat_history: list[dict] | None = None, threshold: float = 0.0) -> str:
        results = self.store.search(question, top_k=top_k, threshold=threshold)
        context_str = "\n".join(f"- {r['content']}" for r in results)
        
        history_str = ""
        if chat_history:
            history_str = "Chat History:\n" + "\n".join(f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history[-4:]) + "\n\n"
            
        prompt = f"""
        Bạn là một trợ lý AI chuyên về thiết bị y tế.

        Nhiệm vụ của bạn:
        - Trả lời câu hỏi dựa CHỈ trên thông tin được cung cấp trong phần CONTEXT.
        - Không được tự suy đoán hoặc thêm thông tin ngoài context.

        Quy tắc quan trọng:
        1. Nếu câu hỏi yêu cầu LIỆT KÊ (ví dụ: "các thiết bị", "tất cả", "danh sách"):
        - Hãy tổng hợp tất cả thiết bị có trong context
        - Trả lời dưới dạng danh sách rõ ràng

        2. Nếu câu hỏi về MỘT thiết bị cụ thể:
        - Trả lời đầy đủ thông tin liên quan đến thiết bị đó

        3. Nếu context KHÔNG đủ thông tin:
        - Trả lời: "Tôi không tìm thấy thông tin phù hợp trong dữ liệu hiện có"

        4. Ưu tiên:
        - Ngắn gọn nhưng đầy đủ
        - Có cấu trúc (bullet points nếu cần)

        CONTEXT:
        {context_str}

        {history_str}

        QUESTION:
        {question}

        ANSWER:
        """
        return self.llm_fn(prompt)
