from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv

from src.agent import KnowledgeBaseAgent
from src.store import EmbeddingStore
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from main import get_llm_fn

# Global scope variables
store = None
agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global store, agent
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

    collection_name = os.getenv("COLLECTION_NAME", "my_rag_db")
    store = EmbeddingStore(collection_name=collection_name, embedding_fn=embedder)
    
    # Init LLM handler from main
    llm = get_llm_fn()
    agent = KnowledgeBaseAgent(store=store, llm_fn=llm)
    print(f"API Initialized. Target VectorDB: '{collection_name}' with {store.get_collection_size()} chunks.")
    yield

app = FastAPI(title="RAG Knowledge Base API", lifespan=lifespan)

class AskRequest(BaseModel):
    question: str
    top_k: int = 50
    session_id: str = "default"
    threshold: float = None

# Global memory storage format: { session_id: [{"role": "user", "content": "..."}, ...] }
chat_sessions = {}

@app.post("/ask")
def ask_question(req: AskRequest):
    if not agent:
        raise HTTPException(status_code=500, detail="RAG Agent not initialized correctly.")
        
    if req.session_id not in chat_sessions:
        chat_sessions[req.session_id] = []
        
    history = chat_sessions[req.session_id]
    
    try:
        env_threshold = float(os.getenv("VEC_SIM_THREDHOLD", "0.0"))
        used_threshold = req.threshold if req.threshold is not None else env_threshold
        
        answer = agent.answer(req.question, top_k=req.top_k, chat_history=history, threshold=used_threshold)
        
        history.append({"role": "user", "content": req.question})
        history.append({"role": "assistant", "content": answer})
        
        if len(history) > 6:
            chat_sessions[req.session_id] = history[-6:]
            
        return {
            "session_id": req.session_id,
            "question": req.question,
            "answer": answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM or Store Error: {str(e)}")
        
@app.get("/health")
def health_check():
    return {
        "status": "ok", 
        "collection_size": store.get_collection_size() if store else 0
    }
