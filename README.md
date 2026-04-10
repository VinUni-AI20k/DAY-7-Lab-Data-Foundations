# Day 7 - Embedding, Vector Store, and RAG Basics

This repository is a Day 7 lab about:
- text chunking
- embeddings
- vector stores
- retrieval-augmented generation (RAG)

The codebase is already in a working state, not a TODO skeleton.

## Quick Start

```bash
pip install -r requirements.txt
pytest tests/ -v
python main.py
```

`main.py` runs a manual demo using sample files from `data/`.

## Embedding Backends

By default, the repo uses a mock embedder, so no API key or external model is required.

If you want to try other backends:
- local: install `sentence-transformers` and set `EMBEDDING_PROVIDER=local`
- OpenAI: install `openai`, set `EMBEDDING_PROVIDER=openai`, and provide `OPENAI_API_KEY`

## Main Files

- `src/chunking.py`: chunking strategies and similarity
- `src/store.py`: `EmbeddingStore`
- `src/agent.py`: `KnowledgeBaseAgent`
- `src/embeddings.py`: mock, local, and OpenAI embedders
- `tests/test_solution.py`: test suite
- `exercises.md`: lab exercises
- `docs/SCORING.md`: grading rubric
- `docs/EVALUATION.md`: retrieval evaluation criteria
- `report/REPORT.md`: report

## Notes

- The repo currently passes the test suite.
- If you are studying the lab, a good reading order is: `exercises.md` -> `tests/test_solution.py` -> `src/` -> `main.py`.
- If you need detailed assignment or grading information, read `docs/` instead of expanding the README.
