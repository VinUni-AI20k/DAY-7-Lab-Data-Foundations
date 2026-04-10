from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        # Split into sentences using punctuation followed by whitespace or a dot+newline
        parts = re.split(r'(?<=[.!?])\s+|\.\n', text)
        sentences = [p.strip() for p in parts if p and p.strip()]

        chunks: list[str] = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            chunk = " ".join(sentences[i : i + self.max_sentences_per_chunk]).strip()
            if chunk:
                chunks.append(chunk)
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        return self._split(text, list(self.separators))

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        current_text = current_text.strip()
        if not current_text:
            return []

        # If already small enough, return as single chunk
        if len(current_text) <= self.chunk_size:
            return [current_text]

        # If no more separators, fall back to fixed-size character splits
        if not remaining_separators:
            out: list[str] = []
            for i in range(0, len(current_text), self.chunk_size):
                out.append(current_text[i : i + self.chunk_size].strip())
            return out

        sep = remaining_separators[0]
        next_seps = remaining_separators[1:]

        # Empty separator means treat as character fallback
        if sep == "":
            return self._split(current_text, next_seps)

        pieces = current_text.split(sep)
        output: list[str] = []
        for p in pieces:
            p = p.strip()
            if not p:
                continue
            if len(p) <= self.chunk_size:
                output.append(p)
            else:
                # Recurse with the remaining separators to try finer splits
                output.extend(self._split(p, next_seps))

        return output


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    if not vec_a or not vec_b:
        return 0.0
    dot = _dot(vec_a, vec_b)
    norm_a = math.sqrt(sum(x * x for x in vec_a))
    norm_b = math.sqrt(sum(x * x for x in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        result: dict[str, dict] = {}

        # Fixed size
        fixed = FixedSizeChunker(chunk_size=chunk_size, overlap=max(0, chunk_size // 10))
        fixed_chunks = fixed.chunk(text)
        fixed_count = len(fixed_chunks)
        fixed_avg = sum(len(c) for c in fixed_chunks) / fixed_count if fixed_count else 0
        result['fixed_size'] = {'count': fixed_count, 'avg_length': fixed_avg, 'chunks': fixed_chunks}

        # By sentences
        max_sent = max(1, chunk_size // 50)
        sent = SentenceChunker(max_sentences_per_chunk=max_sent)
        sent_chunks = sent.chunk(text)
        sent_count = len(sent_chunks)
        sent_avg = sum(len(c) for c in sent_chunks) / sent_count if sent_count else 0
        result['by_sentences'] = {'count': sent_count, 'avg_length': sent_avg, 'chunks': sent_chunks}

        # Recursive
        rec = RecursiveChunker(chunk_size=chunk_size)
        rec_chunks = rec.chunk(text)
        rec_count = len(rec_chunks)
        rec_avg = sum(len(c) for c in rec_chunks) / rec_count if rec_count else 0
        result['recursive'] = {'count': rec_count, 'avg_length': rec_avg, 'chunks': rec_chunks}

        return result
