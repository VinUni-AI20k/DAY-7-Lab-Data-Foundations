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

        parts = re.split(r"(?<=[.!?])(?:\s+|\n+)", text)
        sentences = [s.strip() for s in parts if s and s.strip()]
        if not sentences:
            return []

        max_per_chunk = self.max_sentences_per_chunk
        chunks: list[str] = []
        for i in range(0, len(sentences), max_per_chunk):
            chunks.append(" ".join(sentences[i : i + max_per_chunk]).strip())
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

        separators = self.separators if self.separators else [""]
        chunk_size = self.chunk_size

        def split_rec(current_text: str, sep_idx: int) -> list[str]:
            if len(current_text) <= chunk_size:
                return [current_text]

            if sep_idx >= len(separators) or separators[sep_idx] == "":
                return [current_text[i : i + chunk_size] for i in range(0, len(current_text), chunk_size)]

            sep = separators[sep_idx]
            parts = current_text.split(sep)
            if len(parts) == 1:
                return split_rec(current_text, sep_idx + 1)

            out: list[str] = []
            current = parts[0]

            for part in parts[1:]:
                candidate = f"{current}{sep}{part}" if current else part
                if len(candidate) <= chunk_size:
                    current = candidate
                else:
                    if current:
                        out.extend(split_rec(current, sep_idx + 1))
                    current = part

            if current:
                out.extend(split_rec(current, sep_idx + 1))

            return out

        return split_rec(text, 0)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        chunk_size = self.chunk_size
        if len(current_text) <= chunk_size:
            return [current_text]

        if not remaining_separators:
            return [current_text[i : i + chunk_size] for i in range(0, len(current_text), chunk_size)]

        sep = remaining_separators[0]
        if sep == "":
            return [current_text[i : i + chunk_size] for i in range(0, len(current_text), chunk_size)]

        parts = current_text.split(sep)
        if len(parts) == 1:
            return self._split(current_text, remaining_separators[1:])

        out: list[str] = []
        current = parts[0]
        next_separators = remaining_separators[1:]
        sep_len = len(sep)

        for part in parts[1:]:
            if len(current) + sep_len + len(part) <= chunk_size:
                current = f"{current}{sep}{part}" if current else part
                continue

            if current:
                out.extend(self._split(current, next_separators))
            current = part

        if current:
            out.extend(self._split(current, next_separators))

        return out


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    dot = 0.0
    norm_a_sq = 0.0
    norm_b_sq = 0.0
    for x, y in zip(vec_a, vec_b):
        dot += x * y
        norm_a_sq += x * x
        norm_b_sq += y * y

    denom = math.sqrt(norm_a_sq) * math.sqrt(norm_b_sq)
    if denom == 0.0:
        return 0.0
    return dot / denom


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size).chunk,
            "by_sentences": SentenceChunker().chunk,
            "recursive": RecursiveChunker(chunk_size=chunk_size).chunk,
        }

        result: dict[str, dict] = {}
        for name, chunk_fn in strategies.items():
            chunks = chunk_fn(text)
            count = len(chunks)
            total_len = sum(len(c) for c in chunks)
            avg_length = (total_len / count) if count else 0.0
            result[name] = {
                "count": count,
                "avg_length": avg_length,
                "chunks": chunks,
            }

        return result
