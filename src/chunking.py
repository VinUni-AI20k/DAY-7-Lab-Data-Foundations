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
        if not text.strip():
            return []
        
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
        chunks = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            chunks.append(" ".join(sentences[i:i + self.max_sentences_per_chunk]))
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
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if not current_text:
            return []
            
        if len(current_text) <= self.chunk_size:
            return [current_text]
            
        if not remaining_separators:
            return [current_text[i:i+self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]
            
        sep = remaining_separators[0]
        splits = current_text.split(sep) if sep != "" else list(current_text)
        
        chunks = []
        current_chunk_parts = []
        current_length = 0
        
        for split in splits:
            if not split and sep != "":
                continue
                
            part_len = len(split)
            sep_len = len(sep) if current_chunk_parts else 0
            
            if current_length + sep_len + part_len <= self.chunk_size:
                current_chunk_parts.append(split)
                current_length += sep_len + part_len
            else:
                if current_chunk_parts:
                    chunks.append(sep.join(current_chunk_parts))
                
                if part_len > self.chunk_size:
                    chunks.extend(self._split(split, remaining_separators[1:]))
                    current_chunk_parts = []
                    current_length = 0
                else:
                    current_chunk_parts = [split]
                    current_length = part_len
                    
        if current_chunk_parts:
            chunks.append(sep.join(current_chunk_parts))
            
        return chunks


class MarkdownChunker:
    """
    Split text into chunks based on Markdown headings (#, ##, ###, etc.)
    Ensures that each section under a header is kept together if possible.
    If a section is too long, uses RecursiveChunker to break it down.
    """

    def __init__(self, chunk_size: int = 500) -> None:
        self.chunk_size = chunk_size
        self._fallback_chunker = RecursiveChunker(chunk_size=chunk_size)

    def chunk(self, text: str) -> list[str]:
        if not text.strip():
            return []
            
        # Split text by Markdown headers (# Heading)
        # ^(#{1,6}\s+.*) matches the header line. We split and capture it.
        parts = re.split(r'^(#{1,6}\s+.*?)$', text, flags=re.MULTILINE)
        
        chunks = []
        current_chunk = parts[0].strip() if parts[0].strip() else ""
        
        for i in range(1, len(parts), 2):
            header = parts[i].strip()
            content = parts[i+1].strip() if i+1 < len(parts) else ""
            section = f"{header}\n{content}".strip()
            
            if current_chunk:
                if len(current_chunk) + len(section) + 2 > self.chunk_size:
                    chunks.append(current_chunk)
                    current_chunk = section
                else:
                    current_chunk += f"\n\n{section}"
            else:
                current_chunk = section
                
        if current_chunk:
            chunks.append(current_chunk)
            
        final_chunks = []
        for c in chunks:
            if len(c) > self.chunk_size:
                final_chunks.extend(self._fallback_chunker.chunk(c))
            else:
                final_chunks.append(c)
                
        return final_chunks

from typing import Callable

class SemanticChunker:
    """
    Groups sentences into chunks based on semantic similarity.
    If the cosine similarity between the current sentence and the next 
    drops below `similarity_threshold`, a new chunk is started.
    """
    def __init__(self, embedder_fn: Callable[[str], list[float]], similarity_threshold: float = 0.5):
        self.embedder_fn = embedder_fn
        self.similarity_threshold = similarity_threshold

    def chunk(self, text: str) -> list[str]:
        if not text.strip():
            return []
            
        # Split text into sentences
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
        if not sentences:
            return []
            
        # Embed sentences to calculate topic shift (this may take time locally depending on HW)
        embeddings = [self.embedder_fn(s) for s in sentences]
        
        chunks = []
        current_chunk = [sentences[0]]
        
        for i in range(1, len(sentences)):
            sim = compute_similarity(embeddings[i-1], embeddings[i])
            if sim >= self.similarity_threshold:
                current_chunk.append(sentences[i])
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i]]
                
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    norm_a = math.sqrt(sum(x*x for x in vec_a))
    norm_b = math.sqrt(sum(x*x for x in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return _dot(vec_a, vec_b) / (norm_a * norm_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        fc = FixedSizeChunker(chunk_size=chunk_size, overlap=20)
        sc = SentenceChunker(max_sentences_per_chunk=3)
        rc = RecursiveChunker(chunk_size=chunk_size)

        res = {}
        for name, chunker in [("fixed_size", fc), ("by_sentences", sc), ("recursive", rc)]:
            chunks = chunker.chunk(text)
            count = len(chunks)
            avg_length = sum(len(c) for c in chunks) / count if count > 0 else 0
            res[name] = {
                "count": count,
                "avg_length": avg_length,
                "chunks": chunks
            }
            
        return res


