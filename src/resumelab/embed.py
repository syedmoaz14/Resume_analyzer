"""Semantic matching with small local ONNX embeddings (no torch, CPU-friendly).

For each requirement line in the job description, find the best-matching resume
chunk; the score is the mean of those best similarities. This answers "is each
thing the job asks for covered somewhere?" rather than "do the documents look alike?".
"""
import os
import re
from functools import lru_cache

import numpy as np

MODEL = os.environ.get("RESUMELAB_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
MIN_CHARS = 20
MAX_CHUNKS = 120


def chunks(text: str) -> list[str]:
    """Split into bullet/sentence-sized pieces; drop fragments and cap the count."""
    parts = re.split(r"[\n•·▪]+|(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) >= MIN_CHARS][:MAX_CHUNKS]


@lru_cache(maxsize=1)
def _model():
    from fastembed import TextEmbedding
    return TextEmbedding(MODEL)


_cache: dict[str, np.ndarray] = {}  # chunk text -> unit vector; job posts repeat a lot


def _embed(texts: list[str]) -> np.ndarray:
    todo = [t for t in dict.fromkeys(texts) if t not in _cache]
    if todo:
        vecs = np.array(list(_model().embed(todo)), dtype=np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        _cache.update(zip(todo, vecs))
        if len(_cache) > 50_000:  # bound memory in long-running servers
            for k in list(_cache)[:10_000]:
                del _cache[k]
    return np.stack([_cache[t] for t in texts])


def semantic_coverage(resume: str, job: str) -> float | None:
    """Mean best-match cosine of each job line vs resume chunks, rescaled to 0..1.
    Returns None if embeddings are unavailable or there is nothing to compare."""
    r_chunks, j_chunks = chunks(resume), chunks(job)
    if not r_chunks or not j_chunks:
        return None
    try:
        sims = _embed(j_chunks) @ _embed(r_chunks).T
    except Exception:  # fastembed missing / model download failed
        return None
    best = sims.max(axis=1)
    # bge-small cosine for unrelated text sits ~0.4-0.5 and related ~0.7-0.85
    return float(np.clip((best.mean() - 0.45) / 0.35, 0.0, 1.0))
