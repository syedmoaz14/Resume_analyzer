"""Skill taxonomy (O*NET) linking: technologies by name, work activities by embedding.

Technologies: whole-word, case-insensitive name match -> product + category.
Activities:   resume/job sentences are embedded and linked to the nearest O*NET
              Detailed Work Activity (see activity_profile).
"""
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .embed import _embed, chunks

DATA = Path(__file__).parent / "data" / "taxonomy.json"
MAX_CHUNKS = 40  # per document; bounds CPU time


@lru_cache(maxsize=1)
def _tax() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))


_WORD = re.compile(r"[\w+#]+(?:[.\-/][\w+#]+)*")


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


@lru_cache(maxsize=1)
def _tech_index():
    """Lowercased, word-normalised product name -> TechHit, plus the longest name in words."""
    index, longest = {}, 1
    for t in _tax()["tech"]:
        words = _words(t["name"])
        if words:
            index[" ".join(words)] = TechHit(t["name"], t["category"])
            longest = max(longest, len(words))
    return index, longest


@dataclass(frozen=True)
class TechHit:
    name: str
    category: str


@lru_cache(maxsize=4096)
def tech_hits(text: str) -> frozenset[TechHit]:
    """Products named in the text (whole-word n-gram lookup; fast on long documents)."""
    index, longest = _tech_index()
    words, hits = _words(text), set()
    for i in range(len(words)):
        for n in range(1, longest + 1):
            hit = index.get(" ".join(words[i:i + n]))
            if hit:
                hits.add(hit)
    return frozenset(hits)


@lru_cache(maxsize=1)
def _activity_matrix() -> np.ndarray:
    return _embed(_tax()["activities"])


def activity_profile(text: str) -> np.ndarray:
    """For each O*NET activity: the best cosine similarity to any sentence of the text."""
    parts = chunks(text)[:MAX_CHUNKS]
    if not parts:
        return np.zeros(len(_tax()["activities"]), dtype=np.float32)
    return (_embed(parts) @ _activity_matrix().T).max(axis=0).astype(np.float32)
