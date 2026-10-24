"""Domain-agnostic term weighting learned from a corpus (no hand-written skill list).

IDF is fitted offline on many resumes + job posts (see evals/build_terms.py), so rare,
specific terms ("servsafe", "gaap", "triage") weigh more than filler ("team", "work").
"""
import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

DATA = Path(__file__).parent / "data" / "terms.json"
TOP_JOB_TERMS = 40


@dataclass
class TermFeatures:
    weighted_cov: float  # idf-weighted share of ALL job terms found in the resume
    top_cov: float  # same, over the job's most important terms only
    cosine: float  # cosine of idf-weighted vectors (corpus IDF, not 2-doc IDF)
    matched_top: list[str]
    missing_top: list[str]


@lru_cache(maxsize=1)
def _load():
    d = json.loads(DATA.read_text(encoding="utf-8"))
    vocab = {t: i for i, t in enumerate(d["terms"])}
    cv = CountVectorizer(vocabulary=vocab, ngram_range=(1, 2), stop_words="english",
                         token_pattern=r"(?u)\b[\w+#.]*\w[\w+#]*\b")
    return cv, np.array(d["idf"], dtype=np.float32), d["terms"]


def available() -> bool:
    return DATA.exists()


def _tf(cv, text: str) -> np.ndarray:
    return cv.transform([text.lower()]).toarray()[0].astype(np.float32)


def _dedupe(terms: list[str]) -> list[str]:
    """Drop a single word when a longer phrase in the list already contains it."""
    phrases = [t for t in terms if " " in t]
    return [t for t in terms if " " in t or not any(t in p.split() for p in phrases)]


def term_features(resume: str, job: str) -> TermFeatures:
    cv, idf, terms = _load()
    r, j = _tf(cv, resume), _tf(cv, job)
    jw = np.where(j > 0, (1 + np.log(np.maximum(j, 1))) * idf, 0.0)
    rw = np.where(r > 0, (1 + np.log(np.maximum(r, 1))) * idf, 0.0)
    present = r > 0

    in_job = jw > 0
    total = jw.sum()
    weighted_cov = float(jw[present & in_job].sum() / total) if total else 0.0

    order = np.argsort(-jw)[:TOP_JOB_TERMS]
    order = order[jw[order] > 0]
    top_total = jw[order].sum()
    top_cov = float(jw[order][present[order]].sum() / top_total) if top_total else 0.0
    matched = [terms[i] for i in order if present[i]]
    missing = [terms[i] for i in order if not present[i]]

    denom = math.sqrt(float(rw @ rw) * float(jw @ jw))
    cosine = float(rw @ jw / denom) if denom else 0.0
    return TermFeatures(weighted_cov, top_cov, cosine, _dedupe(matched), _dedupe(missing))
