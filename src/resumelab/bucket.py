"""Coarse fit bucket (Unlikely / Possible / Likely) from corpus-term features.

A percentage would claim accuracy the benchmark does not support (macro-F1 ~0.40), so the
app reports a bucket plus the model's class probabilities.
"""
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .terms import TermFeatures

DATA = Path(__file__).parent / "data" / "bucket_model.json"


@dataclass
class Bucket:
    label: str
    probabilities: dict[str, float]


@lru_cache(maxsize=1)
def _model() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))


def available() -> bool:
    return DATA.exists()


def predict(t: TermFeatures) -> Bucket:
    m = _model()
    raw = {"weighted_cov": t.weighted_cov, "top_cov": t.top_cov, "cosine": t.cosine}
    x = (np.array([raw[f] for f in m["features"]]) - m["mean"]) / m["scale"]
    logits = np.array(m["coef"]) @ x + np.array(m["intercept"])
    p = np.exp(logits - logits.max())
    p /= p.sum()
    probs = dict(zip(m["classes"], (round(float(v), 3) for v in p)))
    return Bucket(max(probs, key=probs.get), probs)
