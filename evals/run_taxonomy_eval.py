"""Does a skill taxonomy add signal beyond corpus terms? (go/no-go experiment)

Each document's sentence vectors are embedded ONCE and cached (evals/.cache/chunks/);
every taxonomy is then scored with a cheap matrix product, so adding a taxonomy is fast.
Taxonomies: O*NET detailed work activities, ESCO skills (evals/data/esco/skills_en.csv).

For a pair, a taxonomy yields 3 features: how strongly the resume covers the job's top-K
taxonomy items (raw and relative to the resume's average), and the profile correlation.
Compared with logistic regression on the held-out test split.

Run: python evals/run_taxonomy_eval.py
"""
import csv
import hashlib
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("RESUMELAB_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent / "src"))
sys.path.insert(0, str(ROOT))

from datasets import load_from_disk  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from resumelab import embed  # noqa: E402
from resumelab.taxonomy import _tax  # noqa: E402
from run_evals import spearman  # noqa: E402
from train_scorer import FEATURES, featurize  # noqa: E402

MODEL_SLUG = re.sub(r"\W+", "_", embed.MODEL)
CACHE = ROOT / ".cache" / MODEL_SLUG
MAX_CHUNKS, MAX_CHARS, TOP_K = 30, 300, 30


def rows_for(split: str, seed: int = 0):
    ds = load_from_disk(str(ROOT / "data/hf_fit"))[split]
    idx = random.Random(seed).sample(range(len(ds)), len(ds))  # same order as train_scorer
    return [ds[i] for i in idx]


def chunk_vectors(text: str) -> np.ndarray:
    (CACHE / "chunks").mkdir(parents=True, exist_ok=True)
    p = CACHE / "chunks" / (hashlib.sha1(text.encode()).hexdigest() + ".npy")
    if p.exists():
        return np.load(p).astype(np.float32)
    parts = [c[:MAX_CHARS] for c in embed.chunks(text)[:MAX_CHUNKS]]
    v = embed._embed(parts) if parts else np.zeros((0, 384), dtype=np.float32)
    np.save(p, v.astype(np.float16))
    return v


def label_matrix(name: str, labels: list[str]) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"labels_{name}.npy"
    if p.exists():
        return np.load(p).astype(np.float32)
    t0 = time.perf_counter()
    m = embed._embed(labels)
    print(f"  embedded {len(labels)} {name} labels in {time.perf_counter() - t0:.0f}s", flush=True)
    np.save(p, m.astype(np.float16))
    return m


def esco_labels() -> list[str]:
    with open(ROOT / "data/esco/skills_en.csv", encoding="utf-8", newline="") as f:
        return [r["preferredLabel"] for r in csv.DictReader(f) if r["preferredLabel"].strip()]


def features(res_v: np.ndarray, job_v: np.ndarray, L: np.ndarray) -> list[float]:
    if len(res_v) == 0 or len(job_v) == 0:
        return [0.0, 0.0, 0.0]
    ar, aj = (res_v @ L.T).max(axis=0), (job_v @ L.T).max(axis=0)
    top = np.argsort(-aj)[:TOP_K]
    corr = np.corrcoef(ar, aj)[0, 1]
    return [float(ar[top].mean() - ar.mean()), float(ar[top].mean()), 0.0 if np.isnan(corr) else float(corr)]


def build(split: str, taxonomies: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    rows = rows_for(split)
    uniq = list(dict.fromkeys([r["resume_text"] for r in rows] + [r["job_description_text"] for r in rows]))
    t0 = time.perf_counter()
    for i, text in enumerate(uniq, 1):
        chunk_vectors(text)
        if i % 25 == 0:
            el = time.perf_counter() - t0
            print(f"  {split}: embedded {i}/{len(uniq)} docs, ~{el / i * (len(uniq) - i):.0f}s left", flush=True)
    out = {}
    for name, L in taxonomies.items():
        out[name] = np.array([features(chunk_vectors(r["resume_text"]), chunk_vectors(r["job_description_text"]), L)
                              for r in rows])
    return out


def evaluate(name, Xtr, ytr, Xte, yte):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    m.fit(Xtr, ytr)
    pred = m.predict(Xte)
    exp = m.predict_proba(Xte) @ np.array([0, 1, 2])
    print(f"{name:<30}{accuracy_score(yte, pred):>6.3f}{f1_score(yte, pred, average='macro'):>10.3f}"
          f"{spearman(list(exp), list(yte)):>10.3f}")


def main() -> None:
    print("embedding model:", embed.MODEL)
    Ctr, ytr = featurize("train", 6241, 0)
    Cte, yte = featurize("test", 1759, 0)
    corp = [FEATURES.index(c) for c in ("weighted_cov", "top_cov", "cosine")]
    Ctr, Cte = Ctr[:, corp], Cte[:, corp]

    taxes = {"onet": label_matrix("onet", _tax()["activities"]),
             "esco": label_matrix("esco", esco_labels())}
    Ttr, Tte = build("train", taxes), build("test", taxes)

    print(f"\ntrain n={len(ytr)} test n={len(yte)}\n{'model':<30}{'acc':>6}{'macro_f1':>10}{'spearman':>10}")
    evaluate("corpus terms (baseline)", Ctr, ytr, Cte, yte)
    for name in taxes:
        evaluate(f"{name} only", Ttr[name], ytr, Tte[name], yte)
        evaluate(f"corpus + {name}", np.hstack([Ctr, Ttr[name]]), ytr, np.hstack([Cte, Tte[name]]), yte)
    evaluate("corpus + onet + esco", np.hstack([Ctr, *Ttr.values()]), ytr, np.hstack([Cte, *Tte.values()]), yte)


if __name__ == "__main__":
    main()
