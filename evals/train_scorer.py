"""Learn the fit scorer from labelled data instead of guessing weights.

Features come from resumelab.score (skills, TF-IDF, embeddings); a small multinomial
logistic regression is fitted on the TRAIN split and evaluated on the held-out TEST
split, against baselines. Features are cached on disk, so reruns are cheap.

Run: python evals/train_scorer.py [--n-train 400] [--n-test 150] [--seed 0]
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent / "src"))
sys.path.insert(0, str(ROOT))

from datasets import load_from_disk  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402

from resumelab.score import score  # noqa: E402
from resumelab.terms import term_features  # noqa: E402
from run_evals import spearman  # noqa: E402

LABELS = {"No Fit": 0, "Potential Fit": 1, "Good Fit": 2}
FEATURES = ["coverage", "n_matched", "n_missing", "tfidf", "weighted_cov", "top_cov", "cosine"]
SETS = {  # which feature columns each candidate model may use
    "tech skills only": ["coverage", "n_matched", "n_missing"],
    "2-doc tfidf only": ["tfidf"],
    "corpus terms only": ["weighted_cov", "top_cov", "cosine"],
    "all (no embeddings)": FEATURES,
}
CACHE = ROOT / ".cache"


def featurize(split: str, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    path = CACHE / f"features_v2_{split}_{n}_{seed}.json"
    if path.exists():
        d = json.loads(path.read_text())
        return np.array(d["X"]), np.array(d["y"])
    ds = load_from_disk(str(ROOT / "data/hf_fit"))[split]
    idx = random.Random(seed).sample(range(len(ds)), min(n, len(ds)))
    X, y, t0 = [], [], time.perf_counter()
    for i, k in enumerate(idx, 1):
        r = ds[k]
        s = score(r["resume_text"], r["job_description_text"], use_embeddings=False)
        t = term_features(r["resume_text"], r["job_description_text"])
        X.append([s.skill_coverage, len(s.matched), len(s.missing), s.text_similarity,
                  t.weighted_cov, t.top_cov, t.cosine])
        y.append(LABELS[r["label"]])
        if i % 20 == 0:
            el = time.perf_counter() - t0
            print(f"  {split}: {i}/{len(idx)}, ~{el / i * (len(idx) - i):.0f}s left", flush=True)
    CACHE.mkdir(exist_ok=True)
    path.write_text(json.dumps({"X": X, "y": y}))
    return np.array(X), np.array(y)


def report(name, model, Xtr, ytr, Xte, yte, cols):
    model.fit(Xtr[:, cols], ytr)
    pred = model.predict(Xte[:, cols])
    out = {"acc": accuracy_score(yte, pred), "macro_f1": f1_score(yte, pred, average="macro")}
    if hasattr(model, "predict_proba"):
        exp = model.predict_proba(Xte[:, cols]) @ np.array([0, 1, 2])
        out["spearman"] = spearman(list(exp), list(yte))
    print(f"{name:<18}{out['acc']:>6.3f}{out['macro_f1']:>10.3f}{out.get('spearman', float('nan')):>10.3f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=6241)
    ap.add_argument("--n-test", type=int, default=1759)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    Xtr, ytr = featurize("train", args.n_train, args.seed)
    Xte, yte = featurize("test", args.n_test, args.seed)
    print(f"train n={len(ytr)} test n={len(yte)}; test class counts {np.bincount(yte).tolist()}")
    print(f"{'model':<18}{'acc':>6}{'macro_f1':>10}{'spearman':>10}")

    results = {"majority": report("majority class", DummyClassifier(strategy="most_frequent"),
                                  Xtr, ytr, Xte, yte, list(range(len(FEATURES))))}
    for name, cols in SETS.items():
        idx = [FEATURES.index(c) for c in cols]
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
        results[name] = report(name, lr, Xtr, ytr, Xte, yte, idx)
        coefs = lr[-1].coef_[2] - lr[-1].coef_[0]  # Good vs No direction
        print("   good-vs-no weights:", {c: round(float(w), 2) for c, w in zip(cols, coefs)})
    (ROOT / "learned_results.json").write_text(
        json.dumps({k: {m: round(float(v), 3) for m, v in r.items()} for k, r in results.items()}, indent=2))


if __name__ == "__main__":
    main()
