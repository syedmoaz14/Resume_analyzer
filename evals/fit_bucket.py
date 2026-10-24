"""Fit the fit-bucket model on the TRAIN split and save its weights as JSON.

Uses the 3 corpus-term features (see resumelab.terms). Weights are exported as plain
numbers so the app needs only numpy, not a pickled scikit-learn object.
Run after build_terms.py and train_scorer.py (which caches the features).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent / "src"))
sys.path.insert(0, str(ROOT))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from train_scorer import FEATURES, featurize  # noqa: E402

COLS = ["weighted_cov", "top_cov", "cosine"]
OUT = ROOT.parent / "src" / "resumelab" / "data" / "bucket_model.json"


def main() -> None:
    X, y = featurize("train", 6241, 0)
    X = X[:, [FEATURES.index(c) for c in COLS]]
    scaler = StandardScaler().fit(X)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(scaler.transform(X), y)
    OUT.write_text(json.dumps({
        "features": COLS,
        "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
        "coef": lr.coef_.tolist(), "intercept": lr.intercept_.tolist(),
        "classes": ["Unlikely fit", "Possible fit", "Likely fit"],
        "note": "Trained on cnamuangtoun/resume-job-description-fit train split; "
                "benchmark macro-F1 ~0.40 (see docs/EXPERIMENTS.md).",
    }, indent=1), encoding="utf-8")
    print("saved", OUT, "| train acc", round(lr.score(scaler.transform(X), y), 3))


if __name__ == "__main__":
    main()
