"""CI regression check: score the held-out benchmark with the COMMITTED model files.

Fails (exit 1) if macro-F1 on the test split drops below --min-macro-f1, so a change to
the term model, bucket model or feature code cannot silently make the scorer worse.

Run: python evals/check_regression.py [--min-macro-f1 0.37]
Needs: pip install -r requirements-evals.txt (datasets); downloads the benchmark if absent.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent / "src"))

from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

from resumelab import bucket, terms  # noqa: E402

LABELS = {"No Fit": "Unlikely fit", "Potential Fit": "Possible fit", "Good Fit": "Likely fit"}
DATASET = "cnamuangtoun/resume-job-description-fit"


def load_test():
    local = ROOT / "data" / "hf_fit"
    if local.exists():
        from datasets import load_from_disk
        return load_from_disk(str(local))["test"]
    from datasets import load_dataset
    return load_dataset(DATASET, split="test")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-macro-f1", type=float, default=0.37)
    args = ap.parse_args()

    test = load_test()
    t0 = time.perf_counter()
    y_true, y_pred = [], []
    for r in test:
        feats = terms.term_features(r["resume_text"], r["job_description_text"])
        y_pred.append(bucket.predict(feats).label)
        y_true.append(LABELS[r["label"]])
    f1 = f1_score(y_true, y_pred, average="macro")
    acc = accuracy_score(y_true, y_pred)
    print(f"{len(y_true)} test pairs in {time.perf_counter() - t0:.0f}s | "
          f"macro-F1 {f1:.3f} (min {args.min_macro_f1}) | accuracy {acc:.3f}")
    if f1 < args.min_macro_f1:
        print("FAIL: macro-F1 below threshold")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
