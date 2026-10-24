"""Evaluate scoring methods on cnamuangtoun/resume-job-description-fit.

Labels: No Fit < Potential Fit < Good Fit. Metrics per method:
  - spearman: rank correlation with the 0/1/2 label
  - auc_good_vs_no: probability a Good Fit pair outscores a No Fit pair
  - mean score per class (a sane method should be monotonic)
Run: python evals/run_hf_evals.py [--split test] [--n 300] [--seed 0]
Download first (see README): load_dataset(...).save_to_disk('evals/data/hf_fit')
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent / "src"))

from datasets import load_from_disk  # noqa: E402

from resumelab.embed import semantic_coverage  # noqa: E402
from resumelab.score import extract_skills, score, tfidf_similarity  # noqa: E402
from run_evals import spearman  # noqa: E402

LABELS = {"No Fit": 0, "Potential Fit": 1, "Good Fit": 2}


def skills_only(resume: str, job: str) -> float:
    j = extract_skills(job)
    return len(extract_skills(resume) & j) / len(j) if j else 0.0


def auc(pos: list[float], neg: list[float]) -> float:
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg)) if pos and neg else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json")
    args = ap.parse_args()

    ds = load_from_disk(str(ROOT / "data/hf_fit"))[args.split]
    idx = random.Random(args.seed).sample(range(len(ds)), min(args.n, len(ds)))
    rows = [ds[i] for i in idx]
    labels = [LABELS[r["label"]] for r in rows]

    # compute each component once; hybrid reuses them via score()
    t0 = time.perf_counter()
    reports = []
    for i, r in enumerate(rows, 1):
        reports.append(score(r["resume_text"], r["job_description_text"]))
        if i % 10 == 0 or i == len(rows):
            el = time.perf_counter() - t0
            print(f"  {i}/{len(rows)} pairs, {el:.0f}s elapsed, ~{el / i * (len(rows) - i):.0f}s left", flush=True)
    t_hybrid = (time.perf_counter() - t0) / len(rows) * 1000
    methods = {
        "tfidf": [r.text_similarity for r in reports],
        "skills": [r.skill_coverage for r in reports],
        "embeddings": [r.semantic if r.semantic is not None else 0.0 for r in reports],
        "hybrid": [r.overall for r in reports],
    }

    results = {}
    print(f"{args.split} split, n={len(rows)}, seed={args.seed}, label counts "
          f"{[labels.count(k) for k in (0, 1, 2)]} (No/Potential/Good)")
    print(f"{'method':<12}{'spearman':>9}{'auc good>no':>13}{'mean no':>9}{'mean pot':>9}{'mean good':>10}")
    for name, s in methods.items():
        by = {k: [v for v, l in zip(s, labels) if l == k] for k in (0, 1, 2)}
        mean = {k: sum(v) / len(v) if v else float("nan") for k, v in by.items()}
        results[name] = {
            "spearman": round(spearman(s, labels), 3),
            "auc_good_vs_no": round(auc(by[2], by[0]), 3),
            "mean_by_class": {k: round(m, 3) for k, m in mean.items()},
        }
        print(f"{name:<12}{results[name]['spearman']:>9.3f}{results[name]['auc_good_vs_no']:>13.3f}"
              f"{mean[0]:>9.3f}{mean[1]:>9.3f}{mean[2]:>10.3f}")
    print(f"\nhybrid cost: {t_hybrid:.0f} ms/pair (incl. embeddings, CPU)")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
