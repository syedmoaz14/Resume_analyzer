"""Compare scoring methods against human labels.

Metric: for each job, how often does a method rank a higher-labelled resume above a
lower-labelled one (pairwise ranking accuracy, ties excluded). Also Spearman rho over
all pairs. Run:  python evals/run_evals.py [--json out.json]
"""
import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent / "src"))

from resumelab.embed import semantic_coverage  # noqa: E402
from resumelab.score import extract_skills, score, tfidf_similarity  # noqa: E402


def skills_only(resume: str, job: str) -> float:
    j = extract_skills(job)
    return len(extract_skills(resume) & j) / len(j) if j else 0.0


METHODS = {
    "tfidf": tfidf_similarity,
    "skills": skills_only,
    "embeddings": lambda r, j: semantic_coverage(r, j) or 0.0,
    "hybrid": lambda r, j: score(r, j).overall,
}


def spearman(xs: list[float], ys: list[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=v.__getitem__)
        out, i = [0.0] * len(v), 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                out[order[k]] = (i + j) / 2
            i = j + 1
        return out
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="write results to this file")
    args = ap.parse_args()

    data = ROOT / "data"
    resumes = json.loads((data / "resumes.json").read_text(encoding="utf-8"))
    jobs = json.loads((data / "jobs.json").read_text(encoding="utf-8"))
    pairs = json.loads((data / "labels.json").read_text(encoding="utf-8"))["pairs"]

    results = {}
    for name, fn in METHODS.items():
        t = time.perf_counter()
        scores = [fn(resumes[p["resume"]], jobs[p["job"]]) for p in pairs]
        ms = (time.perf_counter() - t) * 1000 / len(pairs)

        right = total = 0
        for job in jobs:
            idx = [i for i, p in enumerate(pairs) if p["job"] == job]
            for a, b in combinations(idx, 2):
                if pairs[a]["label"] == pairs[b]["label"]:
                    continue
                hi, lo = (a, b) if pairs[a]["label"] > pairs[b]["label"] else (b, a)
                total += 1
                right += scores[hi] > scores[lo]
        results[name] = {
            "pairwise_acc": round(right / total, 3),
            "spearman": round(spearman(scores, [p["label"] for p in pairs]), 3),
            "ms_per_pair": round(ms, 1),
        }

    print(f"{'method':<12}{'pairwise acc':>14}{'spearman':>10}{'ms/pair':>10}")
    for name, r in results.items():
        print(f"{name:<12}{r['pairwise_acc']:>14.3f}{r['spearman']:>10.3f}{r['ms_per_pair']:>10.1f}")
    print(f"\n{len(pairs)} labelled pairs, {len(jobs)} jobs, {len(resumes)} resumes (synthetic)")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
