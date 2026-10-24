"""Counterfactual bias audit across scorers.

Usage: python evals/run_bias_audit.py [--llm] [--json out.json]
Without --llm only deterministic scorers run (free, instant). --llm adds the LLM
rater (~12 calls per resume/job pair; mind provider rate limits).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent / "src"))

from resumelab.audit import audit, llm_rate  # noqa: E402
from resumelab.llm import LLMUnavailable  # noqa: E402
from resumelab.score import score  # noqa: E402

PAIRS = [("ml_engineer", "ai_engineer"), ("backend_dev", "python_backend")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--json")
    args = ap.parse_args()

    resumes = json.loads((ROOT / "data/resumes.json").read_text(encoding="utf-8"))
    jobs = json.loads((ROOT / "data/jobs.json").read_text(encoding="utf-8"))
    scorers = {"hybrid (0-100)": lambda r, j: score(r, j).overall * 100}
    if args.llm:
        scorers["llm rater (0-100)"] = llm_rate

    report = {}
    print(f"{'scorer':<20}{'pair':<30}{'baseline':>9}{'spread':>8}  worst variant (delta)")
    for sname, fn in scorers.items():
        for r, j in PAIRS:
            try:
                res = audit(resumes[r], jobs[j], fn)
            except LLMUnavailable as e:
                print(f"{sname:<20}{r + ' x ' + j:<30} skipped: {e}")
                continue
            label, delta = res.worst_delta()
            print(f"{sname:<20}{r + ' x ' + j:<30}{res.baseline:>9.1f}{res.spread:>8.1f}  {label} ({delta:+.1f})")
            report[f"{sname} | {r} x {j}"] = res.scores
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
