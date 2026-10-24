"""Fit corpus IDF on the TRAIN split only (never the test split) and save it.

Run: python evals/build_terms.py
Output: src/resumelab/data/terms.json (term list + idf; aggregate statistics only,
no resume text is stored).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT.parent / "src" / "resumelab" / "data" / "terms.json"

from datasets import load_from_disk  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402


def main() -> None:
    train = load_from_disk(str(ROOT / "data/hf_fit"))["train"]
    # unique texts: the dataset repeats the same job posts many times
    docs = list(dict.fromkeys(train["resume_text"])) + list(dict.fromkeys(train["job_description_text"]))
    print(f"{len(docs)} unique documents")
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=5, max_df=0.5,
                          max_features=60000, lowercase=True,
                          token_pattern=r"(?u)\b[\w+#.]*\w[\w+#]*\b")
    vec.fit(docs)
    terms = vec.get_feature_names_out().tolist()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"terms": terms, "idf": [round(float(x), 3) for x in vec.idf_]}),
                   encoding="utf-8")
    print(f"saved {len(terms)} terms to {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
