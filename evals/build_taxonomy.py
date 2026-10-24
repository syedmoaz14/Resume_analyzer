"""Build the skill taxonomy from the O*NET 31.0 text database (CC BY 4.0, USDOL/ETA).

Input:  evals/data/onet/extracted/*.txt   (download: onetcenter.org/dl_files/database/db_31_0_text.zip)
Output: src/resumelab/data/taxonomy.json
Nodes:  technologies (product -> category) and detailed work activities (sentences).
"""
import csv
import json
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

ROOT = Path(__file__).parent
SRC = ROOT / "data" / "onet" / "extracted"
OUT = ROOT.parent / "src" / "resumelab" / "data" / "taxonomy.json"
MIN_NAME_LEN = 4  # short names ("R", "C", "Go") are too ambiguous for string matching


def read(name: str) -> list[dict]:
    path = next(SRC.rglob(name))
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main() -> None:
    tech: dict[str, str] = {}
    for r in read("Software Skills.txt"):
        name = r["Workplace Example"].strip()
        if len(name) >= MIN_NAME_LEN and name.lower() not in ENGLISH_STOP_WORDS:
            tech.setdefault(name, r["Element Name"])

    activities = sorted({r["DWA Element Name"].strip() for r in read("GWAs to IWAs to DWAs.txt")})
    knowledge = sorted({r["Element Name"] for r in read("Knowledge.txt")})
    skills = sorted({r["Element Name"] for r in read("Essential Skills.txt")})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "source": "O*NET 31.0 Database by the U.S. Department of Labor, Employment and Training "
                  "Administration (USDOL/ETA), CC BY 4.0. Modified: deduplicated and filtered.",
        "tech": [{"name": n, "category": c} for n, c in sorted(tech.items())],
        "activities": activities, "knowledge": knowledge, "skills": skills,
    }, ensure_ascii=False), encoding="utf-8")
    print(f"tech {len(tech)} | activities {len(activities)} | knowledge {len(knowledge)} | "
          f"skills {len(skills)} -> {OUT} ({OUT.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
