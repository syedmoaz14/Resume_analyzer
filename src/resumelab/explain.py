"""LLM-written feedback whose claims are verified against the resume text.

Two guards, because small models err in both directions:
- strengths: evidence quotes must appear verbatim in the resume (else dropped)
- gaps: the missing `requirement` must really be absent from the resume (else dropped)
"""
import re

from pydantic import BaseModel, Field

from .llm import complete_json

PROMPT_VERSION = "explain-v2"
MAX_RESUME_CHARS = 14000  # long resumes exist; truncating causes false "gaps"
MAX_JOB_CHARS = 4000

SYSTEM = (
    "You are a precise resume reviewer. Compare the RESUME to the JOB. "
    "Use only facts present in the resume; never invent experience. "
    "Strengths: 'evidence' must be a SHORT exact quote (under 15 words) copied verbatim "
    "from the resume. "
    "Gaps: 'requirement' is the 1-3 word skill or topic the job wants that the resume "
    "does not contain; only list it if you searched the whole resume and found nothing. "
    "Keep each item under 25 words."
)


class Strength(BaseModel):
    text: str
    evidence: str = ""


class Gap(BaseModel):
    text: str
    requirement: str = ""


class Feedback(BaseModel):
    strengths: list[Strength] = Field(max_length=3)
    gaps: list[Gap] = Field(max_length=3)
    suggestions: list[str] = Field(max_length=3)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[-_/]", " ", s)).strip().lower()


def _present(requirement: str, haystack: str) -> bool:
    """True if every word of the requirement appears in the resume (loose stem match)."""
    words = [w for w in _norm(requirement).split() if len(w) > 2]
    return bool(words) and all(w[:5] in haystack for w in words)


def verify(fb: Feedback, resume: str) -> tuple[Feedback, int]:
    """Blank unverifiable quotes and drop false gaps; return (feedback, removed_count)."""
    haystack, removed = _norm(resume), 0
    for s in fb.strengths:
        if s.evidence and _norm(s.evidence) not in haystack:
            s.evidence, removed = "", removed + 1
    kept = []
    for g in fb.gaps:
        if g.requirement and _present(g.requirement, haystack):
            removed += 1
        else:
            kept.append(g)
    fb.gaps = kept
    return fb, removed


def explain(resume: str, job: str) -> tuple[Feedback, int]:
    user = f"RESUME:\n{resume[:MAX_RESUME_CHARS]}\n\nJOB:\n{job[:MAX_JOB_CHARS]}"
    return verify(complete_json(Feedback, SYSTEM, user, PROMPT_VERSION), resume)
