"""PDF parsing with a hidden-text (prompt-injection) shield.

Text spans that a human cannot see (white-on-white, tiny font, off-page) are
separated from visible text so downstream scoring/LLM calls only see the latter.
"""
from dataclasses import dataclass, field

import pymupdf as fitz

MIN_VISIBLE_FONT_PT = 4.0
WHITE = 0xFFFFFF


@dataclass
class ParsedResume:
    visible_text: str
    hidden_text: str = ""
    findings: list[str] = field(default_factory=list)

    @property
    def has_hidden_content(self) -> bool:
        return bool(self.hidden_text.strip())


def _span_hidden_reason(span: dict, page_rect: fitz.Rect) -> str | None:
    if span["color"] == WHITE:
        return "white text"
    if span["size"] < MIN_VISIBLE_FONT_PT:
        return f"tiny font ({span['size']:.1f}pt)"
    if not fitz.Rect(span["bbox"]).intersects(page_rect):
        return "text outside page"
    return None


def parse_pdf(data: bytes) -> ParsedResume:
    visible, hidden, findings = [], [], []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for page_no, page in enumerate(doc, start=1):
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if not text:
                            continue
                        reason = _span_hidden_reason(span, page.rect)
                        if reason:
                            hidden.append(text)
                            findings.append(f"page {page_no}: {reason}: {text[:80]!r}")
                        else:
                            visible.append(text)
    return ParsedResume(" ".join(visible), " ".join(hidden), findings)
