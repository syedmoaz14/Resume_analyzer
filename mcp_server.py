"""MCP server: lets Claude Desktop / Claude Code / any MCP client call ResumeLab as tools.

Runs over stdio on your machine:  python mcp_server.py
Never print to stdout here: stdout carries the MCP protocol.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from mcp.server.mcpserver import MCPServer  # noqa: E402
from mcp.types import ToolAnnotations  # noqa: E402
from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402

from resumelab import bucket, terms  # noqa: E402
from resumelab.explain import explain  # noqa: E402
from resumelab.llm import LLMUnavailable  # noqa: E402
from resumelab.parse import ParsedResume, parse_pdf  # noqa: E402

MAX_PDF_BYTES = 10_000_000
MAX_TEXT_CHARS = 200_000
SHOWN_TERMS = 20

mcp = MCPServer(
    "resumelab",
    instructions=(
        "Resume-vs-job matching for any profession. Give a resume as a local PDF path "
        "(resume_pdf_path) or as plain text (resume_text), plus the job description. "
        "The fit label is a rough estimate (benchmark macro-F1 ~0.40), not a hiring decision."),
)

LOCAL_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=False)


def _load_resume(resume_pdf_path: str, resume_text: str) -> ParsedResume:
    """Read a resume from a PDF path or raw text. Raises ToolError with a clear message."""
    if bool(resume_pdf_path) == bool(resume_text):
        raise ToolError("Provide exactly one of resume_pdf_path or resume_text.")
    if resume_text:
        if len(resume_text) > MAX_TEXT_CHARS:
            raise ToolError(f"resume_text is too long (max {MAX_TEXT_CHARS} characters).")
        return ParsedResume(visible_text=resume_text)

    path = Path(resume_pdf_path).expanduser().resolve()
    if path.suffix.lower() != ".pdf":
        raise ToolError("resume_pdf_path must point to a .pdf file.")
    if not path.is_file():
        raise ToolError(f"No such file: {path}")
    if path.stat().st_size > MAX_PDF_BYTES:
        raise ToolError("PDF is larger than 10 MB.")
    parsed = parse_pdf(path.read_bytes())
    if not parsed.visible_text.strip():
        raise ToolError("No readable text in the PDF (it may be a scan or image-only).")
    return parsed


@mcp.tool(annotations=LOCAL_ONLY)
def analyze_fit(job_description: str, resume_pdf_path: str = "", resume_text: str = "") -> dict:
    """Estimate how well a resume fits a job (Unlikely / Possible / Likely) and list the job
    keywords the resume matches and misses. Runs locally; nothing is sent anywhere."""
    resume = _load_resume(resume_pdf_path, resume_text)
    if not job_description.strip():
        raise ToolError("job_description is empty.")
    if not (terms.available() and bucket.available()):
        raise ToolError("Scoring model files are missing (src/resumelab/data).")
    feats = terms.term_features(resume.visible_text, job_description)
    result = bucket.predict(feats)
    return {
        "fit": result.label,
        "probabilities": result.probabilities,
        "matched_keywords": feats.matched_top[:SHOWN_TERMS],
        "missing_keywords": feats.missing_top[:SHOWN_TERMS],
        "hidden_text_found": resume.has_hidden_content,
        "note": "Rough estimate from a noisy benchmark (macro-F1 ~0.40); not a hiring decision.",
    }


@mcp.tool(annotations=LOCAL_ONLY)
def check_hidden_text(resume_pdf_path: str) -> dict:
    """Detect text a human can't see in a PDF resume (white, tiny or off-page text), which can
    be used to stuff keywords or inject instructions into AI screeners. Runs locally."""
    resume = _load_resume(resume_pdf_path, "")
    return {"hidden_text_found": resume.has_hidden_content, "findings": resume.findings}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True))
def ai_feedback(job_description: str, resume_pdf_path: str = "", resume_text: str = "") -> dict:
    """Strengths, gaps and suggestions written by an LLM, fact-checked against the resume
    (invented quotes and false gaps are removed). NOTE: the resume and job text are sent to the
    configured LLM provider (LLM_BASE_URL); requires LLM_API_KEY / LLM_MODEL to be set."""
    resume = _load_resume(resume_pdf_path, resume_text)
    if not job_description.strip():
        raise ToolError("job_description is empty.")
    try:
        feedback, removed = explain(resume.visible_text, job_description)
    except LLMUnavailable as e:
        raise ToolError(f"AI feedback is unavailable: {e}") from e
    return {**feedback.model_dump(), "claims_removed_by_fact_check": removed}


if __name__ == "__main__":
    mcp.run()
