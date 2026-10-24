import os
import sys
from datetime import date
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from resumelab import bucket, terms  # noqa: E402
from resumelab.explain import explain  # noqa: E402
from resumelab.llm import LLMUnavailable  # noqa: E402
from resumelab.parse import parse_pdf  # noqa: E402

MAX_LLM_CALLS_PER_SESSION = 5
MAX_LLM_CALLS_PER_DAY = int(os.environ.get("MAX_LLM_CALLS_PER_DAY", "150"))  # app-wide, protects free quota
SHOWN_TERMS = 15

# Hosted deploys: put LLM_BASE_URL / LLM_API_KEY / LLM_MODEL in Streamlit secrets.
try:
    for _k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        if _k in st.secrets:
            os.environ[_k] = st.secrets[_k]
except Exception:  # no secrets file locally
    pass

@st.cache_resource
def _daily_budget() -> dict:
    """Process-wide AI-call counter (resets daily, and when the app restarts)."""
    return {"day": date.today(), "used": 0}


def _budget_left() -> bool:
    b = _daily_budget()
    if b["day"] != date.today():
        b["day"], b["used"] = date.today(), 0
    return b["used"] < MAX_LLM_CALLS_PER_DAY


st.set_page_config(page_title="ResumeLab", page_icon="📄")
st.title("📄 ResumeLab")
st.markdown("Explainable resume-to-job matching for any profession, with a hidden-text safety check.")
st.caption("🔒 The fit estimate runs inside this app. If you click **Generate AI feedback**, your resume "
           "and job text are sent to a third-party LLM provider. Please use sample or non-sensitive "
           "resumes on this public demo.")

uploaded_file = st.file_uploader("📤 Upload Resume (PDF)", type=["pdf"])
job_input = st.text_area("💼 Paste Job Description Here", height=200)

if uploaded_file and job_input.strip():
    parsed = parse_pdf(uploaded_file.read())

    if not parsed.visible_text.strip():
        st.error("No readable text found. The PDF may be a scan or image-only.")
        st.stop()

    if parsed.has_hidden_content:
        st.warning("⚠️ Hidden text detected and excluded from the analysis.")
        with st.expander("Show findings"):
            st.write(parsed.findings)

    if terms.available() and bucket.available():
        feats = terms.term_features(parsed.visible_text, job_input)
        result = bucket.predict(feats)
        icon = {"Likely fit": "🟢", "Possible fit": "🟡", "Unlikely fit": "🔴"}[result.label]
        st.subheader(f"{icon} {result.label}")
        st.caption(
            "A rough estimate, not a verdict. On a labelled benchmark this model beats guessing "
            "but is far from perfect (macro-F1 ≈ 0.40). Probabilities: "
            + ", ".join(f"{k} {v:.0%}" for k, v in result.probabilities.items()))
        c1, c2 = st.columns(2)
        c1.markdown("**✅ Job keywords found in your resume**")
        c1.write(", ".join(feats.matched_top[:SHOWN_TERMS]) or "None")
        c2.markdown("**❌ Important job keywords missing**")
        c2.write(", ".join(feats.missing_top[:SHOWN_TERMS]) or "None")
        st.caption("Keywords are ranked by how distinctive they are across a corpus of resumes "
                   "and job posts, so rare, specific terms count more than filler.")
    else:
        st.info("Scoring model files are missing; showing AI feedback only.")

    st.divider()
    st.subheader("🤖 AI feedback")
    used = st.session_state.get("llm_calls", 0)
    if used >= MAX_LLM_CALLS_PER_SESSION:
        st.info("AI feedback limit reached for this session.")
    elif not _budget_left():
        st.info("The demo's daily AI-feedback budget is used up. The fit estimate above still works; "
                "try again tomorrow, or run the app yourself with your own key (see the README).")
    elif st.button("Generate AI feedback"):
        try:
            with st.spinner("Thinking... (can take 20-40s on small models)"):
                feedback, dropped = explain(parsed.visible_text, job_input)
            st.session_state["llm_calls"] = used + 1
            _daily_budget()["used"] += 1
            st.markdown("**Strengths**")
            for p in feedback.strengths:
                st.markdown(f"- {p.text}" + (f"  \n  > _{p.evidence}_" if p.evidence else ""))
            st.markdown("**Gaps**")
            for g in feedback.gaps:
                st.markdown(f"- {g.text}")
            st.markdown("**Suggestions**")
            for s in feedback.suggestions:
                st.markdown(f"- {s}")
            if dropped:
                st.caption(f"{dropped} unverifiable claim(s) removed by the fact-check.")
        except LLMUnavailable:
            st.info("AI feedback is unavailable right now; the analysis above is still valid.")
else:
    st.info("Upload a resume and paste a job description to begin.")
