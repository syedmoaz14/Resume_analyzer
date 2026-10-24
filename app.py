import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from resumelab.explain import explain  # noqa: E402
from resumelab.llm import LLMUnavailable  # noqa: E402
from resumelab.parse import parse_pdf  # noqa: E402
from resumelab.score import score  # noqa: E402

MAX_LLM_CALLS_PER_SESSION = 5

# Hosted deploys: put LLM_BASE_URL / LLM_API_KEY / LLM_MODEL in Streamlit secrets.
try:
    for _k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        if _k in st.secrets:
            os.environ[_k] = st.secrets[_k]
except Exception:  # no secrets file locally
    pass

st.set_page_config(page_title="ResumeLab", page_icon="📄")
st.title("📄 ResumeLab")
st.markdown("Explainable resume-to-job matching, with a hidden-text safety check.")

uploaded_file = st.file_uploader("📤 Upload Resume (PDF)", type=["pdf"])
job_input = st.text_area("💼 Paste Job Description Here", height=200)

if uploaded_file and job_input.strip():
    parsed = parse_pdf(uploaded_file.read())

    if not parsed.visible_text.strip():
        st.error("No readable text found. The PDF may be a scan or image-only.")
        st.stop()

    if parsed.has_hidden_content:
        st.warning("⚠️ Hidden text detected and excluded from scoring.")
        with st.expander("Show findings"):
            st.write(parsed.findings)

    report = score(parsed.visible_text, job_input)
    st.success(f"✅ Match Score: {report.overall * 100:.1f}%")

    c1, c2, c3 = st.columns(3)
    c1.metric("Skill coverage", f"{report.skill_coverage * 100:.0f}%")
    c2.metric("Semantic match", "n/a" if report.semantic is None else f"{report.semantic * 100:.0f}%")
    c3.metric("Keyword overlap", f"{report.text_similarity * 100:.0f}%")

    st.subheader("✅ Matched skills")
    st.write(", ".join(report.matched) or "None")
    st.subheader("❌ Missing skills")
    st.write(", ".join(report.missing) or "None")
    if report.extra:
        st.caption("Other skills on the resume: " + ", ".join(report.extra))

    st.divider()
    st.subheader("🤖 AI feedback")
    used = st.session_state.get("llm_calls", 0)
    if used >= MAX_LLM_CALLS_PER_SESSION:
        st.info("AI feedback limit reached for this session.")
    elif st.button("Generate AI feedback"):
        try:
            with st.spinner("Thinking... (can take 20-40s on small models)"):
                feedback, dropped = explain(parsed.visible_text, job_input)
            st.session_state["llm_calls"] = used + 1
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
            st.info("AI feedback is unavailable right now; the score above is still valid.")
else:
    st.info("Upload a resume and paste a job description to begin.")
