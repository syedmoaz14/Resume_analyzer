# ResumeLab 📄

[![CI](https://github.com/syedmoaz14/Resume_analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/syedmoaz14/Resume_analyzer/actions/workflows/ci.yml)

An explainable resume-to-job matcher that works for **any profession**, not just tech.
Upload a PDF resume, paste a job description, and get a fit bucket, the job keywords you
match and miss, and optional AI feedback whose claims are fact-checked against your resume.

**🚀 Live demo: https://resume-analyzer-portfolio.streamlit.app**
(free hosting: it may take a few seconds to wake up, and AI feedback has a daily limit)

> **Status: portfolio project.** The scoring model is deliberately honest about its limits:
> it beats guessing on a real benchmark, but it is not a hiring decision tool.

## What it does

- **Fit bucket** (Unlikely / Possible / Likely) with class probabilities, not a fake-precise percentage.
- **Matched and missing job keywords**, ranked by how distinctive they are across a corpus of
  resumes and job posts (so "servsafe" or "kubernetes" count more than "team" or "work").
- **Hidden-text shield:** white, tiny or off-page text in the PDF is excluded and reported
  (a defence against keyword stuffing and prompt injection aimed at AI screeners).
- **AI feedback** (strengths, gaps, suggestions) from any OpenAI-compatible LLM, with two
  fact-checks on its output (see below). Optional: the app still works with no LLM.

## How it works

```
PDF ──► parse + hidden-text shield ──► visible text
                                         │
        job description ─────────────────┤
                                         ▼
                 corpus-weighted term features (3 numbers)
                                         │
                                         ▼
              logistic regression (weights in a JSON file)
                                         │
                                         ▼
          fit bucket  +  matched / missing keywords
                                         
visible text + job ──► LLM (1 call, JSON schema) ──► fact-check ──► feedback
```

Design choices and why:

- **Terms learned from data, no hand-written skill list.** Term weights (IDF over 1–2 grams)
  are fitted on the training split of a public benchmark, so it generalises across professions.
  An early version used a tech-only skill list and failed on non-tech jobs.
- **A tiny, inspectable model.** Three features and a logistic regression exported as JSON; the
  app needs only numpy to score. No GPU, no model download for scoring.
- **LLM only where it adds value.** One short call per analysis, results cached on disk by hash,
  a per-session cap, a lock so concurrent users don't burn the rate limit, and automatic retry
  on HTTP 429. If the LLM is down, the scoring and keywords still work.
- **Fact-check the LLM.**
  - Strength quotes must appear verbatim in the resume or they are removed.
  - A "gap" (missing skill) is dropped if the requirement's words are actually in the resume.
    (This catches a real failure observed during development.)

## Results (honest version)

Benchmark: [`cnamuangtoun/resume-job-description-fit`](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit),
3 labels (No / Potential / Good fit). Trained on the 6,241-pair train split, evaluated on the
held-out 1,759-pair test split, which comes from **71 job posts not seen in training**.

| Model | Accuracy | Macro-F1 | Spearman |
|---|---|---|---|
| Always predict "No Fit" | 0.487 | 0.218 | 0.00 |
| Tech skill list | 0.369 | 0.363 | 0.08 |
| 2-document TF-IDF (the original app) | 0.417 | 0.356 | 0.13 |
| **Corpus-weighted terms (shipped)** | **0.451** | **0.401** | **0.155** |

What this means:

- It is **better than guessing and better than the earlier approaches**, in the same range as a
  fine-tuned transformer reported for this dataset (≈0.43 macro-F1, from its model card).
- It is **far from reliable.** Labels are noisy and every method I tried plateaus near 0.40
  macro-F1. That is why the app shows a bucket with probabilities and a caveat.
- Only 71 distinct jobs are in the test set, so differences of ~0.01 are noise. No confidence
  intervals yet.

### What did *not* work (kept on purpose)

- A hybrid of skills + embeddings looked perfect on 12 hand-made pairs (1.00) but was
  **close to chance on real data** (AUC 0.53). The embedding half ranked backwards.
- Linking text to **O\*NET** (work activities, technologies) and **ESCO** (13,960 skills)
  taxonomies gave **no measurable gain** (macro-F1 0.401 → 0.403–0.408, within noise).
- The counterfactual **bias audit** (name and career-gap swaps) is built, but with this little
  data its result is **inconclusive**.

Full numbers, caveats and reproduction commands: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## Cost and latency

| Step | Typical cost |
|---|---|
| PDF parse + hidden-text check | ~70 ms (short resume) |
| Term features + bucket | a few ms |
| AI feedback (Groq, `openai/gpt-oss-20b`) | ~3 s for a 14k-character resume; free tier |
| Repeat analysis of the same inputs | instant (disk cache) |

The free Groq tier allows 8,000 tokens/minute on this model, so two back-to-back long-resume
requests can hit a rate limit; the client waits and retries.
(An embedding-based score cost ≈2.4 s per pair on CPU and did not help, so it was removed.)

## Quick start

```bash
git clone https://github.com/syedmoaz14/Resume_analyzer.git
cd Resume_analyzer
pip install -r requirements.txt
streamlit run app.py
```

The scoring works immediately. For AI feedback, copy `.env.example` to `.env` and set any
OpenAI-compatible provider:

```
LLM_API_KEY=your-key
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=openai/gpt-oss-20b
```

For fully local inference use Ollama (`LLM_BASE_URL=http://localhost:11434/v1`,
`LLM_API_KEY=ollama`, `LLM_MODEL=qwen2.5:3b`). On Streamlit Cloud put the same three values
in the app's Secrets. Never commit `.env`.

## Docker

```bash
docker build -t resumelab .
docker run -p 8501:8501 --env-file .env resumelab   # then open http://localhost:8501
```

The image contains only the app and the small model files; the LLM settings are passed at run
time and never baked in. (To reach an Ollama server running on your machine from inside the
container, use `LLM_BASE_URL=http://host.docker.internal:11434/v1`.)

## MCP server

`mcp_server.py` exposes the analyzer as [MCP](https://modelcontextprotocol.io) tools, so Claude
Desktop, Claude Code or any MCP client can call it:

| Tool | What it does | Leaves your machine? |
|---|---|---|
| `analyze_fit` | fit bucket + matched/missing job keywords for a resume (PDF path or text) and a job description | no |
| `check_hidden_text` | finds white, tiny or off-page text in a PDF resume | no |
| `ai_feedback` | fact-checked strengths, gaps and suggestions from an LLM | **yes**, to your configured LLM provider |

```bash
pip install -r requirements-mcp.txt
# Claude Code (use the full path to the python that has the requirements installed):
claude mcp add resumelab -- python /full/path/to/Resume_analyzer/mcp_server.py
```

For Claude Desktop, add this to `claude_desktop_config.json` and restart it:

```json
{
  "mcpServers": {
    "resumelab": {
      "command": "python",
      "args": ["/full/path/to/Resume_analyzer/mcp_server.py"]
    }
  }
}
```

Then ask, for example: *"Using resumelab, how well does ~/Downloads/cv.pdf fit this job: …"*.
`ai_feedback` reads `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` from `.env`. The server only
reads `.pdf` files up to 10 MB, and tool errors come back as clear messages.

## CI

GitHub Actions runs on every push and pull request
([workflow](.github/workflows/ci.yml)):

1. unit tests;
2. a **scorer regression check**: scores the 1,759-pair held-out benchmark with the committed
   model files and fails the build if macro-F1 drops below 0.37 (currently 0.399);
3. a Docker build plus a container health-check smoke test.

## Tests and evals

```bash
pip install -r requirements-dev.txt  # or requirements-evals.txt for the benchmark scripts
python -m pytest -q                  # unit tests (parser, shield, fact-checks, scoring, audit)
python evals/check_regression.py     # what CI runs: macro-F1 on the held-out benchmark
python evals/run_evals.py            # small synthetic comparison (sanity check only)
python evals/train_scorer.py         # benchmark table above (needs the dataset, see docs)
python evals/run_bias_audit.py --llm # counterfactual audit
```

To reproduce the benchmark, download the dataset first:

```python
from datasets import load_dataset
load_dataset("cnamuangtoun/resume-job-description-fit").save_to_disk("evals/data/hf_fit")
```

## Project structure

```
app.py                      Streamlit UI
mcp_server.py               MCP server (3 tools) for Claude Desktop / Claude Code
src/resumelab/
  parse.py                  PDF parsing + hidden-text shield
  terms.py                  corpus-weighted term features, matched/missing keywords
  bucket.py                 fit bucket from term features (numpy only)
  llm.py                    OpenAI-compatible client: cache, lock, retry, JSON schema
  explain.py                LLM feedback + fact-checks
  audit.py                  counterfactual bias audit
  embed.py, taxonomy.py,    experiment code (not used by the app); see docs/EXPERIMENTS.md
  score.py
  data/                     term weights and bucket model (numbers only, no resume text)
evals/                      training, benchmarks, audit, experiments, CI regression check
docs/EXPERIMENTS.md         full experiment log, including failures
tests/
Dockerfile, .github/        container image and CI workflow
requirements.txt            runtime; -dev adds pytest; -evals adds datasets + fastembed
```

## Limitations

- Fit buckets are rough estimates from noisy labels; do not use them for hiring decisions.
- English-language resumes and job posts only; text-based PDFs only (no OCR for scans).
- The LLM can still be wrong in ways the fact-checks cannot catch.
- The bias audit is too small to support any claim about fairness.

## Roadmap

A stronger bias audit.

## Attribution and licence

Code: MIT. Benchmark: `cnamuangtoun/resume-job-description-fit` (MIT). Experiments used the
O\*NET 31.0 Database by the U.S. Department of Labor, Employment and Training Administration
(USDOL/ETA), licensed CC BY 4.0 (O\*NET® is a trademark of USDOL/ETA), and ESCO v1.2.1 © European
Union. Neither dataset is included in this repository.
