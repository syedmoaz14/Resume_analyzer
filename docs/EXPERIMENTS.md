# Experiments log

What was tried while building ResumeLab, what the numbers said, and what was decided.
Negative results are kept on purpose: the first version of the match score looked good on
hand-made data and turned out to be close to a coin flip on real data.

## Benchmark

[`cnamuangtoun/resume-job-description-fit`](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit)
(Hugging Face, MIT): resume + job description pairs labelled *No Fit / Potential Fit / Good Fit*.

| | Train | Test |
|---|---|---|
| Pairs | 6,241 | 1,759 |
| Unique resumes | 642 | 477 (almost all also in train) |
| Unique job posts | 280 | **71 (none in train)** |
| No / Potential / Good | 3143 / 1556 / 1542 | 857 / 444 / 458 |

Implications:

- The split is by **job post**, so the test measures generalisation to unseen jobs.
- The 1,759 test rows come from only **71 distinct jobs**: the effective sample is much
  smaller than it looks, and differences of about 0.01 macro-F1 are noise. No confidence
  intervals were computed yet (bootstrapping by job post is the right way to do it).
- Labels are noisy. The model card of a DistilBERT classifier fine-tuned on this dataset
  reports roughly 0.50 accuracy / 0.43 macro-F1, and says a TF-IDF baseline matched it.
  (Secondary source: read from the model card, not reproduced here.)

Metrics: macro-F1 (class-balanced), accuracy, Spearman between predicted fit and the 0/1/2
label, and AUC of "Good Fit scores above No Fit".

## 1. First version: hand-made evals looked perfect (misleading)

12 synthetic resume/job pairs written by the author, labels written by the author.

| Method | Pairwise acc. | Spearman |
|---|---|---|
| TF-IDF (2-document) | 0.94 | 0.84 |
| Tech skill list | 0.94 | 0.93 |
| Embeddings (bge-small, line matching) | 1.00 | 0.91 |
| Hybrid (50/50 skills + embeddings) | 1.00 | 0.96 |

**Lesson:** tiny, self-authored, easy data. Embedding rescaling constants were guesses.
Do not quote these numbers.

## 2. Same methods on real data (150 random test pairs)

| Method | Spearman | AUC good>no | Mean score No / Potential / Good |
|---|---|---|---|
| TF-IDF | 0.14 | 0.58 | 0.075 / 0.089 / 0.083 |
| Tech skill list | 0.22 | 0.65 | 0.140 / 0.189 / 0.254 |
| Embeddings (bge-small) | -0.07 | 0.44 | 0.513 / 0.546 / 0.475 |
| Hybrid (what the app showed) | 0.06 | 0.53 | 0.409 / 0.476 / 0.409 |

The embedding half ranked *backwards* (Good Fit pairs scored below Potential Fit) and
dragged the hybrid to chance. Likely causes: long noisy postings where boilerplate lines
are treated as requirements, and uncalibrated constants. Cost: about 2.4 s per pair on CPU.

A second issue: the hand-written skill list was almost entirely tech terms, but the
benchmark covers every profession, so the feature was often empty.

## 3. Learned scorer, domain-agnostic features (full train/test)

Features fitted on the train split only; multinomial logistic regression with balanced
class weights. Corpus terms = term weights (IDF over 1-2 grams) learned from the training
resumes and job posts, so rare specific terms count more than filler. No hand-written skill
list.

| Model | Accuracy | Macro-F1 | Spearman |
|---|---|---|---|
| Always predict "No Fit" | 0.487 | 0.218 | 0.00 |
| Tech skill list only | 0.369 | 0.363 | 0.08 |
| 2-document TF-IDF only | 0.417 | 0.356 | 0.13 |
| **Corpus-weighted terms only** | **0.451** | **0.401** | **0.155** |
| All features (no embeddings) | 0.438 | 0.405 | 0.160 |

**Decision:** ship the corpus-weighted terms model. It is domain-agnostic, tiny
(three numeric features), fast, explainable (it yields the top matched/missing job terms)
and in the same range as a fine-tuned transformer on this data.

## 4. Taxonomy linking (go / no-go experiment)

Hypothesis: linking text to a standard skill taxonomy would add signal and explainability.

- **Tech product names** (O*NET technology skills, 8,730 products, n-gram lookup):
  alone 0.338 macro-F1; with corpus terms 0.404 vs 0.401. Ambiguous names hurt
  (the person "chef" matches the configuration-management tool "Chef").
- **Sentence-to-taxonomy embedding** (all-MiniLM-L6-v2): resume sentences linked to the nearest
  O*NET detailed work activity (2,087) or ESCO skill (13,960 preferred labels).

| Model | Accuracy | Macro-F1 | Spearman |
|---|---|---|---|
| Corpus terms (baseline) | 0.451 | 0.401 | 0.155 |
| O*NET only | 0.379 | 0.350 | 0.115 |
| Corpus + O*NET | 0.442 | 0.403 | 0.167 |
| ESCO only | 0.388 | 0.352 | 0.099 |
| Corpus + ESCO | 0.449 | 0.408 | 0.164 |
| Corpus + O*NET + ESCO | 0.439 | 0.406 | 0.156 |

**Decision: no-go for the score.** Each taxonomy alone is worse than the baseline, and added
on top the change (+0.002 to +0.007) is inside the noise. Every method plateaus near
0.40 macro-F1, which suggests the label noise, not the features, is the ceiling.
Not tested: whether taxonomy links would improve the *explanations* (e.g. grouping missing
skills by category). Left as future work. ESCO data is not redistributed in this repo
(licence terms not confirmed).

## 5. LLM feedback: failure modes found and fixed

- **False "gaps":** the model claimed a missing skill that was in the resume. Cause 1: the
  prompt cut a 14k-character resume at 6k characters. Cause 2: small models assert absence
  carelessly. Fix: raise the limit, make the model name the missing requirement, then drop
  any gap whose requirement words appear in the resume.
- **Invented quotes:** evidence quotes must appear verbatim in the resume or are removed.
- **Rate limits:** the free Groq tier allows 8,000 tokens/minute on the chosen model; two
  long-resume calls in a minute return HTTP 429. The client waits and retries, and the app
  falls back to the non-LLM analysis if the model is unavailable.

## 6. Bias audit (inconclusive)

Counterfactual variants of one resume (10 names from five name groups, plus a career-gap
line), scored by the deterministic score and by an LLM rater (`openai/gpt-oss-20b`).

- The deterministic score never reads the name, so its spread is 0 by construction.
- The LLM rater moved by 3 points on one pair (favouring a white-coded name) and 0 on
  another. With one sample per variant, a rater that scores everything 95 (ceiling effect),
  and two pairs, this is **not evidence either way**. A real audit needs mid-fit pairs,
  repeated runs and a significance test.

## 7. Hidden-text shield

PDF text spans that are white, under 4 pt, or off-page are excluded from scoring and
reported to the user (defence against prompt-injection / keyword-stuffing). Verified by unit
tests on constructed PDFs; no prevalence measurement was done on real resumes.

## Limitations (summary)

- Benchmark effective size is 71 job posts; no confidence intervals yet; one random seed.
- Embedding experiments used two different small models (bge-small, MiniLM) with default
  settings; a fine-tuned or larger model was not tried.
- Noisy labels cap every method; results are "better than guessing", not "reliable".
- The bias audit is too small to support any claim.

## Attribution

Taxonomy experiments used the O*NET 31.0 Database by the U.S. Department of Labor,
Employment and Training Administration (USDOL/ETA), licensed CC BY 4.0, and ESCO v1.2.1
(European Commission). The O*NET/ESCO files are not committed.

## Reproduce

```
python -c "from datasets import load_dataset; load_dataset('cnamuangtoun/resume-job-description-fit').save_to_disk('evals/data/hf_fit')"
python evals/build_terms.py          # corpus term weights (train split only)
python evals/train_scorer.py         # section 3
python evals/run_taxonomy_eval.py    # section 4 (needs O*NET/ESCO files, see build_taxonomy.py)
python evals/run_hf_evals.py --n 150 # section 2
python evals/run_evals.py            # section 1 (synthetic)
python evals/run_bias_audit.py --llm # section 6
```
