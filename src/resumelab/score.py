"""Explainable scoring: a skill breakdown plus lexical and semantic similarity."""
import re
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .embed import semantic_coverage

# Tokens keep +, #, . so "c++", "c#", "node.js" survive.
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#.\-/]*[a-z0-9+#]|[a-z0-9]")

SKILLS = {
    # languages
    "python", "java", "javascript", "typescript", "c++", "c#", "golang", "rust", "sql",
    "scala", "kotlin", "swift", "php", "ruby", "bash",
    # ml / data
    "pytorch", "tensorflow", "keras", "scikit-learn", "xgboost", "pandas", "numpy", "spark",
    "pyspark", "airflow", "dbt", "kafka", "hadoop", "tableau", "power bi", "opencv",
    "machine learning", "deep learning", "nlp", "computer vision", "time series",
    "statistics", "data analysis", "data visualization", "a/b testing", "feature engineering",
    "hugging face", "transformers", "fine-tuning", "reinforcement learning",
    # genai
    "llm", "rag", "embeddings", "prompt engineering", "langchain", "langgraph", "llamaindex",
    "vector database", "pinecone", "faiss", "chroma", "openai", "agents", "generative ai",
    "mcp", "guardrails", "evals",
    # mlops / cloud / infra
    "mlops", "llmops", "mlflow", "docker", "kubernetes", "terraform", "aws", "gcp", "azure",
    "sagemaker", "bedrock", "vertex ai", "git", "ci/cd", "github actions", "linux",
    "microservices", "rest api", "graphql", "grpc",
    # web / db
    "fastapi", "flask", "django", "react", "node.js", "streamlit", "html", "css",
    "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "neo4j", "nosql",
    # practice
    "agile", "scrum", "unit testing", "tdd", "system design",
}
ALIASES = {
    "ml": "machine learning", "dl": "deep learning", "k8s": "kubernetes", "js": "javascript",
    "ts": "typescript", "postgres": "postgresql", "sklearn": "scikit-learn", "gen ai": "generative ai",
    "genai": "generative ai", "large language model": "llm", "large language models": "llm",
    "llms": "llm", "retrieval-augmented generation": "rag", "retrieval augmented generation": "rag",
    "natural language processing": "nlp", "vector db": "vector database",
    "vector databases": "vector database", "restful": "rest api", "rest apis": "rest api",
    "apis": "rest api", "ai agents": "agents", "agentic": "agents", "huggingface": "hugging face",
    "node": "node.js", "nodejs": "node.js", "cicd": "ci/cd", "unit tests": "unit testing",
    "ab testing": "a/b testing", "a/b tests": "a/b testing", "finetuning": "fine-tuning",
    "fine tuning": "fine-tuning", "evaluations": "evals", "evaluation": "evals",
}
_ALIAS_RE = re.compile(
    r"(?<![\w+#.])(" + "|".join(re.escape(a) for a in sorted(ALIASES, key=len, reverse=True)) + r")(?![\w+#])"
)
_SKILL_RE = re.compile(
    r"(?<![\w+#.])(" + "|".join(re.escape(s) for s in sorted(SKILLS, key=len, reverse=True)) + r")(?![\w+#])"
)


def _normalize(text: str) -> str:
    return _ALIAS_RE.sub(lambda m: ALIASES[m.group(1)], text.lower())


def extract_skills(text: str) -> set[str]:
    return set(_SKILL_RE.findall(_normalize(text)))


@dataclass
class ScoreReport:
    overall: float
    skill_coverage: float
    text_similarity: float  # TF-IDF cosine (lexical)
    semantic: float | None  # embedding coverage; None if embeddings unavailable
    matched: list[str]
    missing: list[str]
    extra: list[str]


def tfidf_similarity(resume: str, job: str) -> float:
    try:
        vec = TfidfVectorizer(token_pattern=None, tokenizer=TOKEN_RE.findall,
                              lowercase=True, stop_words="english", sublinear_tf=True)
        m = vec.fit_transform([resume, job])
        return float(cosine_similarity(m[0:1], m[1:2])[0][0])
    except ValueError:  # empty vocabulary (e.g. scanned PDF)
        return 0.0


def score(resume: str, job: str, use_embeddings: bool = True) -> ScoreReport:
    r_sk, j_sk = extract_skills(resume), extract_skills(job)
    matched, missing = sorted(r_sk & j_sk), sorted(j_sk - r_sk)
    coverage = len(matched) / len(j_sk) if j_sk else 0.0
    sim = tfidf_similarity(resume, job)
    sem = semantic_coverage(resume, job) if use_embeddings else None

    parts = [(coverage, 0.5)] if j_sk else []
    parts.append((sem, 0.5) if sem is not None else (sim, 0.3 if j_sk else 1.0))
    total_w = sum(w for _, w in parts)
    overall = sum(v * w for v, w in parts) / total_w
    return ScoreReport(overall, coverage, sim, sem, matched, missing, sorted(r_sk - j_sk))
