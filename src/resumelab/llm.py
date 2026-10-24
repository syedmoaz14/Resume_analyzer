"""Provider-agnostic LLM client (any OpenAI-compatible endpoint) with disk cache.

Defaults to a local Ollama server. Point it at a hosted provider with:
    LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
"""
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

load_dotenv()  # picks up .env locally; real env vars / Streamlit secrets still win

T =TypeVar("T", bound=BaseModel)

CACHE_DIR = Path(os.environ.get("RESUMELAB_CACHE", ".cache/llm"))
MAX_OUTPUT_TOKENS = 1500  # headroom: reasoning models (gpt-oss) spend tokens thinking
MAX_RATE_WAITS = 3
RATE_WAIT_SECONDS = 10
_lock = threading.Lock()  # one generation at a time; protects CPU boxes and budgets


class LLMUnavailable(RuntimeError):
    pass


def _config() -> dict:
    return {
        "base_url": os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"),
        "api_key": os.environ.get("LLM_API_KEY", "ollama"),
        "model": os.environ.get("LLM_MODEL", "qwen2.5:3b"),
    }


def _cache_path(model: str, prompt_version: str, system: str, user: str) -> Path:
    key = hashlib.sha256(json.dumps([model, prompt_version, system, user]).encode()).hexdigest()
    return CACHE_DIR / f"{key}.json"


def complete_json(schema: type[T], system: str, user: str, prompt_version: str = "v1",
                  retries: int = 1) -> T:
    """Return a validated `schema` instance, or raise LLMUnavailable."""
    cfg = _config()
    path = _cache_path(cfg["model"], prompt_version, system, user)
    try:
        return schema.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError):
        pass

    try:
        from openai import OpenAI
    except ImportError as e:
        raise LLMUnavailable("openai package not installed") from e

    client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"], timeout=120, max_retries=0)
    sys_prompt = (f"{system}\n\nReply with ONLY a JSON object matching this schema:\n"
                  f"{json.dumps(schema.model_json_schema())}")
    from openai import RateLimitError

    last_err: Exception | None = None
    rate_waits = 0
    with _lock:
        attempt = 0
        while attempt <= retries:
            try:
                resp = client.chat.completions.create(
                    model=cfg["model"],
                    messages=[{"role": "system", "content": sys_prompt},
                              {"role": "user", "content": user}],
                    response_format={"type": "json_object"},
                    max_tokens=MAX_OUTPUT_TOKENS,
                    temperature=0,
                )
                result = schema.model_validate_json(resp.choices[0].message.content or "")
            except ValidationError as e:  # malformed JSON: retry once
                last_err = e
                attempt += 1
                continue
            except RateLimitError as e:  # free tiers cap tokens/minute: wait, then retry
                rate_waits += 1
                if rate_waits > MAX_RATE_WAITS:
                    raise LLMUnavailable("rate limited; try again in a minute") from e
                time.sleep(RATE_WAIT_SECONDS)
                continue
            except Exception as e:  # connection refused, auth, quota, timeout...
                raise LLMUnavailable(str(e)) from e
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(result.model_dump_json(), encoding="utf-8")
            return result
    raise LLMUnavailable(f"model returned invalid JSON: {last_err}")
