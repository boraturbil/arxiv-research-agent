"""Single entry point for all LLM calls.

Every step of the agent calls `chat()` or `chat_json()`, so changing the
model only means editing the .env file, and every call gets logged in one place.
"""
import os
import re
import time
from collections import defaultdict, deque

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import TypeAdapter

load_dotenv()

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL") or None  # optional

# Temporary errors worth retrying: rate limit / quota, overloaded server
RETRYABLE_CODES = {429, 500, 503}
MAX_RETRIES = 3

# One entry per successful call: which model answered, how long it took, how many retries.
CALL_LOG: list[dict] = []

# Circuit breaker: a model that failed all its retries is skipped for the rest of the run,
# so later calls go straight to the fallback instead of waiting ~30s every time.
UNAVAILABLE_MODELS: set[str] = set()

# Client-side rate limit: max requests per minute per model. Set it a bit below the
# RPM shown in Google AI Studio (free tier Flash Lite: 15), so we never hit 429 errors.
RPM_LIMIT = int(os.getenv("GEMINI_RPM", "12"))
_recent_calls: dict[str, deque] = defaultdict(deque)  # model -> timestamps of last calls

# The client reads GEMINI_API_KEY from the environment automatically.
client = genai.Client()


def _wait_for_rate_limit(model: str) -> None:
    """Sleep if this model already got RPM_LIMIT requests in the last 60 seconds."""
    calls = _recent_calls[model]
    while True:
        now = time.monotonic()
        while calls and now - calls[0] >= 60:
            calls.popleft()
        if len(calls) < RPM_LIMIT:
            calls.append(now)
            return
        wait = 60 - (now - calls[0]) + 0.5
        print(f"[llm] {model}: {RPM_LIMIT} requests/min reached, waiting {wait:.0f}s")
        time.sleep(wait)


def _call(prompt, system, model, temperature, schema) -> str:
    _wait_for_rate_limit(model)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            response_mime_type="application/json" if schema else None,
            response_schema=schema,
        ),
    )
    return response.text


def _retry_delay(error: Exception, default: int = 30, cap: int = 90) -> int:
    """Read how long a 429 rate-limit error asks us to wait ('retryDelay': '38s')."""
    m = re.search(r"retryDelay['\"]?\s*:\s*['\"](\d+(?:\.\d+)?)s", str(error))
    return min(int(float(m.group(1))) + 1, cap) if m else default


def chat(prompt: str, system: str | None = None, model: str | None = None,
         temperature: float = 0.3, schema=None, allow_fallback: bool = True) -> str:
    """Send a prompt and return the model's text answer.

    Retries temporary errors, then switches to GEMINI_FALLBACK_MODEL (if set) as a
    last resort. Evals pass allow_fallback=False so every answer comes from one judge.
    """
    models = [model or DEFAULT_MODEL]
    if allow_fallback and FALLBACK_MODEL and FALLBACK_MODEL not in models:
        models.append(FALLBACK_MODEL)
    # Skip models already marked unavailable, but always keep at least the last one.
    available = [m for m in models if m not in UNAVAILABLE_MODELS]
    models = available or models[-1:]

    last_error = None
    retries = 0
    start = time.perf_counter()
    for current_model in models:
        for attempt in range(MAX_RETRIES):
            try:
                text = _call(prompt, system, current_model, temperature, schema)
                CALL_LOG.append({
                    "model": current_model,
                    "seconds": round(time.perf_counter() - start, 1),
                    "retries": retries,
                })
                return text
            except errors.APIError as e:
                if e.code not in RETRYABLE_CODES:
                    raise  # e.g. 404 wrong model name, 401 bad key: retrying won't help
                last_error = e
                retries += 1
                if e.code == 429 and "PerDay" in str(e):
                    # Daily quota used up: waiting won't help today, go to the fallback.
                    print(f"[llm] {current_model}: daily quota exhausted")
                    break
                if e.code == 429:
                    # Per-minute rate limit: wait as long as the API asks (usually < 60s).
                    wait = _retry_delay(e)
                else:
                    wait = 2 ** (attempt + 1)  # overloaded server: 2, 4, 8... seconds
                print(f"[llm] {current_model}: error {e.code}, retrying in {wait}s "
                      f"({attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait)
        if current_model != models[-1]:
            UNAVAILABLE_MODELS.add(current_model)
            print(f"[llm] {current_model} unavailable, using fallback model "
                  f"for the rest of this run")

    raise last_error


def chat_json(prompt: str, schema, system: str | None = None,
              model: str | None = None, temperature: float = 0.2,
              allow_fallback: bool = True):
    """Ask for JSON matching `schema` (e.g. list[str] or a Pydantic model)
    and return it parsed and validated."""
    text = chat(prompt, system=system, model=model, temperature=temperature,
                schema=schema, allow_fallback=allow_fallback)
    return TypeAdapter(schema).validate_json(text)
