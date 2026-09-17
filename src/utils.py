import os
import time
import threading
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

_groq_client = None
_gemini_client = None

_RATE_LIMIT = 20
_WINDOW_SECONDS = 60
_request_times = []
_lock = threading.Lock()


def _wait_for_slot():
    with _lock:
        now = time.time()
        while _request_times and (now - _request_times[0]) > _WINDOW_SECONDS:
            _request_times.pop(0)
        if len(_request_times) >= _RATE_LIMIT:
            sleep_for = _WINDOW_SECONDS - (now - _request_times[0]) + 0.5
            time.sleep(sleep_for)
            now = time.time()
            while _request_times and (now - _request_times[0]) > _WINDOW_SECONDS:
                _request_times.pop(0)
        _request_times.append(now)


def get_groq_client():
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in .env")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in .env")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def _groq_call(prompt, model, temperature, max_tokens):
    client = get_groq_client()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
    except TypeError:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    content = response.choices[0].message.content
    if content is None or not content.strip():
        raise RuntimeError("Groq returned empty content")
    return content.strip()


def _gemini_call(prompt, temperature, max_tokens):
    from google.genai import types
    client = get_gemini_client()
    resp = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )
    text = getattr(resp, "text", None)
    if not text:
        raise RuntimeError("Gemini returned empty response")
    return text.strip()


def groq_complete(prompt, model="openai/gpt-oss-20b", temperature=0.0, max_tokens=512):
    """Groq primary with Gemini fallback."""
    _wait_for_slot()

    groq_error = None
    for attempt in range(3):
        try:
            return _groq_call(prompt, model, temperature, max_tokens)
        except Exception as e:
            groq_error = e
            msg = str(e).lower()
            if "429" in msg or "rate" in msg:
                wait = 10 * (attempt + 1)
                print(f"[groq 429, retry #{attempt+1} in {wait}s]")
                time.sleep(wait)
                _wait_for_slot()
                continue
            print(f"[groq failed: {type(e).__name__} -> gemini fallback]")
            break

    gem_error = None
    for attempt in range(4):
        try:
            return _gemini_call(prompt, temperature, max_tokens)
        except Exception as e:
            gem_error = e
            msg = str(e).lower()
            if "429" in msg or "resource_exhausted" in msg or "503" in msg or "unavailable" in msg:
                wait = 30 * (attempt + 1)
                print(f"[gemini {type(e).__name__}, retry #{attempt+1} in {wait}s]")
                time.sleep(wait)
                _wait_for_slot()
                continue
            break

    raise RuntimeError(
        f"Both providers failed.\n"
        f"  Groq: {type(groq_error).__name__}: {groq_error}\n"
        f"  Gemini: {type(gem_error).__name__ if gem_error else 'n/a'}: {gem_error}"
    )


def get_project_root():
    return Path(__file__).resolve().parent.parent


def get_paths():
    root = get_project_root()
    return {
        "root": root,
        "raw": root / "data" / "raw",
        "processed": root / "data" / "processed",
        "golden": root / "data" / "golden_set",
        "results": root / "results",
    }