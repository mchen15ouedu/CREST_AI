"""LLM routing — self-hosted vLLM first, paid OpenAI as the last-resort fallback.

vLLM (https://github.com/vllm-project/vllm) serves an OpenAI-compatible API, so
the same `openai` client works by pointing `base_url` at the vLLM endpoint. Set
`VLLM_BASE_URL` (e.g. http://vllm-host:8000/v1) to use a self-hosted model; only
if that is absent/unreachable do we fall through to the OpenAI API
(`OPENAI_API_KEY`). Extra providers can be inserted between the two.

vLLM needs a GPU; on the cpu-basic Space it runs as a *separate* endpoint that
this app points at — not inside this container.
"""
from __future__ import annotations

import os


def _providers() -> list[dict]:
    """Ordered provider list: vLLM (primary) → OpenAI (last resort)."""
    out = []
    if os.environ.get("VLLM_BASE_URL"):
        out.append({
            "name": "vllm",
            "base_url": os.environ["VLLM_BASE_URL"],
            "api_key": os.environ.get("VLLM_API_KEY", "EMPTY"),  # vLLM ignores the key
            "model": os.environ.get("VLLM_MODEL", "local-model"),
        })
    if os.environ.get("OPENAI_API_KEY"):
        out.append({
            "name": "openai",
            "base_url": os.environ.get("OPENAI_BASE_URL"),       # None -> api.openai.com
            "api_key": os.environ["OPENAI_API_KEY"],
            "model": os.environ.get("OPENAI_MODEL", "gpt-4o"),
        })
    return out


def available() -> list[str]:
    return [p["name"] for p in _providers()]


def chat(messages: list[dict], temperature: float = 0.2, max_tokens: int = 800,
         json_mode: bool = False) -> tuple[str, str]:
    """Try each provider in priority order; return (content, provider_name)."""
    from openai import OpenAI
    providers = _providers()
    if not providers:
        raise RuntimeError("No LLM configured — set VLLM_BASE_URL (self-hosted vLLM) "
                           "or OPENAI_API_KEY (fallback).")
    errors = []
    for p in providers:
        try:
            client = OpenAI(base_url=p["base_url"] or None, api_key=p["api_key"] or "EMPTY")
            kwargs = dict(model=p["model"], messages=messages,
                          temperature=temperature, max_tokens=max_tokens)
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content, p["name"]
        except Exception as e:
            errors.append(f"{p['name']}: {type(e).__name__}: {str(e)[:120]}")
    raise RuntimeError("All LLM providers failed — " + " | ".join(errors))
