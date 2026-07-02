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


def event_brief(label: str, t_start: str, t_end: str) -> tuple[str, str]:
    """Short, readable brief about a flood event (impacts, damage, fatalities,
    links) shown in chat while the model runs. Tries OpenAI's web-search tool
    (Responses API) for post-cutoff events; falls back to plain LLM knowledge
    with an explicit caveat. Returns (markdown_text, provider_tag)."""
    q = (f"Flood event near {label}, roughly {t_start} to {t_end}. In <=150 words of "
         "markdown, summarize for a dashboard reader: what happened, rainfall/river "
         "context, damage, fatalities if known, and 1-3 source links (markdown links). "
         "If you are not certain of specifics, say so plainly rather than inventing "
         "numbers or links.")
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        try:                                    # web-grounded (Responses API)
            from openai import OpenAI
            r = OpenAI(api_key=key).responses.create(
                model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
                tools=[{"type": "web_search_preview"}],
                input=q)
            txt = getattr(r, "output_text", "") or ""
            if txt.strip():
                return txt.strip(), "openai+web"
        except Exception:
            pass
    txt, prov = chat([{"role": "system",
                       "content": "You are a concise flood-event briefer for a dashboard."},
                      {"role": "user", "content": q}], temperature=0.3, max_tokens=400)
    return txt.strip(), prov


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
