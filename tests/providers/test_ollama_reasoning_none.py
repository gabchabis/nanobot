"""Ollama must be told explicitly to disable thinking.

Ollama's OpenAI-compatible endpoint turns thinking off only when it receives
``reasoning_effort: "none"``. Omitting the field leaves thinking models (Qwen3,
...) reasoning, which exhausts small token budgets (title generation) and can
make history consolidation end in ``finish_reason=length``.
"""

from __future__ import annotations

from nanobot.providers.openai_compat_provider import OpenAICompatProvider
from nanobot.providers.registry import PROVIDERS


def _provider(name: str, model: str) -> OpenAICompatProvider:
    specs = {s.name: s for s in PROVIDERS}
    return OpenAICompatProvider(
        api_key="test-key",
        api_base="http://localhost:11434/v1",
        default_model=model,
        spec=specs[name],
    )


def _kwargs(provider: OpenAICompatProvider, effort: str | None) -> dict:
    return provider._build_kwargs(
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        model=None,
        max_tokens=96,
        temperature=0.1,
        reasoning_effort=effort,
        tool_choice=None,
    )


def test_ollama_none_sends_reasoning_effort_none() -> None:
    kwargs = _kwargs(_provider("ollama", "qwen3.8-27b-64k"), "none")
    assert kwargs["reasoning_effort"] == "none"


def test_ollama_unset_effort_sends_nothing() -> None:
    kwargs = _kwargs(_provider("ollama", "qwen3.8-27b-64k"), None)
    assert "reasoning_effort" not in kwargs


def test_ollama_low_is_still_forwarded() -> None:
    kwargs = _kwargs(_provider("ollama", "qwen3.8-27b-64k"), "low")
    assert kwargs["reasoning_effort"] == "low"


def test_other_providers_still_omit_none() -> None:
    kwargs = _kwargs(_provider("openrouter", "openai/gpt-4o"), "none")
    assert "reasoning_effort" not in kwargs
