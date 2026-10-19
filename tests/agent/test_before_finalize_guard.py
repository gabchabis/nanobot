"""Runner ignores before_finalize results that are not a non-empty str."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.runner_helpers import make_run_spec
from nanobot.agent.hook import AgentHook
from nanobot.config.schema import AgentDefaults
from nanobot.providers.base import LLMProvider, LLMResponse


def _provider(*responses: LLMResponse) -> MagicMock:
    provider = MagicMock(spec=LLMProvider)
    provider.chat_stream_with_retry = AsyncMock(side_effect=list(responses))
    return provider


class _Veto(AgentHook):
    """Returns the given values from before_finalize, one per call, then None."""

    def __init__(self, *values) -> None:
        super().__init__()
        self._values = list(values)

    async def before_finalize(self, context, content):
        return self._values.pop(0) if self._values else None


async def _run(provider, hook):
    from nanobot.agent.runner import AgentRunner

    tools = MagicMock()
    tools.get_definitions.return_value = []
    return await AgentRunner().run(make_run_spec(
        provider,
        initial_messages=[{"role": "user", "content": "do the task"}],
        tools=tools,
        model="test-model",
        max_iterations=5,
        max_tool_result_chars=AgentDefaults().max_tool_result_chars,
        hook=hook,
    ))


def _answer(text: str) -> LLMResponse:
    return LLMResponse(content=text, tool_calls=[], usage=None)


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [MagicMock(), AsyncMock(), 123, ["x"], {"a": 1}, "   ", "\n"])
async def test_non_str_or_blank_veto_is_ignored(bad) -> None:
    provider = _provider(_answer("first"), _answer("second"))
    result = await _run(provider, _Veto(bad))
    assert provider.chat_stream_with_retry.await_count == 1
    assert result.final_content == "first"
    assert all(isinstance(m.get("content"), (str, list, type(None))) for m in result.messages)


@pytest.mark.asyncio
async def test_valid_str_veto_still_continues_the_run() -> None:
    provider = _provider(_answer("first"), _answer("second"))
    result = await _run(provider, _Veto("please also do X"))
    assert provider.chat_stream_with_retry.await_count == 2
    assert result.final_content == "second"
    injected = [m for m in result.messages
                if m.get("role") == "user" and m.get("content") == "please also do X"]
    assert len(injected) == 1
