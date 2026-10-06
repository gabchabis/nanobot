"""SupervisorHook: announce-without-act, repeated calls, stalls, runner veto."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.runner_helpers import make_run_spec
from nanobot.agent.hook import AgentHook, AgentHookContext, CompositeHook
from nanobot.agent.hooks.supervisor import (
    SupervisorHook,
    looks_like_unfulfilled_announcement,
    make_supervisor_hook_factory,
)
from nanobot.config.schema import AgentDefaults, SupervisorConfig
from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest


def _cfg(**kw) -> SupervisorConfig:
    return SupervisorConfig(**{"mode": "enforce", **kw})


def _ctx(iteration: int = 0, messages=None, usage=None) -> AgentHookContext:
    return AgentHookContext(iteration=iteration, messages=messages if messages is not None else [],
                            usage=usage)


def _call(name="read_file", **args) -> ToolCallRequest:
    return ToolCallRequest(id="1", name=name, arguments=args or {"path": "a.txt"})


@pytest.mark.parametrize("text", [
    "J'ai lu le fichier. Je vais maintenant lancer la génération.",
    "Voici le plan. Je vais commencer par :",
    "Let me run the tests now.",
    "Parfait, ensuite je passe à l'export…",
    "I'll check the logs next.",
])
def test_detects_unfulfilled_announcement(text):
    assert looks_like_unfulfilled_announcement(text)


@pytest.mark.parametrize("text", [
    "",
    None,
    "La tâche est terminée. Le fichier est dans out/.",
    "Je vais te résumer : tout est fait.",
    "Si tu veux, je vais ensuite l'exporter en PNG.",
    "Je vais lancer le rendu, tu veux 1080p ou 4K ?",
    "Voici le résultat. Dis-moi si je dois continuer.",
    "Le fichier contient 3 sections et 12 lignes.",
])
def test_ignores_normal_final_answers(text):
    assert not looks_like_unfulfilled_announcement(text)


@pytest.mark.asyncio
async def test_repeated_identical_call_triggers_nudge_in_enforce():
    hook = SupervisorHook(_cfg(repeat_threshold=3))
    for i in range(3):
        await hook.after_execute_tool(_ctx(i), _call(), None, {}, "same result")
    messages: list = []
    await hook.before_iteration(_ctx(3, messages))
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert "[supervision]" in messages[0]["content"]
    assert "read_file" in messages[0]["content"]


@pytest.mark.asyncio
async def test_same_call_with_changing_results_is_not_a_repeat():
    hook = SupervisorHook(_cfg(repeat_threshold=2))
    for i in range(5):
        await hook.after_execute_tool(_ctx(i), _call("check_job"), None, {}, f"progress {i}%")
    messages: list = []
    await hook.before_iteration(_ctx(5, messages))
    assert messages == []


@pytest.mark.asyncio
async def test_observe_mode_never_injects():
    hook = SupervisorHook(_cfg(mode="observe", repeat_threshold=2))
    for i in range(4):
        await hook.after_execute_tool(_ctx(i), _call(), None, {}, "same")
    messages: list = []
    await hook.before_iteration(_ctx(4, messages))
    assert messages == []
    assert await hook.before_finalize(_ctx(), "Je vais lancer la génération.") is None


@pytest.mark.asyncio
async def test_stall_after_iterations_without_new_information():
    hook = SupervisorHook(_cfg(stall_iterations=3))
    await hook.after_execute_tool(_ctx(0), _call(), None, {}, "first")
    for i in range(1, 4):
        await hook.after_iteration(_ctx(i))
    messages: list = []
    await hook.before_iteration(_ctx(4, messages))
    assert len(messages) == 1
    assert "aucune information nouvelle" in messages[0]["content"]


@pytest.mark.asyncio
async def test_overthinking_without_new_info_is_flagged():
    hook = SupervisorHook(_cfg(overthink_tokens=1000))
    usage = SimpleNamespace(output_tokens=2500)
    await hook.after_iteration(_ctx(1, usage=usage))
    messages: list = []
    await hook.before_iteration(_ctx(2, messages))
    assert len(messages) == 1
    assert "très longue" in messages[0]["content"]


@pytest.mark.asyncio
async def test_long_reasoning_with_new_info_is_fine():
    hook = SupervisorHook(_cfg(overthink_tokens=1000))
    await hook.after_execute_tool(_ctx(1), _call(), None, {}, "brand new")
    await hook.after_iteration(_ctx(1, usage=SimpleNamespace(output_tokens=2500)))
    messages: list = []
    await hook.before_iteration(_ctx(2, messages))
    assert messages == []


@pytest.mark.asyncio
async def test_nudges_are_capped_per_turn():
    hook = SupervisorHook(_cfg(max_nudges=2, max_announce_nudges=5))
    assert await hook.before_finalize(_ctx(), "Je vais lancer la génération.")
    assert await hook.before_finalize(_ctx(), "Je vais lancer la génération.")
    assert await hook.before_finalize(_ctx(), "Je vais lancer la génération.") is None


@pytest.mark.asyncio
async def test_announce_nudges_have_their_own_cap():
    hook = SupervisorHook(_cfg(max_announce_nudges=1))
    assert await hook.before_finalize(_ctx(), "Je vais lancer la génération.")
    assert await hook.before_finalize(_ctx(), "Je vais lancer la génération.") is None


@pytest.mark.asyncio
async def test_composite_returns_first_veto_and_isolates_errors():
    class Broken(AgentHook):
        async def before_finalize(self, context, content):
            raise RuntimeError("boom")

    hook = CompositeHook([Broken(), SupervisorHook(_cfg())])
    assert await hook.before_finalize(_ctx(), "Je vais lancer la génération.")
    assert await hook.before_finalize(_ctx(), "Terminé.") is None


def test_factory_skips_off_mode_and_ephemeral_turns():
    from nanobot.agent.hook import AgentTurnHookContext

    assert make_supervisor_hook_factory(_cfg(mode="off"))(AgentTurnHookContext()) is None
    assert make_supervisor_hook_factory(_cfg())(AgentTurnHookContext(ephemeral=True)) is None
    assert isinstance(make_supervisor_hook_factory(_cfg())(AgentTurnHookContext()),
                      SupervisorHook)


def _provider(*responses: LLMResponse) -> MagicMock:
    provider = MagicMock(spec=LLMProvider)
    provider.chat_stream_with_retry = AsyncMock(side_effect=list(responses))
    return provider


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


@pytest.mark.asyncio
async def test_runner_vetoes_announce_then_stop_in_enforce_mode():
    provider = _provider(
        LLMResponse(content="Analyse faite. Je vais maintenant lancer le rendu.",
                    tool_calls=[], usage=None),
        LLMResponse(content="Rendu lancé, terminé.", tool_calls=[], usage=None),
    )
    result = await _run(provider, SupervisorHook(_cfg()))
    assert provider.chat_stream_with_retry.await_count == 2
    assert result.final_content == "Rendu lancé, terminé."
    injected = [m for m in result.messages
                if m.get("role") == "user" and "[supervision]" in str(m.get("content"))]
    assert len(injected) == 1


@pytest.mark.asyncio
async def test_runner_does_not_veto_in_observe_mode():
    provider = _provider(
        LLMResponse(content="Analyse faite. Je vais maintenant lancer le rendu.",
                    tool_calls=[], usage=None),
    )
    result = await _run(provider, SupervisorHook(_cfg(mode="observe")))
    assert provider.chat_stream_with_retry.await_count == 1
    assert result.stop_reason == "completed"


@pytest.mark.asyncio
async def test_runner_stops_after_nudge_cap_even_if_model_keeps_announcing():
    announce = LLMResponse(content="Je vais maintenant lancer le rendu.", tool_calls=[], usage=None)
    provider = _provider(announce, announce, announce, announce)
    result = await _run(provider, SupervisorHook(_cfg(max_announce_nudges=2)))
    assert provider.chat_stream_with_retry.await_count == 3
    assert result.stop_reason == "completed"
