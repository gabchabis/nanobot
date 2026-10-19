"""Tests for ToolResultAgingHook."""

from __future__ import annotations

from pathlib import Path

from nanobot.agent.hook import AgentHookContext, AgentTurnHookContext
from nanobot.agent.hooks.tool_result_aging import (
    AGED_MARKER,
    ToolResultAgingHook,
    make_tool_result_aging_hook_factory,
)
from nanobot.config.schema import ToolResultAgingConfig


def _transcript(sizes: list[int], names: list[str] | None = None) -> list[dict]:
    """user, then one assistant tool call + tool result per entry in ``sizes``."""
    messages: list[dict] = [{"role": "user", "content": "go"}]
    for i, size in enumerate(sizes):
        name = (names or ["web_fetch"] * len(sizes))[i]
        call_id = f"call_{i}"
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": call_id, "type": "function",
                            "function": {"name": name, "arguments": "{}"}}],
        })
        messages.append({"role": "tool", "tool_call_id": call_id,
                         "content": f"R{i}:" + "x" * size})
    return messages


def _hook(tmp_path: Path, **kw) -> ToolResultAgingHook:
    cfg = ToolResultAgingConfig(mode=kw.pop("mode", "enforce"), **kw)
    return ToolResultAgingHook(cfg, workspace=tmp_path, session_key="cli:test")


async def test_collapses_old_keeps_recent_and_archives(tmp_path: Path) -> None:
    messages = _transcript([5000, 5000, 5000, 5000, 5000])
    original = [m.get("content") for m in messages]
    await _hook(tmp_path, keep_recent=2).before_iteration(
        AgentHookContext(iteration=1, messages=messages)
    )
    tools = [m for m in messages if m["role"] == "tool"]
    assert all(t["content"].startswith(AGED_MARKER) for t in tools[:3])
    assert tools[3]["content"].startswith("R3:") and tools[4]["content"].startswith("R4:")
    # ids, names and message count untouched
    assert [t["tool_call_id"] for t in tools] == [f"call_{i}" for i in range(5)]
    assert len(messages) == len(original)
    # the full text is recoverable from disk
    archived = list((tmp_path / ".nanobot" / "tool-results").rglob("call_0.txt"))
    assert len(archived) == 1
    assert archived[0].read_text(encoding="utf-8") == original[2]
    assert "web_fetch" in tools[0]["content"] and str(archived[0].resolve()) in tools[0]["content"]


async def test_observe_mode_does_not_modify(tmp_path: Path) -> None:
    messages = _transcript([5000, 5000, 5000, 5000, 5000])
    before = [dict(m) for m in messages]
    await _hook(tmp_path, mode="observe", keep_recent=1).before_iteration(
        AgentHookContext(iteration=1, messages=messages)
    )
    assert messages == before
    assert not (tmp_path / ".nanobot").exists()


async def test_skips_short_exempt_error_list_and_already_aged(tmp_path: Path) -> None:
    messages = _transcript([100, 5000, 5000, 5000, 5000, 5000],
                           ["exec", "read_file", "exec", "exec", "exec", "exec"])
    messages[6]["content"] = "Error: " + "e" * 5000          # call_2 (error)
    messages[8]["content"] = [{"type": "text", "text": "y" * 5000}]  # call_3 (blocks)
    messages[10]["content"] = f"{AGED_MARKER} - exec, 9000 chars]\nPreview: ..."  # call_4
    snapshot = [m.get("content") for m in messages]
    await _hook(tmp_path, keep_recent=1).before_iteration(
        AgentHookContext(iteration=1, messages=messages)
    )
    assert [m.get("content") for m in messages] == snapshot


async def test_is_idempotent_and_no_workspace_never_drops(tmp_path: Path) -> None:
    messages = _transcript([5000, 5000, 5000, 5000])
    hook = _hook(tmp_path, keep_recent=1)
    ctx = AgentHookContext(iteration=1, messages=messages)
    await hook.before_iteration(ctx)
    once = [m.get("content") for m in messages]
    await hook.before_iteration(ctx)
    assert [m.get("content") for m in messages] == once

    messages2 = _transcript([5000, 5000, 5000])
    snap = [m.get("content") for m in messages2]
    nows = ToolResultAgingHook(ToolResultAgingConfig(mode="enforce", keep_recent=1),
                               workspace=None)
    await nows.before_iteration(AgentHookContext(iteration=1, messages=messages2))
    assert [m.get("content") for m in messages2] == snap  # no archive -> keep original


def test_factory_off_and_ephemeral(tmp_path: Path) -> None:
    off = make_tool_result_aging_hook_factory(ToolResultAgingConfig(mode="off"))
    on = make_tool_result_aging_hook_factory(ToolResultAgingConfig(mode="enforce"))
    assert off(AgentTurnHookContext(workspace=tmp_path)) is None
    assert on(AgentTurnHookContext(workspace=tmp_path, ephemeral=True)) is None
    assert isinstance(on(AgentTurnHookContext(workspace=tmp_path)), ToolResultAgingHook)
