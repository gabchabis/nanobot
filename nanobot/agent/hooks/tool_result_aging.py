"""Tool-result aging hook: shrinks *old* tool outputs once they stopped being useful.

Why: with a small context window (e.g. 32k on a local model) a handful of
``web_fetch`` / ``exec`` results that each fit under ``max_tool_result_chars``
still add up and push the turn into truncation loops. The existing offload only
looks at the size of a single result; it never looks at *age*.

What: before every model call, tool results older than the last ``keep_recent``
ones and longer than ``min_chars`` are written in full to the workspace
(``.nanobot/tool-results/<session>/<tool_call_id>.txt``, the same place the
built-in offload uses) and replaced in the transcript by a short stub: tool name,
original size, a short preview and the path to re-read with ``read_file``.
Nothing is lost, it is just no longer paid for on every iteration.

``mode="observe"`` only logs what would be saved (grep ``[tool-aging]``) so the
thresholds can be calibrated; ``mode="enforce"`` rewrites the transcript.
Only the ``content`` of ``role="tool"`` messages is touched, never ids, names or
the number of messages, so compaction boundaries stay valid.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.agent.hook import AgentHook, AgentHookContext, AgentTurnHookContext, AgentTurnHookFactory
from nanobot.utils.helpers import (
    _TOOL_RESULTS_DIR,
    _write_text_atomic,
    ensure_dir,
    safe_filename,
)

if TYPE_CHECKING:
    from nanobot.config.schema import ToolResultAgingConfig

AGED_MARKER = "[tool result archived"


def _call_names(messages: list[dict[str, Any]]) -> dict[str, str]:
    """Map tool_call_id -> tool name from assistant messages (best effort)."""
    names: dict[str, str] = {}
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            call_id = call.get("id")
            fn = call.get("function")
            name = fn.get("name") if isinstance(fn, dict) else call.get("name")
            if isinstance(call_id, str) and isinstance(name, str):
                names[call_id] = name
    return names


def render_stub(tool_name: str, size: int, preview: str, path: str | None) -> str:
    """Compact replacement for an aged tool result."""
    flat = " ".join(preview.split())
    where = (
        f"Full output: {path} (use read_file with offset/limit if you really need it)."
        if path
        else "Full output not kept."
    )
    return (
        f"{AGED_MARKER} - {tool_name}, {size} chars]\n"
        f"Preview: {flat}\n"
        f"{where}"
    )


class ToolResultAgingHook(AgentHook):
    """Per-turn hook. One instance is created for each agent turn."""

    def __init__(
        self,
        config: ToolResultAgingConfig,
        *,
        workspace: Path | None,
        session_key: str | None = None,
    ) -> None:
        super().__init__()
        self._cfg = config
        self._workspace = workspace
        self._session_key = session_key
        self._saved_chars = 0
        self._collapsed = 0

    @property
    def enforcing(self) -> bool:
        return self._cfg.mode == "enforce"

    def _archive(self, tool_call_id: str, text: str) -> str | None:
        """Write the full text to the workspace; return its path, or None on failure."""
        if self._workspace is None:
            return None
        try:
            root = ensure_dir(self._workspace / _TOOL_RESULTS_DIR)
            bucket = ensure_dir(root / safe_filename(self._session_key or "default"))
            path = bucket / f"{safe_filename(tool_call_id)}.txt"
            if not path.exists():
                _write_text_atomic(path, text)
            return str(path.resolve())
        except Exception:
            logger.exception("[tool-aging] archive failed for {}", tool_call_id)
            return None

    async def before_iteration(self, context: AgentHookContext) -> None:
        messages = context.messages
        tool_indexes = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
        keep = max(self._cfg.keep_recent, 0)
        candidates = tool_indexes[:-keep] if keep else tool_indexes
        if not candidates:
            return

        names = _call_names(messages)
        exempt = set(self._cfg.exempt_tools)
        collapsed = 0
        saved = 0
        for idx in candidates:
            message = messages[idx]
            content = message.get("content")
            # Only plain-text results: lists may carry images/blocks we must not flatten.
            if not isinstance(content, str) or content.startswith(AGED_MARKER):
                continue
            if len(content) < self._cfg.min_chars:
                continue
            call_id = str(message.get("tool_call_id") or f"tool_{idx}")
            name = str(message.get("name") or names.get(call_id) or "tool")
            if name in exempt:
                continue
            # Never shrink an error: the model may still be reasoning about it.
            if content.lstrip().lower().startswith("error"):
                continue

            if not self.enforcing:
                saved += max(len(content) - self._cfg.preview_chars, 0)
                collapsed += 1
                continue

            path = self._archive(call_id, content)
            if path is None:
                continue  # cannot archive -> never drop the only copy
            stub = render_stub(name, len(content), content[: self._cfg.preview_chars], path)
            if len(stub) >= len(content):
                continue
            message["content"] = stub
            saved += len(content) - len(stub)
            collapsed += 1

        if not collapsed:
            return
        self._collapsed += collapsed
        self._saved_chars += saved
        logger.info(
            "[tool-aging] {} {} result(s) at iteration {}, ~{} chars (~{} tokens) saved, "
            "{} chars total this turn",
            "collapsed" if self.enforcing else "would collapse",
            collapsed,
            context.iteration,
            saved,
            saved // 4,
            self._saved_chars,
        )


def make_tool_result_aging_hook_factory(config: ToolResultAgingConfig) -> AgentTurnHookFactory:
    """Build the per-turn factory registered by ``AgentLoop.from_config``."""

    def _factory(context: AgentTurnHookContext) -> AgentHook | None:
        if config.mode == "off" or context.ephemeral:
            return None
        return ToolResultAgingHook(
            config,
            workspace=context.workspace,
            session_key=context.session_key,
        )

    return _factory
