"""Spawn tool for creating background subagents."""

# pyright: reportIncompatibleMethodOverride=false

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool, ToolResult, tool_parameters
from nanobot.agent.tools.context import current_request_context
from nanobot.agent.tools.schema import (
    BooleanSchema,
    NumberSchema,
    StringSchema,
    tool_parameters_schema,
)
from nanobot.security.workspace_access import current_workspace_scope

if TYPE_CHECKING:
    from nanobot.agent.subagent import SubagentManager
    from nanobot.agent.tools.context import ToolContext


@tool_parameters(
    tool_parameters_schema(
        task=StringSchema("The task for the subagent to complete"),
        label=StringSchema("Optional short label for the task (for display)"),
        model_preset=StringSchema(
            description=(
                "Optional named model preset for this subagent (e.g. 'secondary' for "
                "complex or demanding tasks). Defaults to the current session's model."
            ),
        ),
        temperature=NumberSchema(
            description=(
                "Optional sampling temperature for the subagent "
                "(0.0 = deterministic, higher = more creative). "
                "Defaults to the provider's configured temperature."
            ),
            minimum=0.0,
            maximum=2.0,
        ),
        wait=BooleanSchema(
            description=(
                "Wait for the subagent and return its result directly. Use this for a "
                "blocking consultation that must inform the current turn. Defaults to "
                "false for background execution."
            ),
            default=False,
        ),
        required=["task"],
    )
)
class SpawnTool(Tool):
    """Tool to spawn a subagent for background task execution."""

    def __init__(self, manager: "SubagentManager", runtime_control: "RuntimeControl | None" = None):
        self._manager = manager
        self._runtime_control = runtime_control

    @classmethod
    def create(cls, ctx: ToolContext) -> Tool:
        manager = ctx.subagent_manager
        if manager is None:
            raise RuntimeError("SpawnTool requires an initialized subagent manager")
        return cls(manager=manager, runtime_control=ctx.runtime_control)

    @property
    def name(self) -> str:
        return "spawn"

    @property
    def description(self) -> str:
        return (
            "Spawn a subagent to handle a task in the background. "
            "Use this for complex or time-consuming tasks that can run independently. "
            "Set wait=true for a consultation whose result must inform the current turn. "
            "Set model_preset to run the subagent on a different (e.g. stronger) model "
            "than the current session. "
            "The subagent will complete the task and report back when done. "
            "For deliverables or existing projects, inspect the workspace first "
            "and use a dedicated subdirectory when helpful."
        )

    @property
    def concurrency_safe(self) -> bool:
        """Each call owns its task state; the manager serializes capacity admission."""
        return True

    async def execute(
        self,
        task: str,
        label: str | None = None,
        model_preset: str | None = None,
        temperature: float | None = None,
        wait: bool = False,
        **kwargs: Any,
    ) -> str:
        """Spawn a subagent to execute the given task."""
        request_ctx = current_request_context()
        if request_ctx is None or request_ctx.runtime is None:
            return ToolResult.error("Error: spawn requires an active model runtime")

        runtime = request_ctx.runtime
        if model_preset:
            if self._runtime_control is None:
                return ToolResult.error(
                    "Error: model_preset override is not available in this context"
                )
            try:
                runtime = self._runtime_control.resolve_preset(model_preset)
            except KeyError as exc:
                return ToolResult.error(f"Error: {exc}")
                
        origin_channel = request_ctx.channel
        origin_chat_id = request_ctx.chat_id
        session_key = request_ctx.session_key or f"{origin_channel}:{origin_chat_id}"
        method = self._manager.run_inline if wait else self._manager.spawn
        return await method(
            task=task,
            runtime=request_ctx.runtime,
            label=label,
            origin_channel=origin_channel,
            origin_chat_id=origin_chat_id,
            session_key=session_key,
            origin_message_id=request_ctx.message_id,
            temperature=temperature,
            workspace_scope=current_workspace_scope(),
        )
