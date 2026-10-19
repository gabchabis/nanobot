"""Concrete agent hook implementations."""

from nanobot.agent.hooks.file_edit_activity import (
    FileEditActivityHook,
    create_file_edit_activity_hook,
)
from nanobot.agent.hooks.supervisor import SupervisorHook, make_supervisor_hook_factory
from nanobot.agent.hooks.tool_result_aging import (
    ToolResultAgingHook,
    make_tool_result_aging_hook_factory,
)

__all__ = [
    "FileEditActivityHook",
    "SupervisorHook",
    "ToolResultAgingHook",
    "create_file_edit_activity_hook",
    "make_supervisor_hook_factory",
    "make_tool_result_aging_hook_factory",
]
