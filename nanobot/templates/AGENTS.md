# Agent Instructions

## Workspace Guidance

Use this file for project-specific preferences, recurring workflow conventions, and instructions you want the agent to remember for this workspace. Keep durable facts about the user in `USER.md`, personality/style guidance in `SOUL.md`, and long-term memory in `memory/MEMORY.md`.

## Proactive Tool Use

- When a request can be satisfied by calling an available tool (displaying/reading a file,
  searching, fetching a page, spawning a subagent, etc.), call the tool directly. Do not wait
  for the user to say "tool", "skill", or name the tool explicitly — infer the need for a
  tool from the request itself.
- If unsure whether a tool applies, prefer attempting the tool call over describing what you
  would do or asking the user to confirm first.

## Scheduled Reminders

- Before scheduling reminders, check available skills and follow skill guidance first.
- Use the built-in `cron` tool to create/list/remove jobs (do not call `nanobot cron` via `exec`).
- Get USER_ID and CHANNEL from the current session (e.g., `8281248569` and `telegram` from `telegram:8281248569`).
- Cron jobs run as scheduled turns in the origin chat/session and normally deliver the result back to that channel. Do not use cron for background checks that should stay silent when there is nothing useful to report; use `HEARTBEAT.md` instead.

**Do NOT just write reminders to MEMORY.md** — that won't trigger actual notifications.

## Heartbeat Tasks

`HEARTBEAT.md` is checked periodically by the protected heartbeat cron job that `nanobot gateway` registers when `gateway.heartbeat.enabled` is true. Do not create a duplicate heartbeat job unless the user has disabled the built-in one and explicitly wants a custom schedule.

- Use `apply_patch` for normal task-list updates, especially when adding, removing, or changing multiple lines.
- Use `edit_file` only for small exact replacements copied from the current `HEARTBEAT.md`.
- Use `write_file` for first creation or intentional full-file rewrites.

When the user asks for a recurring/periodic heartbeat task, or for a periodic background check that should only notify on actionable changes, update `HEARTBEAT.md` instead of creating a one-time reminder. Use the built-in `cron` tool for explicit reminders, scheduled tasks that should report every run, or custom schedules that should not be part of the heartbeat task list.

