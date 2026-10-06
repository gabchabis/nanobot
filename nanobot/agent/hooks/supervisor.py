"""Supervisor hook: keeps weaker (often local) models on track during a turn.

The hook keeps its own deterministic bookkeeping of what the model did, so it does
not depend on the model remembering it. It watches for three failure modes:

* **Going in circles / over-verifying**: the same tool call returning the same
  result again and again, many iterations without any new information, or very
  long reasoning that produced nothing new.
* **Announcing without acting**: a final answer such as "Je vais maintenant
  lancer X." with no tool call, after which the run would otherwise stop.

``mode="observe"`` only logs (grep ``[supervisor]``) so thresholds can be
calibrated on real runs. ``mode="enforce"`` injects one short user message per
detection, bounded by ``max_nudges`` per turn so the supervisor cannot loop.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.agent.hook import AgentHook, AgentHookContext, AgentTurnHookContext, AgentTurnHookFactory

if TYPE_CHECKING:
    from nanobot.config.schema import SupervisorConfig

_ANNOUNCE = re.compile(
    r"\b(?:je\s+vais|je\s+commence|je\s+lance|je\s+proc[èe]de|je\s+m['’]occupe|"
    r"je\s+passe|je\s+poursuis|je\s+continue|maintenant,?\s+je|ensuite,?\s+je|"
    r"d['’]abord,?\s+je|passons|proc[ée]dons|"
    r"let\s+me|i['’]ll|i\s+will|i['’]m\s+going\s+to|i\s+am\s+going\s+to|"
    r"now\s+i|next,?\s+i|let['’]s)\b",
    re.IGNORECASE,
)
# The sentence is an offer or a question to the user, not an unfulfilled promise.
_OFFER = re.compile(
    r"\b(?:si\s+tu\s+veux|si\s+vous\s+voulez|dis-moi|dites-moi|n['’]h[ée]site|"
    r"souhaites?-tu|souhaitez-vous|veux-tu|voulez-vous|"
    r"if\s+you\s+(?:want|like)|let\s+me\s+know|feel\s+free|would\s+you)\b",
    re.IGNORECASE,
)
_DONE = re.compile(
    r"\b(?:termin[ée]e?s?|fini|voil[àa]|c['’]est\s+fait|en\s+r[ée]sum[ée]|"
    r"done|completed|finished|all\s+set)\b",
    re.IGNORECASE,
)
_SENTENCES = re.compile(r"(?<=[.!?:…])\s+|\n+")


def looks_like_unfulfilled_announcement(text: str | None) -> bool:
    """True when a final answer ends by promising an action it did not perform."""
    if not text or not text.strip():
        return False
    tail = text.strip()[-500:]
    sentences = [s.strip() for s in _SENTENCES.split(tail) if s.strip()]
    if not sentences:
        return False
    last = sentences[-1]
    recent = sentences[-2:]
    if "?" in last or _OFFER.search(last):
        return False
    if any(_DONE.search(s) for s in recent):
        return False
    if _ANNOUNCE.search(last):
        return True
    # "Je vais lancer la génération :" followed by nothing.
    trailing_open = tail.rstrip().endswith((":", "…", "..."))
    return trailing_open and any(_ANNOUNCE.search(s) for s in recent)


def _canonical(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def _digest(*parts: str) -> str:
    h = hashlib.sha1()
    for part in parts:
        h.update(part.encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def _looks_failed(result: Any) -> bool:
    return isinstance(result, str) and result.lstrip()[:5].lower() == "error"


class SupervisorHook(AgentHook):
    """Per-turn supervisor. One instance is created for each agent turn."""

    def __init__(self, config: SupervisorConfig, *, session_key: str | None = None) -> None:
        super().__init__()
        self._cfg = config
        self._session_key = session_key
        self._seen: dict[str, int] = {}
        self._actions: list[str] = []
        self._last_new_iteration = 0
        self._new_info_iteration = -1
        self._pending: list[tuple[str, str]] = []
        self._nudges = 0
        self._announce_nudges = 0

    @property
    def enforcing(self) -> bool:
        return self._cfg.mode == "enforce"

    # ------------------------------------------------------------------ ledger
    def _record(self, context: AgentHookContext, tool_call: Any, result: Any) -> None:
        name = getattr(tool_call, "name", "?")
        args = _canonical(getattr(tool_call, "arguments", None))
        failed = _looks_failed(result)
        signature = _digest(name, args, _canonical(result)[:4000])
        count = self._seen.get(signature, 0) + 1
        self._seen[signature] = count

        if count == 1:
            self._last_new_iteration = context.iteration
            self._new_info_iteration = context.iteration
            if not failed:
                self._actions.append(f"{name}({args[:60]})")
                del self._actions[:-8]
            return
        if count >= self._cfg.repeat_threshold or (failed and count >= 2):
            what = "échoue de la même façon" if failed else "renvoie exactement le même résultat"
            done = "; ".join(self._actions[-5:]) or "rien de nouveau"
            self._queue(
                "repeat",
                f"Tu as déjà appelé {name} avec les mêmes arguments {count} fois et cela {what}. "
                f"Ne le refais pas. Déjà établi : {done}. "
                "Passe à l'étape suivante, change d'approche, ou conclus si la tâche est faite.",
            )

    async def after_execute_tool(
        self,
        context: AgentHookContext,
        tool_call: Any,
        tool: Any,
        params: Any,
        result: Any,
    ) -> None:
        self._record(context, tool_call, result)

    async def on_execute_tool_error(
        self,
        context: AgentHookContext,
        tool_call: Any,
        tool: Any,
        params: Any,
        error: Any,
    ) -> None:
        self._record(context, tool_call, f"Error: {error}")

    # --------------------------------------------------------------- detection
    def _queue(self, reason: str, message: str) -> None:
        if all(r != reason for r, _ in self._pending):
            self._pending.append((reason, message))

    async def after_iteration(self, context: AgentHookContext) -> None:
        stalled = context.iteration - self._last_new_iteration
        if stalled >= self._cfg.stall_iterations:
            self._last_new_iteration = context.iteration
            self._queue(
                "stall",
                f"Depuis {stalled} étapes tu n'as obtenu aucune information nouvelle. "
                "Arrête de vérifier : conclus avec ce que tu as, ou change d'approche "
                "(autre outil, autres arguments). Si tu es bloqué, dis précisément ce qui manque.",
            )
        out_tokens = getattr(context.usage, "output_tokens", 0) or 0
        if out_tokens >= self._cfg.overthink_tokens and self._new_info_iteration != context.iteration:
            self._queue(
                "overthink",
                f"Ta dernière réflexion était très longue ({out_tokens} tokens) sans produire "
                "d'information nouvelle. Décide maintenant : exécute l'action suivante ou conclus. "
                "Ne revérifie pas ce qui est déjà établi.",
            )

    # ------------------------------------------------------------------ action
    async def before_iteration(self, context: AgentHookContext) -> None:
        if not self._pending:
            return
        reason, message = self._pending[0]
        self._pending.clear()
        if not self.enforcing or self._nudges >= self._cfg.max_nudges:
            logger.info("[supervisor] {} detected (not enforced): {}", reason, message[:120])
            return
        self._nudges += 1
        logger.info("[supervisor] nudge {}/{} ({}): {}", self._nudges, self._cfg.max_nudges,
                    reason, message[:120])
        context.messages.append({"role": "user", "content": f"[supervision] {message}"})

    async def before_finalize(self, context: AgentHookContext, content: str | None) -> str | None:
        if not looks_like_unfulfilled_announcement(content):
            return None
        preview = (content or "").strip()[-100:]
        if (
            not self.enforcing
            or self._announce_nudges >= self._cfg.max_announce_nudges
            or self._nudges >= self._cfg.max_nudges
        ):
            logger.info("[supervisor] announce-without-act detected (not enforced): …{}", preview)
            return None
        self._nudges += 1
        self._announce_nudges += 1
        logger.info("[supervisor] nudge {}/{} (announce): …{}", self._nudges,
                    self._cfg.max_nudges, preview)
        return (
            "[supervision] Tu as annoncé une action sans l'exécuter. Exécute-la maintenant en "
            "appelant l'outil approprié. Si la tâche est en fait terminée, dis-le explicitement."
        )


def make_supervisor_hook_factory(config: SupervisorConfig) -> AgentTurnHookFactory:
    """Build the per-turn factory registered by ``AgentLoop.from_config``."""

    def _factory(context: AgentTurnHookContext) -> AgentHook | None:
        if config.mode == "off" or context.ephemeral:
            return None
        return SupervisorHook(config, session_key=context.session_key)

    return _factory
