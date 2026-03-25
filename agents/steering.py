"""User steering module.

Provides the Strands plugin and registry for injecting real-time user
remarks into running agent conversations, plus the send_steering helper
and the steering event log.
"""

import json
import threading
import time
from datetime import datetime

from strands.plugins import Plugin, hook
from strands.hooks import BeforeModelCallEvent, MessageAddedEvent

from core.state import _log_event, _agent_skills_used, _skills_lock

# ---------------------------------------------------------------------------
# Steering Log
# ---------------------------------------------------------------------------

_steering_log: list[dict] = []
_steering_log_lock = threading.Lock()


def _render_message_content(content) -> str:
    """Convert Strands message content blocks into readable text."""
    if content is None:
        return ""
    if hasattr(content, "model_dump"):
        try:
            content = content.model_dump()
        except Exception:
            content = str(content)
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        content = [content]
    if not isinstance(content, list):
        return str(content)

    parts = []
    for block in content:
        if hasattr(block, "model_dump"):
            try:
                block = block.model_dump()
            except Exception:
                block = str(block)
        if isinstance(block, str):
            parts.append(block)
            continue
        if not isinstance(block, dict):
            parts.append(str(block))
            continue
        if "text" in block:
            parts.append(str(block["text"]))
            continue
        if "toolUse" in block:
            tool_use = block["toolUse"]
            name = tool_use.get("name", "unknown")
            tool_input = tool_use.get("input", {})
            parts.append(
                f"[tool call] {name}({json.dumps(tool_input, ensure_ascii=False)})"
            )
            continue
        if "toolResult" in block:
            tool_result = block["toolResult"]
            status = tool_result.get("status", "unknown")
            result_content = _render_message_content(tool_result.get("content"))
            parts.append(f"[tool result:{status}] {result_content}")
            continue
        parts.append(json.dumps(block, ensure_ascii=False))
    return "\n".join(part for part in parts if part).strip()


class AgentTracePlugin(Plugin):
    """Push assistant conversation messages into the shared event log."""

    name = "agent-trace"

    def __init__(self, agent_name: str):
        super().__init__()
        self.agent_name = agent_name

    @hook
    def on_message_added(self, event: MessageAddedEvent):
        """Log assistant messages so the web UI shows every LLM step."""
        message = event.message
        role = getattr(message, "role", None)
        if role is None and isinstance(message, dict):
            role = message.get("role")
        if role != "assistant":
            return

        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")

        # Track skills usage
        raw = content
        if hasattr(raw, "model_dump"):
            try:
                raw = raw.model_dump()
            except Exception:
                raw = []
        if isinstance(raw, dict):
            raw = [raw]
        if isinstance(raw, list):
            for block in raw:
                if hasattr(block, "model_dump"):
                    try:
                        block = block.model_dump()
                    except Exception:
                        continue
                if isinstance(block, dict) and "toolUse" in block:
                    tu = block["toolUse"]
                    if tu.get("name") == "skills":
                        skill_name = tu.get("input", {}).get("skill_name", "unknown")
                        with _skills_lock:
                            _agent_skills_used.setdefault(self.agent_name, [])
                            if skill_name not in _agent_skills_used[self.agent_name]:
                                _agent_skills_used[self.agent_name].append(skill_name)

        rendered = _render_message_content(content)
        if rendered:
            _log_event(self.agent_name, "THINK", rendered)


# ---------------------------------------------------------------------------
# Strands Plugin
# ---------------------------------------------------------------------------


class UserSteeringPlugin(Plugin):
    """Strands plugin that injects user remarks into agent conversations.

    Queued remarks are injected into the message list on the next LLM call.
    Thread-safe: the dashboard thread writes remarks, agent threads read them.
    """

    name = "user-steering"

    def __init__(self, agent_name: str):
        super().__init__()
        self.agent_name = agent_name
        self._lock = threading.Lock()
        self._pending_remarks: list[dict] = []
        self._injected_ids: set[int] = set()

    def inject_remark(self, remark: str):
        """Queue a remark for injection before the next model invocation."""
        with self._lock:
            self._pending_remarks.append(
                {
                    "id": id(remark) ^ int(time.time() * 1000),
                    "text": remark,
                    "timestamp": time.time(),
                }
            )

    @hook
    def on_before_model_call(self, event: BeforeModelCallEvent):
        """Hook: prepend queued remarks to the agent message list."""
        with self._lock:
            new_remarks = [
                r for r in self._pending_remarks if r["id"] not in self._injected_ids
            ]
            if not new_remarks:
                return
            for r in new_remarks:
                self._injected_ids.add(r["id"])

        remarks_text = "\n".join(f"- {r['text']}" for r in new_remarks)
        injection = (
            "[USER REMARK -- broadcast to all agents]\n"
            "The user just sent the following remark(s) while you are working:\n"
            f"{remarks_text}\n\n"
            "Everyone on the team received this message. "
            "Adapt your current work if this remark concerns you. "
            "If it doesn't concern your role, acknowledge it and continue."
        )
        event.agent.messages.append(
            {
                "role": "user",
                "content": [{"text": injection}],
            }
        )
        _log_event(self.agent_name, "MSG", f"User steering received: {remarks_text}")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class UserSteeringRegistry:
    """Singleton registry mapping agent names to their steering plugins.

    Provides broadcast capability so the dashboard can inject a remark
    into every running agent simultaneously.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._plugins = {}
            return cls._instance

    def register(self, agent_name: str, plugin: UserSteeringPlugin):
        self._plugins[agent_name] = plugin

    def get(self, agent_name: str) -> UserSteeringPlugin | None:
        return self._plugins.get(agent_name)

    def broadcast(self, remark: str):
        """Send a remark to every registered agent plugin."""
        for plugin in self._plugins.values():
            plugin.inject_remark(remark)

    def agent_names(self) -> list[str]:
        return list(self._plugins.keys())

    def clear(self):
        self._plugins.clear()


# ---------------------------------------------------------------------------
# Send Helper
# ---------------------------------------------------------------------------


def send_steering(message: str, target: str = "all"):
    """Dispatch a user steering remark to one or all agents."""
    ts = datetime.now().strftime("%H:%M:%S")
    registry = UserSteeringRegistry()
    if target == "all":
        registry.broadcast(message)
    else:
        plugin = registry.get(target)
        if plugin:
            plugin.inject_remark(message)
    with _steering_log_lock:
        _steering_log.append({"ts": ts, "target": target, "message": message})
    _log_event("user", "STEER", f"[{target}] {message[:80]}")
