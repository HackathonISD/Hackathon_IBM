"""Shared state and infrastructure module.

Contains all module-level globals (blackboard, inboxes, event log, health
dicts, stuck queue/futures, memory, agent state tracking), the AgentState
enum, path-resolution helpers, logging helpers, iteration task builders,
and related constants.

All other modules import shared state from here rather than from main.
"""

import asyncio
import json
import threading
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from memory import HybridMemory

# ---------------------------------------------------------------------------
# Communication Infrastructure
# ---------------------------------------------------------------------------

_blackboard: dict = {}
_bb_lock = threading.Lock()

_agent_inboxes: dict[str, list] = {
    "analyste": [],
    "architecte": [],
    "devops": [],
    "developpeur": [],
    "qa": [],
    "reviewer": [],
    "spawner": [],
}
_inbox_lock = threading.Lock()

_stuck_queue: asyncio.Queue = None

_stuck_futures: dict[str, asyncio.Future] = {}
_futures_lock = threading.Lock()

_event_log: list[dict] = []
_event_lock = threading.Lock()

_agent_health: dict[str, dict] = {}
_health_lock = threading.Lock()

_memory: Optional[HybridMemory] = None
_workspace_root = Path(".").resolve()
_output_root = (_workspace_root / "output").resolve()
_use_rich_display = True
_use_web_ui = False

_current_agent_local = threading.local()

try:
    import web_dashboard as _web_dashboard_module

    _WEB_UI_AVAILABLE = True
except Exception:
    _web_dashboard_module = None
    _WEB_UI_AVAILABLE = False

_swarm_done: asyncio.Event = None

# Tracing flag — set to True by main._setup_tracing()
_tracing_enabled: bool = False

# Active LLM provider — set to a BaseLLMProvider instance by main.main()
_llm_provider: Optional[object] = None

# ---------------------------------------------------------------------------
# Agent State
# ---------------------------------------------------------------------------


class AgentState(Enum):
    """Lifecycle states for a team agent."""

    WAITING = "WAITING"
    WORKING = "WORKING"
    DONE = "DONE"
    BLOCKED = "BLOCKED"


_agent_states: dict[str, AgentState] = {}
_agent_sleep_durations: dict[str, int] = {}
_agent_turn_count: dict[str, int] = {}
_agent_summaries: dict[str, str] = {}
_debate_tracker: dict[str, int] = {}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SLIDING_WINDOW_THRESHOLD = 10
MAX_SLEEP_SECONDS = 180
MAX_DEBATE_EXCHANGES = 2

TEAM_DESCRIPTION = """
== YOUR TEAM ==
You work in a parallel team of 6 persistent agents. Everyone runs simultaneously.

- ANALYSTE: Analyses project specs -> produces feature_list.json, ADR documents
- ARCHITECTE: Designs architecture -> produces C4 diagrams, shared_dependencies.json
- DEVELOPPEUR: Writes all source code and tests -> produces files in output/src/ and output/tests/
- DEVOPS: Creates CI/CD, Docker, requirements -> produces files in output/devops/
- QA: Executes tests, measures coverage -> produces qa_report.md
- REVIEWER: Final quality gatekeeper -> validates everything, writes PASS/FAIL verdict

== COMMUNICATION RULES ==
- When you finish a deliverable, ALWAYS notify the agents who need it via send_message()
- When you're waiting for something, check the blackboard first, then message the agent you need
- You can read other agents' deliverables and critique them IF they misuse YOUR deliverable
- Critique is a DISCUSSION, not an order. You might be wrong. Ask questions first.
- If you disagree with another agent after 2 exchanges on the same topic, escalate to the reviewer
- When you have nothing to do, call set_sleep_duration() with how long you want to wait (max 180s)
- When you finish your work, call set_status("DONE") -- but stay ready, you can be woken up

== DELIVERABLE DEFENSE ==
- You are responsible for YOUR deliverables being correctly used by others
- When you wake up, check if other agents have used your output correctly
- If someone misuses your deliverable, send them a message ASKING why (don't assume you're right)
- Example: "I defined AuthService in shared_dependencies.json but you used plain functions in auth.py. Was that intentional? If so, can you explain why?"
"""

# ---------------------------------------------------------------------------
# Path Helpers
# ---------------------------------------------------------------------------


def _resolve_agent_path(path: str, default_to_output: bool = True) -> Path:
    """Resolve a user-supplied path to an absolute path within the workspace.

    Args:
        path: Raw path string from an agent tool call.
        default_to_output: Route bare relative paths into the output directory.

    Returns:
        Resolved absolute Path guaranteed to be inside the workspace.

    Raises:
        ValueError: If the resolved path escapes the workspace root.
    """
    raw = (path or "").strip().replace("\\", "/")

    if not raw or raw == ".":
        target = _output_root if default_to_output else _workspace_root
    elif raw == "repo":
        target = _workspace_root
    elif raw.startswith("repo/"):
        target = _workspace_root / raw[len("repo/") :]
    elif raw == "output":
        target = _workspace_root / "output"
    elif raw.startswith("output/"):
        target = _workspace_root / raw
    elif raw.startswith("./"):
        target = (
            (_output_root / raw[2:])
            if default_to_output
            else (_workspace_root / raw[2:])
        )
    elif Path(raw).is_absolute():
        target = Path(raw)
    else:
        target = (_output_root / raw) if default_to_output else (_workspace_root / raw)

    resolved = target.resolve()
    try:
        resolved.relative_to(_workspace_root)
    except ValueError:
        raise ValueError("Path escapes workspace")
    return resolved


def _blackboard_text(key: str) -> str:
    """Return the lowercased string value of a blackboard entry."""
    with _bb_lock:
        entry = _blackboard.get(key)
    if entry is None:
        return ""
    if isinstance(entry, dict):
        value = entry.get("value", "")
    else:
        value = entry
    return str(value).lower()


def _reviewer_said_pass() -> bool:
    """Return True if the blackboard contains a reviewer PASS verdict."""
    raw = _blackboard_text("reviewer_verdict")
    if not raw:
        return False
    try:
        verdict = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(verdict, dict):
        return False
    return verdict.get("verdict", "").upper() == "PASS"


def _all_agents_done(exclude: str = "reviewer") -> bool:
    """Check whether all agents except *exclude* have reached DONE state."""
    for name, state in _agent_states.items():
        if name == exclude:
            continue
        if state != AgentState.DONE:
            return False
    return True


def _output_file_count() -> int:
    """Count the total number of files under the output directory."""
    if not _output_root.exists():
        return 0
    return sum(1 for p in _output_root.rglob("*") if p.is_file())


# ---------------------------------------------------------------------------
# Logging Helpers
# ---------------------------------------------------------------------------

_trace_lock = threading.Lock()


def _log_to_file(category: str, source: str, message: str):
    """Append a timestamped log entry to the agent trace file."""
    try:
        log_path = _output_root / "agent_trace.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().isoformat(timespec="seconds")
        line = f"[{ts}] [{category}] [{source}] {message}\n"
        with _trace_lock:
            with open(log_path, "a", encoding="utf-8", errors="replace") as f:
                f.write(line)
    except Exception:
        pass


def _log_event(agent: str, phase: str, msg: str):
    """Record an event in the in-memory event log and optionally to the web UI."""
    ts = datetime.now().strftime("%H:%M:%S")
    entry = {"ts": ts, "agent": agent, "phase": phase, "msg": msg}
    with _event_lock:
        _event_log.append(entry)

    with _health_lock:
        if agent not in _agent_health:
            _agent_health[agent] = {
                "status": "IDLE",
                "phase": phase,
                "last_msg": msg[:120],
                "steps": 0,
                "total": 6,
                "progress": 0.0,
            }
        else:
            _agent_health[agent]["phase"] = phase
            _agent_health[agent]["last_msg"] = msg[:120]
            _agent_health[agent]["steps"] = _agent_health[agent].get("steps", 0) + 1

    if _use_web_ui and _WEB_UI_AVAILABLE and _web_dashboard_module:
        try:
            _web_dashboard_module.push_event(agent, phase, msg)
        except Exception:
            pass


def _push_status(agent: str, state: AgentState):
    """Update an agent's state in both the state dict and health table."""
    _agent_states[agent] = state

    with _health_lock:
        if agent not in _agent_health:
            _agent_health[agent] = {
                "status": state.value,
                "phase": "IDLE",
                "last_msg": "",
                "steps": 0,
                "total": 6,
                "progress": 0.0,
            }
        else:
            _agent_health[agent]["status"] = state.value

    if _use_web_ui and _WEB_UI_AVAILABLE and _web_dashboard_module:
        try:
            _web_dashboard_module.push_agent_status(agent, state.value)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Iteration Task Builders
# ---------------------------------------------------------------------------


def _build_iteration_task(
    agent_name: str, config: dict, specs: str, is_first: bool
) -> str:
    """Assemble a context-enriched prompt for the next agent iteration.

    Includes current team state, blackboard snapshot, and drained inbox
    messages so the persistent agent has full situational awareness.

    Args:
        agent_name: Identifier of the target agent.
        config: Agent configuration dict from AGENT_CONFIGS.
        specs: Project specification string.
        is_first: Whether this is the agent's first cycle.

    Returns:
        Plain-text prompt string.
    """
    parts = []

    if is_first:
        parts.append(f"PROJECT SPECS: {specs}\n")
        parts.append(f"YOUR TASK: {config['task']}")

    state_lines = []
    for name, state in _agent_states.items():
        if name != agent_name:
            sleep_info = ""
            sleep_s = _agent_sleep_durations.get(name, 0)
            if sleep_s > 0 and state == AgentState.WAITING:
                sleep_info = f" (sleeping {sleep_s}s)"
            state_lines.append(f"  {name}: {state.value}{sleep_info}")
    if state_lines:
        parts.append(f"\nTeam status:\n" + "\n".join(state_lines))

    with _bb_lock:
        bb_summary = {k: str(v.get("value", ""))[:300] for k, v in _blackboard.items()}
    parts.append(f"\nCurrent blackboard state:\n{json.dumps(bb_summary, indent=2)}")

    with _inbox_lock:
        msgs = list(_agent_inboxes.get(agent_name, []))
        _agent_inboxes[agent_name] = []
    if msgs:
        urgent_msgs = [m for m in msgs if m.get("urgent")]
        normal_msgs = [m for m in msgs if not m.get("urgent")]
        if urgent_msgs:
            parts.append(f"\nURGENT messages:\n{json.dumps(urgent_msgs, indent=2)}")
        if normal_msgs:
            parts.append(f"\nInbox messages:\n{json.dumps(normal_msgs, indent=2)}")

    if not is_first:
        parts.append(
            "\nDecide what to do: work on deliverables, respond to messages, "
            "or call set_sleep_duration() if waiting for others. "
            "Call set_status() to reflect your current state."
        )

    return "\n".join(parts)


def _build_iteration_task_multimodal(
    agent_name: str,
    config: dict,
    specs: str,
    is_first: bool,
    frontend_blocks: list[dict] | None = None,
) -> list[dict] | str:
    """Build a multimodal prompt with optional frontend reference blocks.

    Args:
        agent_name: Target agent identifier.
        config: Agent configuration dict.
        specs: Project specification string.
        is_first: Whether this is the first iteration.
        frontend_blocks: Optional list of Strands ContentBlock dicts.

    Returns:
        A list of Strands ContentBlock dicts when a frontend file is
        attached and the agent should see it, otherwise a plain string.
    """
    text = _build_iteration_task(agent_name, config, specs, is_first)

    if not frontend_blocks:
        return text

    if not is_first or agent_name not in (
        "analyste",
        "architecte",
        "reviewer",
        "developpeur",
    ):
        return text

    ref_note = (
        "\n\n== FRONTEND REFERENCE DOCUMENT ==\n"
        "A reference file has been provided (see attached).\n"
        "- If it is a UI mockup or screenshot: reproduce the layout, colors, "
        "components, and visual style as closely as possible in the frontend code.\n"
        "- If it is a chart or data visualization: respect the chart type, axes, "
        "color scheme, and data representation.\n"
        "- If it is a multi-page PDF: study ALL pages for layout, components, "
        "and design requirements.\n"
        "Your implementation MUST visually match this reference. "
        "Write your assessment of the document to the blackboard key 'frontend_ref_analysis' "
        "before writing any code."
    )

    return frontend_blocks + [{"text": text + ref_note}]
