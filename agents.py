"""Agent execution module.

Contains _summarize_history, _manage_sliding_window, run_agent, and
_demo_agent.  These functions drive the persistent agent loop and the
demo simulation used when Strands is unavailable.
"""

import asyncio
import functools
import json
import os
from datetime import datetime
from pathlib import Path

import state as _state_mod
from state import (
    AgentState,
    SLIDING_WINDOW_THRESHOLD,
    _agent_health,
    _agent_inboxes,
    _agent_sleep_durations,
    _agent_states,
    _agent_summaries,
    _agent_turn_count,
    _bb_lock,
    _blackboard,
    _build_iteration_task_multimodal,
    _current_agent_local,
    _health_lock,
    _inbox_lock,
    _log_event,
    _log_to_file,
    _push_status,
    _reviewer_said_pass,
    _all_agents_done,
    _pending_core_agents,
)
from agent_config import AGENT_CONFIGS, AGENT_PREREQS
from agent_tools import make_communication_tools, make_stuck_tool
from steering import AgentTracePlugin, UserSteeringPlugin, UserSteeringRegistry


_browser_instance = None


def _get_browser_tool_local():
    """Return a shared headless browser tool for agent web tasks."""
    global _browser_instance
    if _browser_instance is None:
        os.environ.setdefault("STRANDS_BROWSER_HEADLESS", "true")
        from strands_tools.browser import LocalChromiumBrowser

        _browser_instance = LocalChromiumBrowser(launch_options={"channel": "chrome"})
    return _browser_instance.browser


def _summarize_history(agent_name: str, messages: list[dict]) -> str:
    """Condense older conversation messages into a short summary via the LLM.

    Args:
        agent_name: Agent whose history is being summarized.
        messages: List of message dicts to compress.

    Returns:
        Bullet-point summary string, or a truncated fallback if the LLM is unavailable.
    """
    from state import _llm_provider  # late import to pick up runtime value

    if not messages:
        return ""
    text_parts = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                c.get("text", str(c)) if isinstance(c, dict) else str(c)
                for c in content
            )
        text_parts.append(f"[{role}]: {content[:500]}")

    conversation_text = "\n".join(text_parts)

    if _llm_provider:
        try:
            summary = _llm_provider.complete(
                user_prompt=(
                    "Summarize this agent conversation history in 3-5 bullet points. "
                    "Keep key decisions, deliverables produced, and pending issues:\n\n"
                    f"{conversation_text[:4000]}"
                ),
                system_prompt=(
                    "You are a concise summarizer. Output only bullet points, no preamble."
                ),
            )
            return summary
        except Exception:
            pass

    return (
        f"[Previous work summary -- {len(messages)} exchanges]: "
        + conversation_text[:1000]
    )


def _manage_sliding_window(agent_name: str, agent_obj) -> None:
    """Compress an agent's conversation history when it grows too long.

    Summarises messages beyond the last *keep_count* into a single
    context block, preventing unbounded token growth.

    Args:
        agent_name: The agent whose history should be trimmed.
        agent_obj: The live Strands Agent instance.
    """
    turn_count = _agent_turn_count.get(agent_name, 0)
    if turn_count <= SLIDING_WINDOW_THRESHOLD:
        return

    messages = None
    for attr in ("messages", "_messages", "conversation", "_conversation"):
        messages = getattr(agent_obj, attr, None)
        if messages and isinstance(messages, list):
            break

    if not messages or len(messages) <= SLIDING_WINDOW_THRESHOLD:
        return

    keep_count = 10
    old_messages = messages[:-keep_count]
    recent_messages = messages[-keep_count:]

    existing_summary = _agent_summaries.get(agent_name, "")
    summary = _summarize_history(agent_name, old_messages)

    if existing_summary:
        full_summary = f"{existing_summary}\n\n--- Later ---\n{summary}"
    else:
        full_summary = summary

    _agent_summaries[agent_name] = full_summary

    summary_message = {
        "role": "user",
        "content": (
            "[CONTEXT SUMMARY -- your previous work and conversations]:\n"
            f"{full_summary}\n\n"
            "[End of summary -- the following messages are your recent history]"
        ),
    }

    new_messages = [summary_message] + recent_messages

    for attr in ("messages", "_messages", "conversation", "_conversation"):
        if hasattr(agent_obj, attr) and isinstance(getattr(agent_obj, attr), list):
            setattr(agent_obj, attr, new_messages)
            break

    _agent_turn_count[agent_name] = len(recent_messages)
    _log_event(
        agent_name,
        "PLAN",
        f"Sliding window applied: {len(old_messages)} old messages summarized",
    )


async def run_agent(agent_name: str, specs: str, loop: asyncio.AbstractEventLoop):
    """Execute a persistent agent loop until the swarm terminates.

    The agent is instantiated once and receives enriched context prompts
    each cycle while preserving its full conversation history. Sleep
    duration between cycles is controlled by the LLM via tool calls.

    Args:
        agent_name: Identifier matching a key in AGENT_CONFIGS.
        specs: Project specification string.
        loop: The running asyncio event loop.

    Returns:
        Completion message string.
    """
    frontend_blocks = getattr(_state_mod, "_frontend_blocks", None)

    config = AGENT_CONFIGS[agent_name]
    _log_event(agent_name, "PLAN", f"Starting -- {config['task']}")

    comm_tools = make_communication_tools(agent_name)
    report_stuck_tool = make_stuck_tool(agent_name, loop)

    web_tools = []
    try:
        from strands_tools import tavily

        web_tools.append(tavily)
    except Exception as e:
        _log_event(agent_name, "WARN", f"Tavily tool unavailable: {e}")

    try:
        web_tools.append(_get_browser_tool_local())
    except Exception as e:
        _log_event(agent_name, "WARN", f"Browser tool unavailable: {e}")

    all_tools = comm_tools + [report_stuck_tool] + web_tools

    if agent_name == "analyste":
        with _bb_lock:
            _blackboard["specs"] = {
                "value": specs,
                "author": "system",
                "timestamp": datetime.now().isoformat(),
            }

    agent = None
    try:
        from strands import Agent, AgentSkills

        strands_model = None
        if _state_mod._llm_provider is not None:
            strands_model = _state_mod._llm_provider.to_strands_model()
        final_prompt = config["system_prompt"]
        final_prompt += (
            "\n\nIMPORTANT: Try to solve issues independently first. "
            "If you have failed multiple times or are genuinely blocked, "
            "call report_stuck(). Do not remain stuck without asking for help."
        )

        agent_kwargs: dict = {"system_prompt": final_prompt, "tools": all_tools}
        if strands_model:
            agent_kwargs["model"] = strands_model

        steering_plugin = UserSteeringPlugin(agent_name)
        trace_plugin = AgentTracePlugin(agent_name)
        UserSteeringRegistry().register(agent_name, steering_plugin)
        agent_plugins = [steering_plugin, trace_plugin]

        skills_dir = Path(__file__).parent / "skills"
        enable_skills = os.getenv("SWARM_ENABLE_SKILLS", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if skills_dir.is_dir() and enable_skills:
            try:
                skills_plugin = AgentSkills(skills=str(skills_dir))
                agent_plugins.append(skills_plugin)
            except Exception as e:
                _log_event(agent_name, "WARN", f"AgentSkills disabled: {e}")

        agent_kwargs["plugins"] = agent_plugins

        if _state_mod._tracing_enabled:
            agent_kwargs["trace_attributes"] = {
                "gen_ai.agent.name": agent_name,
                "swarm.agent.role": config.get("task", agent_name),
                "swarm.agent.color": config.get("color", ""),
                "session.id": os.environ.get("SWARM_SESSION_ID", "default"),
            }

        agent = Agent(**agent_kwargs)
    except ImportError:
        await _demo_agent(agent_name, specs)
        return f"[DEMO] {agent_name} completed"

    _push_status(agent_name, AgentState.WAITING)
    _agent_turn_count[agent_name] = 0
    _agent_sleep_durations[agent_name] = 0

    is_first = True

    while not _state_mod._swarm_done.is_set():
        if _agent_states.get(agent_name) == AgentState.DONE and not is_first:
            _log_event(
                agent_name, "PLAN", "DONE -- parking until woken by another agent"
            )
            while not _state_mod._swarm_done.is_set():
                has_messages = False
                with _inbox_lock:
                    has_messages = len(_agent_inboxes.get(agent_name, [])) > 0
                if has_messages:
                    _log_event(
                        agent_name, "ACT", "Woken from DONE -- inbox has messages"
                    )
                    _push_status(agent_name, AgentState.WORKING)
                    break
                await asyncio.sleep(2)
            if _state_mod._swarm_done.is_set():
                break

        sleep_seconds = _agent_sleep_durations.get(agent_name, 0)
        if sleep_seconds > 0 and not is_first:
            _push_status(agent_name, AgentState.WAITING)
            _log_event(agent_name, "PLAN", f"Sleeping {sleep_seconds}s")
            elapsed = 0
            while elapsed < sleep_seconds and not _state_mod._swarm_done.is_set():
                if _agent_sleep_durations.get(agent_name, 0) == 0 and elapsed > 0:
                    _log_event(agent_name, "ACT", "Woken up early by another agent")
                    break
                await asyncio.sleep(min(2, sleep_seconds - elapsed))
                elapsed += 2

        if _state_mod._swarm_done.is_set():
            break

        prereqs = AGENT_PREREQS.get(agent_name, [])
        if prereqs:
            with _bb_lock:
                missing = [k for k in prereqs if k not in _blackboard]
            if missing:
                _log_event(
                    agent_name,
                    "PLAN",
                    f"Prereqs missing: {', '.join(missing)} -- sleeping 15s",
                )
                _push_status(agent_name, AgentState.WAITING)
                await asyncio.sleep(15)
                continue

        _push_status(agent_name, AgentState.WORKING)

        task_prompt = _build_iteration_task_multimodal(
            agent_name,
            config,
            specs,
            is_first,
            frontend_blocks=frontend_blocks,
        )
        is_first = False

        _agent_sleep_durations[agent_name] = 30

        try:
            _log_to_file("AGENT_START", agent_name, f"{str(task_prompt)[:200]}")
            _log_event(agent_name, "PROMPT", str(task_prompt))

            def _invoke_agent(a, tp, name):
                _current_agent_local.name = name
                return str(a(tp))

            result = await asyncio.get_event_loop().run_in_executor(
                None,
                functools.partial(_invoke_agent, agent, task_prompt, agent_name),
            )
            _log_to_file("AGENT_FINISH", agent_name, f"{result}")
            _log_event(agent_name, "RESPONSE", result)

            _agent_turn_count[agent_name] = _agent_turn_count.get(agent_name, 0) + 1

            from state import _memory as mem

            if mem:
                mem.store(
                    f"Agent {agent_name}: {result}",
                    {"agent": agent_name, "type": "completion"},
                )

            _manage_sliding_window(agent_name, agent)

        except Exception as e:
            _log_event(agent_name, "ERROR", f"Agent error: {e}")
            _agent_sleep_durations[agent_name] = 10

        if agent_name == "reviewer" and _reviewer_said_pass():
            if _all_agents_done():
                with _bb_lock:
                    _blackboard["final_status"] = {
                        "value": "PASS",
                        "author": "reviewer",
                        "timestamp": datetime.now().isoformat(),
                    }
                _log_event(
                    "reviewer",
                    "DONE",
                    "Reviewer PASS + all agents DONE -- stopping swarm",
                )
                _state_mod._swarm_done.set()
                break
            else:
                agents_not_done = _pending_core_agents(exclude="reviewer")
                _log_event(
                    "reviewer",
                    "REASON",
                    f"PASS blocked -- agents not done: {', '.join(agents_not_done)}",
                )
                with _inbox_lock:
                    _agent_inboxes["reviewer"].append(
                        {
                            "from": "system",
                            "message": (
                                f"Your PASS verdict is on hold because these agents are "
                                f"not yet DONE: {', '.join(agents_not_done)}. "
                                "Wait for them to finish."
                            ),
                            "urgent": True,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
                _agent_sleep_durations["reviewer"] = 15

        await asyncio.sleep(0)

    _push_status(agent_name, AgentState.DONE)
    _log_event(agent_name, "DONE", "Agent stopped")
    return f"{agent_name} finished"


async def _demo_agent(agent_name: str, specs: str) -> str:
    """Simulate agent work without an LLM for demonstration purposes.

    Args:
        agent_name: Agent to simulate.
        specs: Project specification string.

    Returns:
        Demo completion message.
    """
    import json as _json

    config = AGENT_CONFIGS[agent_name]
    phases = [
        (0.1, "PLAN", f"Reading specs: {specs}", AgentState.WAITING),
        (0.3, "ACT", "Analysing requirements", AgentState.WORKING),
        (0.5, "ACT", "Generating artefacts", AgentState.WORKING),
        (0.7, "REASON", "Checking consistency", AgentState.WORKING),
        (0.9, "ACT", "Writing output files", AgentState.WORKING),
        (1.0, "DONE", "Complete -- sleeping until needed", AgentState.DONE),
    ]

    stagger = {
        "analyste": 0,
        "architecte": 1,
        "devops": 0.5,
        "developpeur": 1.5,
        "qa": 2.5,
    }
    await asyncio.sleep(stagger.get(agent_name, 0))

    _push_status(agent_name, AgentState.WAITING)

    for score, phase, msg, state in phases:
        await asyncio.sleep(0.8 + (hash(agent_name) % 5) * 0.2)
        _push_status(agent_name, state)
        _log_event(agent_name, phase, msg)
        with _health_lock:
            if agent_name in _agent_health:
                _agent_health[agent_name]["progress"] = score
                _agent_health[agent_name]["status"] = state.value

    Path("./output/src").mkdir(parents=True, exist_ok=True)
    Path("./output/tests").mkdir(parents=True, exist_ok=True)
    Path("./output/adr").mkdir(parents=True, exist_ok=True)
    Path("./output/diagrams").mkdir(parents=True, exist_ok=True)

    if agent_name == "analyste":
        Path("./output/feature_list.json").write_text(
            _json.dumps(
                {
                    "modules": ["auth", "tasks", "dashboard"],
                    "language": "python",
                    "framework": "fastapi",
                },
                indent=2,
            )
        )
    elif agent_name == "architecte":
        Path("./output/diagrams/c4_context.mmd").write_text(
            "graph TD\n  User --> ToDoApp\n  ToDoApp --> DB[(Database)]"
        )
        Path("./output/shared_dependencies.json").write_text(
            _json.dumps(
                {
                    "UserService": {"methods": ["get_user", "create_user"]},
                    "TaskService": {"methods": ["get_tasks", "create_task"]},
                },
                indent=2,
            )
        )
    elif agent_name == "devops":
        Path("./output/devops").mkdir(parents=True, exist_ok=True)
        Path("./output/devops/Dockerfile").write_text(
            "FROM python:3.11-slim\n"
            "WORKDIR /app\n"
            "COPY . .\n"
            "RUN pip install -r requirements.txt\n"
            'CMD ["uvicorn", "main:app"]'
        )
        Path("./output/devops/README.md").write_text(
            "# DevOps Run Guide\n\n"
            "## Local\n"
            "1. Install dependencies\n"
            "2. Start the generated app from output/src\n\n"
            "## Container\n"
            "1. Build Docker image\n"
            "2. Run compose stack\n"
        )
        with _bb_lock:
            _blackboard["devops_ready"] = {
                "value": "PASS: environment templates prepared",
                "author": agent_name,
                "timestamp": datetime.now().isoformat(),
            }
    elif agent_name == "developpeur":
        Path("./output/src/main.py").write_text(
            "from fastapi import FastAPI\n"
            "app = FastAPI()\n\n"
            "@app.get('/')\n"
            "def root(): return {'status': 'ok'}"
        )
        Path("./output/tests/test_main.py").write_text(
            "from fastapi.testclient import TestClient\n"
            "from src.main import app\n"
            "client = TestClient(app)\n\n"
            "def test_root():\n"
            "    r = client.get('/')\n"
            "    assert r.status_code == 200"
        )
        with _bb_lock:
            _blackboard["code_ready"] = {
                "value": "PASS: code generated",
                "author": agent_name,
                "timestamp": datetime.now().isoformat(),
            }
    elif agent_name == "qa":
        Path("./output/qa_report.md").write_text(
            "# QA Report\n\n"
            "## Results\n"
            "- Tests run: 8\n"
            "- Passed: 8\n"
            "- Failed: 0\n"
            "- Coverage: 84%"
        )
        with _bb_lock:
            _blackboard["test_results"] = {
                "value": "PASS - Tests run: 8, Failed: 0, Coverage: 84%",
                "author": agent_name,
                "timestamp": datetime.now().isoformat(),
            }
            _blackboard["qa_status"] = {
                "value": "PASS",
                "author": agent_name,
                "timestamp": datetime.now().isoformat(),
            }
    elif agent_name == "reviewer":
        Path("./output/review_report.md").write_text(
            "# Review Report\n\n"
            "- Launch checks: PASS\n"
            "- Code quality: PASS\n"
            "- Recommendations: continue\n"
        )
        with _bb_lock:
            _blackboard["review_status"] = {
                "value": "PASS",
                "author": agent_name,
                "timestamp": datetime.now().isoformat(),
            }
            _blackboard["launch_status"] = {
                "value": "PASS",
                "author": agent_name,
                "timestamp": datetime.now().isoformat(),
            }

    return f"[DEMO] {agent_name} completed successfully"
