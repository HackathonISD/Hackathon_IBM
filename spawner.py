"""Spawner module.

Contains create_specialist (runs a one-shot specialist agent) and
spawner_loop (background coroutine that listens on the stuck queue and
delegates to specialists).
"""

import asyncio
from datetime import datetime
import state as _state_mod

from state import (
    _bb_lock,
    _blackboard,
    _futures_lock,
    _log_event,
    _log_to_file,
    _stuck_futures,
)
from agent_tools import make_communication_tools, _strands_tool


def create_specialist(
    name: str,
    system_prompt: str,
    task: str,
    tool_names: list[str],
    context: str = "",
) -> str:
    """Instantiate a one-shot specialist agent, execute its task, and return output.

    Args:
        name: Descriptive identifier for the specialist.
        system_prompt: System instructions tailored to the specialist's expertise.
        task: The concrete task the specialist must complete.
        tool_names: Subset of tool names the specialist should receive.
        context: Optional context or partial work from the stuck agent.

    Returns:
        The specialist's textual output.
    """
    _log_event("spawner", "ACT", f"Creating specialist: {name}")

    comm_tools = make_communication_tools(f"specialist:{name}")
    tool_map = {t.__name__: t for t in comm_tools}
    resolved_tools = [tool_map[n] for n in tool_names if n in tool_map]

    full_prompt = system_prompt
    if context:
        full_prompt += f"\n\nContext from the stuck agent:\n{context}"
    full_prompt += "\n\nAlways write your output to the blackboard and relevant files."

    try:
        from strands import Agent

        strands_model = None
        if _state_mod._llm_provider is not None:
            strands_model = _state_mod._llm_provider.to_strands_model()
        agent_kwargs: dict = {"system_prompt": full_prompt, "tools": resolved_tools}
        if strands_model:
            agent_kwargs["model"] = strands_model

        specialist = Agent(**agent_kwargs)
        _log_to_file("SPECIALIST_START", name, f"Task: {task}")
        _log_event(f"specialist:{name}", "PROMPT", task)
        result = str(specialist(task))
        _log_to_file("SPECIALIST_FINISH", name, f"Result: {result}")
        _log_event(f"specialist:{name}", "RESPONSE", result)
        _log_event("spawner", "DONE", f"Specialist {name} finished")

        with _bb_lock:
            _blackboard[f"specialist:{name}"] = {
                "value": result,
                "author": f"specialist:{name}",
                "timestamp": datetime.now().isoformat(),
            }

        return result

    except ImportError:
        result = f"[DEMO] Specialist {name} completed task: {task}"
        _log_event("spawner", "DONE", f"[DEMO] Specialist {name} done")
        return result


async def spawner_loop(loop: asyncio.AbstractEventLoop):
    """Background coroutine that processes stuck-agent requests.

    Listens on the stuck queue, analyses each request, and delegates to
    specialist agents created via the LLM-powered spawner agent.

    Args:
        loop: The active asyncio event loop.
    """
    _log_event("spawner", "PLAN", "Spawner ready -- listening for stuck agents")

    spawner_comm_tools = make_communication_tools("spawner")

    try:
        from strands import Agent

        def spawn_specialist_tool(
            specialist_name: str,
            system_prompt: str,
            task: str,
            tools: list[str],
            context: str = "",
        ) -> str:
            """Create a new specialist agent and run it on the task."""
            return create_specialist(
                specialist_name, system_prompt, task, tools, context
            )

        spawn_specialist_tool = _strands_tool(spawn_specialist_tool, owner="spawner")

        strands_model = None
        if _state_mod._llm_provider is not None:
            strands_model = _state_mod._llm_provider.to_strands_model()
        spawner_kwargs: dict = {
            "system_prompt": """You are the Spawner -- the agent factory.
Your job: when a team agent is stuck, analyze their problem and spawn the right specialist.

When you receive a stuck request:
1. Analyze what specific expertise is needed
2. Call spawn_specialist_tool() with:
   - specialist_name: descriptive name (e.g. "typescript_generic_expert")
   - system_prompt: precise instructions for the specialist
   - task: the exact task to complete
   - tools: subset of ["write_file","read_file","write_blackboard","memory_store","memory_search","run_command"]
   - context: the partial work from the stuck agent
3. Return the specialist's output directly

Specialist types you can create:
- Language specialists (TypeScript, Go, Rust, Java...)
- Framework specialists (FastAPI, NestJS, Django...)
- Debugging specialists (test failures, import errors...)
- Documentation specialists (C4 diagrams, ADRs...)
- DevOps specialists (CI/CD, Docker, GitHub Actions...)
- Database specialists (SQL, migrations, ORM...)

Always use the minimal tools needed for the specialist.""",
            "tools": [spawn_specialist_tool] + spawner_comm_tools,
        }
        if strands_model:
            spawner_kwargs["model"] = strands_model

        spawner_agent = Agent(**spawner_kwargs)

    except ImportError:
        spawner_agent = None

    while True:
        try:
            request = await asyncio.wait_for(_state_mod._stuck_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            _log_event("spawner", "DONE", "Spawner stopped")
            raise

        request_id = request["request_id"]
        _log_event(
            "spawner",
            "REASON",
            f"Received stuck request from {request['agent_name']}: {request['what_i_need']}",
        )

        prompt = f"""
Agent '{request['agent_name']}' is stuck.

Task they were doing: {request['task_description']}
What they tried: {request['what_i_tried']}
What they need: {request['what_i_need']}
Partial work: {request['partial_work'] if request['partial_work'] else 'none'}

Spawn the right specialist and solve this problem.
"""

        try:
            if spawner_agent:
                _log_to_file("SPAWNER_START", "spawner", f"Prompt: {prompt}")
                result = str(spawner_agent(prompt))
                _log_to_file("SPAWNER_FINISH", "spawner", f"Result: {result}")
            else:
                result = f"[DEMO] Specialist created for: {request['what_i_need']}"
                _log_event(
                    "spawner", "DONE", f"[DEMO] Resolved: {request['agent_name']}"
                )
        except Exception as e:
            result = (
                f"Spawner error: {e}. The stuck agent should try a simpler approach."
            )
            _log_event("spawner", "PLAN", f"Spawner error: {e}")

        with _futures_lock:
            future = _stuck_futures.pop(request_id, None)
        if future and not future.done():
            loop.call_soon_threadsafe(future.set_result, result)

        _state_mod._stuck_queue.task_done()
