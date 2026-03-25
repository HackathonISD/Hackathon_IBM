"""Agent tool factories module.

Provides _strands_tool instrumentation wrapper, make_communication_tools
factory (creates the full set of blackboard/inbox/memory/filesystem/command
tools for one agent), make_stuck_tool (spawner request tool), and
ALL_TOOL_NAMES constant.
"""

import asyncio
import functools
import json
import queue
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from state import (
    AgentState,
    MAX_SLEEP_SECONDS,
    _agent_health,
    _agent_inboxes,
    _agent_sleep_durations,
    _agent_states,
    _bb_lock,
    _blackboard,
    _current_agent_local,
    _event_lock,
    _futures_lock,
    _health_lock,
    _inbox_lock,
    _log_event,
    _log_to_file,
    _memory,
    _output_root,
    _push_status,
    _resolve_agent_path,
    _stuck_futures,
    _stuck_queue,
    _workspace_root,
)

# ---------------------------------------------------------------------------
# Tool Decoration Helper
# ---------------------------------------------------------------------------

ALL_TOOL_NAMES = [
    "write_file",
    "read_file",
    "list_files",
    "run_command",
    "write_blackboard",
    "read_blackboard",
    "send_message",
    "check_inbox",
    "memory_store",
    "memory_search",
    "report_progress",
]


def _preview_text(value, limit: int = 800) -> str:
    """Return a readable preview string for UI/event logging."""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False)
        except Exception:
            text = str(value)
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def _strands_tool(fn, owner: str | None = None):
    """Wrap a callable with Strands @tool and event-logging instrumentation.

    Args:
        fn: The function to decorate.
        owner: Agent name used for log attribution.

    Returns:
        Decorated tool function.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        caller = owner or getattr(_current_agent_local, "name", None) or "system"
        call_desc = f"{fn.__name__}({', '.join(repr(a) for a in args[:3])})"
        _log_to_file("TOOL_CALL", fn.__name__, f"args={args} kwargs={kwargs}")
        _log_event(caller, "TOOL", f"Called: {call_desc}")
        try:
            res = fn(*args, **kwargs)
            _log_to_file("TOOL_RESULT", fn.__name__, f"Result: {res}")
            res_str = str(res) if res is not None else "None"
            _log_event(caller, "TOOL", f"Result of {fn.__name__}: {res_str}")
            return res
        except Exception as e:
            _log_to_file("TOOL_ERROR", fn.__name__, f"Error: {e}")
            _log_event(caller, "TOOL", f"Error in {fn.__name__}: {e}")
            raise

    try:
        from strands import tool

        return tool(wrapper)
    except (ImportError, Exception):
        return wrapper


# ---------------------------------------------------------------------------
# Communication Tool Factory
# ---------------------------------------------------------------------------


def make_communication_tools(agent_name: str):
    """Create the full set of agent communication tools bound to *agent_name*.

    Returns:
        List of Strands-decorated tool callables for blackboard, inbox,
        memory, filesystem, command execution, and lifecycle management.
    """
    # Import here to avoid circular reference at module load time
    from state import _memory as _mem  # noqa: F401 — accessed via global name

    def write_blackboard(key: str, value: str) -> str:
        """Write a result to the shared blackboard. All agents will see it."""
        from state import _memory as mem

        with _bb_lock:
            _blackboard[key] = {
                "value": value,
                "author": agent_name,
                "timestamp": datetime.now().isoformat(),
            }
        _log_event(
            agent_name,
            "ACT",
            f"Wrote to blackboard: {key} = {_preview_text(value)}",
        )
        if mem:
            mem.store(
                f"[blackboard:{key}] {value}",
                {"agent": agent_name, "type": "blackboard", "key": key},
            )
        return f"Written: {key}"

    def read_blackboard(key: str = "") -> str:
        """Read from the blackboard. Leave key empty to see all entries."""
        with _bb_lock:
            if key:
                entry = _blackboard.get(key)
                return json.dumps(entry) if entry else "Not found"
            return json.dumps({k: v["value"] for k, v in _blackboard.items()})

    def send_message(target_agent: str, message: str) -> str:
        """Send a direct message to another agent."""
        with _inbox_lock:
            if target_agent not in _agent_inboxes:
                return f"Unknown agent: {target_agent}"
            _agent_inboxes[target_agent].append(
                {
                    "from": agent_name,
                    "message": message,
                    "timestamp": datetime.now().isoformat(),
                }
            )
        _agent_sleep_durations[target_agent] = 0
        preview = _preview_text(message)
        _log_event(agent_name, "MSG", f"To {target_agent}: {preview}")
        _log_event(target_agent, "MSG", f"From {agent_name}: {preview}")
        return f"Message delivered to {target_agent}"

    def check_inbox() -> str:
        """Check your inbox for messages from other agents or specialists."""
        with _inbox_lock:
            msgs = _agent_inboxes.get(agent_name, [])
            if not msgs:
                return "No new messages"
            result = json.dumps(msgs)
            _agent_inboxes[agent_name] = []
            return result

    def memory_store(content: str, category: str = "general") -> str:
        """Save information to long-term hybrid memory (BM25 + semantic)."""
        from state import _memory as mem

        if not mem:
            return "Memory not initialized"
        doc_id = mem.store(
            content,
            {"agent": agent_name, "category": category},
        )
        _log_event(agent_name, "ACT", f"Stored in memory: {content}")
        return f"Stored with id: {doc_id}"

    def memory_search(query: str, top_k: int = 5) -> str:
        """Search long-term memory using hybrid BM25 + semantic retrieval."""
        from state import _memory as mem

        if not mem:
            return "Memory not initialized"
        results = mem.search(query, top_k=top_k)
        _log_event(
            agent_name, "REASON", f"Memory search: '{query}' -> {len(results)} results"
        )
        if not results:
            return "No relevant memories found"
        return json.dumps(
            [{"text": r["text"], "meta": r["metadata"]} for r in results],
            indent=2,
        )

    def write_file(filepath: str, content: str) -> str:
        """Write content to a file. Use repo/... to target workspace files; default is ./output."""
        try:
            out = _resolve_agent_path(filepath, default_to_output=True)
        except ValueError as e:
            return f"Invalid path: {e}"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(content, encoding="utf-8")
        rel = str(out.relative_to(_workspace_root)).replace("\\", "/")
        _log_event(agent_name, "ACT", f"Wrote file: {rel}")
        with _inbox_lock:
            for other in _agent_inboxes:
                if other != agent_name:
                    _agent_inboxes[other].append(
                        {
                            "from": agent_name,
                            "message": f"New file available: {rel}",
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
        return f"File written: {rel}"

    def read_file(filepath: str, start_line: int = 1, end_line: int = 500) -> str:
        """Read a file by line range to avoid token limits. Default is lines 1-500."""
        try:
            out = _resolve_agent_path(filepath, default_to_output=True)
        except ValueError as e:
            return f"Invalid path: {e}"
        if not out.exists():
            return f"File not found: {filepath}"

        try:
            lines = out.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            return "Error: File is binary or not UTF-8 encoded."

        total_lines = len(lines)
        if start_line < 1:
            start_line = 1
        if end_line > total_lines:
            end_line = total_lines
        if start_line > end_line:
            return (
                f"Invalid range: start_line={start_line} > end_line={end_line}. "
                f"Total lines: {total_lines}"
            )

        content = "\n".join(
            f"{i}: {line}"
            for i, line in enumerate(lines[start_line - 1 : end_line], start=start_line)
        )

        meta = f"--- File: {filepath} (Lines {start_line}-{end_line} of {total_lines}) ---\n"
        if end_line < total_lines:
            meta += (
                f"Note: File incomplete. Use read_file with start_line={end_line + 1} "
                "to read more.\n"
            )

        return meta + content + "\n--- EOF ---"

    def list_files(directory: str = "") -> str:
        """List files. Use repo/... for workspace folders; default is ./output."""
        try:
            base = _resolve_agent_path(directory or "", default_to_output=True)
        except ValueError as e:
            return f"Invalid path: {e}"
        if not base.exists():
            return "Directory not found"
        files = [
            str(p.relative_to(_workspace_root)).replace("\\", "/")
            for p in base.rglob("*")
            if p.is_file()
        ]

        if len(files) > 100:
            return json.dumps(
                {
                    "message": (
                        f"Too many files ({len(files)}). Returning first 100. "
                        "Be more specific."
                    ),
                    "files": files[:100],
                }
            )

        return json.dumps(files)

    def run_command(command: str, cwd: str = ".", timeout: int = 30) -> str:
        """Run a shell command on Windows (cmd.exe) or Linux/macOS (bash).

        Stdout and stderr are captured separately.  Automatically detects when
        the process waits for input (password, y/n, username, number, etc.) by
        reading stderr in real time, then prompts the user and resumes.
        Kills the process and returns an error if total runtime exceeds timeout.
        cwd defaults to workspace root; relative paths like 'output/x' are safe.
        """
        try:
            workdir = _resolve_agent_path(cwd, default_to_output=False)
        except ValueError as e:
            return json.dumps(
                {
                    "cwd": cwd,
                    "exit_code": 1,
                    "stdout": "",
                    "stderr": f"Invalid cwd: {e}",
                },
                ensure_ascii=False,
            )

        if not workdir.exists():
            workdir.mkdir(parents=True, exist_ok=True)

        _log_event(agent_name, "ACT", f"Run command: {command}")

        stream_encoding = (
            f"cp{subprocess.check_output('chcp', shell=True).decode('ascii', errors='ignore').strip().split()[-1]}"
            if sys.platform == "win32"
            else "utf-8"
        )

        def stream_reader(stream, q: queue.Queue):
            try:
                while True:
                    ch = stream.read(1)
                    if not ch:
                        q.put("")
                        break
                    q.put(ch.decode(stream_encoding, errors="replace"))
            except Exception:
                q.put("")

        def drain_queue(q: queue.Queue, silence_threshold: float) -> tuple[str, bool]:
            """Drain character queue until EOF or silence exceeds threshold."""
            chunk = []
            deadline = time.time() + silence_threshold
            while time.time() < deadline:
                try:
                    ch = q.get(timeout=0.05)
                    if ch == "":
                        return "".join(chunk), False
                    chunk.append(ch)
                    deadline = time.time() + silence_threshold
                except queue.Empty:
                    pass
            return "".join(chunk), True

        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []

        try:
            process = subprocess.Popen(
                command,
                cwd=str(workdir),
                shell=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )

            stdout_q: queue.Queue = queue.Queue()
            stderr_q: queue.Queue = queue.Queue()

            threading.Thread(
                target=stream_reader, args=(process.stdout, stdout_q), daemon=True
            ).start()
            threading.Thread(
                target=stream_reader, args=(process.stderr, stderr_q), daemon=True
            ).start()

            start_time = time.time()
            stdout_done = False
            stderr_done = False

            DRAIN_WINDOW = 0.5
            PROMPT_SILENCE_WITH_PROMPT = 2.0
            PROMPT_SILENCE_NO_PROMPT = 120.0
            stderr_silent_since: float | None = None
            stdout_silent_since: float | None = None
            stderr_prompt_pos = 0
            stdout_prompt_pos = 0
            prompt_count = 0
            MAX_PROMPTS = 3
            any_output_ever = False

            while not (stdout_done and stderr_done):
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    process.kill()
                    process.wait(timeout=5)
                    return json.dumps(
                        {
                            "cwd": str(workdir),
                            "exit_code": -1,
                            "stdout": "".join(stdout_chunks).strip(),
                            "stderr": (
                                f"Process killed: exceeded timeout of {timeout}s.\n"
                                + "".join(stderr_chunks).strip()
                            ),
                        },
                        ensure_ascii=False,
                    )

                if not stderr_done:
                    chunk, waiting = drain_queue(stderr_q, DRAIN_WINDOW)
                    if chunk:
                        stderr_chunks.append(chunk)
                        stderr_silent_since = None
                        any_output_ever = True
                    if not waiting:
                        stderr_done = True
                        stderr_silent_since = None
                    elif not chunk and stderr_silent_since is None:
                        stderr_silent_since = time.time()

                if not stdout_done:
                    chunk, waiting = drain_queue(stdout_q, DRAIN_WINDOW)
                    if chunk:
                        stdout_chunks.append(chunk)
                        stdout_silent_since = None
                        any_output_ever = True
                    if not waiting:
                        stdout_done = True
                        stdout_silent_since = None
                    elif not chunk and stdout_silent_since is None:
                        stdout_silent_since = time.time()

                if process.poll() is None:
                    raw_stderr = "".join(stderr_chunks)
                    raw_stdout = "".join(stdout_chunks)

                    stderr_silence = (
                        (time.time() - stderr_silent_since)
                        if stderr_silent_since is not None
                        else 0.0
                    )
                    stdout_silence = (
                        (time.time() - stdout_silent_since)
                        if stdout_silent_since is not None
                        else 0.0
                    )

                    stderr_has_partial_line = raw_stderr and not raw_stderr.endswith(
                        "\n"
                    )
                    stdout_has_partial_line = raw_stdout and not raw_stdout.endswith(
                        "\n"
                    )

                    if (
                        not stderr_done
                        and stderr_has_partial_line
                        and stderr_silence >= PROMPT_SILENCE_WITH_PROMPT
                        and prompt_count < MAX_PROMPTS
                    ):
                        last_stderr = raw_stderr.rstrip()
                        new_text = last_stderr[stderr_prompt_pos:].strip()
                        prompt_line = (
                            new_text if new_text else last_stderr.splitlines()[-1]
                        )
                        print("\n[Command is waiting for input]")
                        print(f"  > {prompt_line}")
                        user_answer = input("Your answer: ")
                        process.stdin.write((user_answer + "\n").encode("utf-8"))
                        process.stdin.flush()
                        prompt_count += 1
                        stderr_prompt_pos = len(last_stderr)
                        stderr_silent_since = None
                        stdout_silent_since = None

                    elif (
                        stdout_has_partial_line
                        and stdout_silence >= PROMPT_SILENCE_WITH_PROMPT
                        and prompt_count < MAX_PROMPTS
                    ):
                        last_stdout = raw_stdout.rstrip()
                        new_text = last_stdout[stdout_prompt_pos:].strip()
                        prompt_line = (
                            new_text if new_text else last_stdout.splitlines()[-1]
                        )
                        print("\n[Command is waiting for input]")
                        print(f"  > {prompt_line}")
                        user_answer = input("Your answer: ")
                        process.stdin.write((user_answer + "\n").encode("utf-8"))
                        process.stdin.flush()
                        prompt_count += 1
                        stdout_prompt_pos = len(last_stdout)
                        stderr_silent_since = None
                        stdout_silent_since = None

                    else:
                        if (
                            stderr_silence >= PROMPT_SILENCE_NO_PROMPT
                            and stdout_silence >= PROMPT_SILENCE_NO_PROMPT
                            and prompt_count < MAX_PROMPTS
                            and not any_output_ever
                        ):
                            print(
                                "\n[Command appears to be waiting for input "
                                "(no prompt detected)]"
                            )
                            user_answer = input(
                                "Your answer (or press Enter to skip): "
                            )
                            if user_answer:
                                process.stdin.write(
                                    (user_answer + "\n").encode("utf-8")
                                )
                                process.stdin.flush()
                            prompt_count += 1
                            stderr_silent_since = None
                            stdout_silent_since = None

            remaining = max(0, timeout - (time.time() - start_time))
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
                return json.dumps(
                    {
                        "cwd": str(workdir),
                        "exit_code": -1,
                        "stdout": "".join(stdout_chunks).strip(),
                        "stderr": (
                            f"Process killed: exceeded timeout of {timeout}s.\n"
                            + "".join(stderr_chunks).strip()
                        ),
                    },
                    ensure_ascii=False,
                )

            return json.dumps(
                {
                    "cwd": str(workdir),
                    "exit_code": process.returncode,
                    "stdout": "".join(stdout_chunks).strip(),
                    "stderr": "".join(stderr_chunks).strip(),
                },
                ensure_ascii=False,
            )

        except Exception as e:
            return json.dumps(
                {
                    "cwd": str(workdir),
                    "exit_code": 1,
                    "stdout": "".join(stdout_chunks).strip(),
                    "stderr": str(e),
                },
                ensure_ascii=False,
            )

    def report_progress(score: float, message: str) -> str:
        """Report your current progress (score 0.0-1.0). Used by dashboard."""
        _log_event(
            agent_name,
            "PLAN" if score < 0.3 else "ACT" if score < 0.8 else "DONE",
            message,
        )
        with _health_lock:
            if agent_name in _agent_health:
                _agent_health[agent_name]["progress"] = score
        return "Progress reported"

    def set_sleep_duration(seconds: int, reason: str = "") -> str:
        """Decide how long to sleep before your next cycle (5-180 seconds).

        Call this when you have nothing to do right now.  The system will wake
        you up after this duration, or earlier if you receive a message.
        """
        capped = max(5, min(int(seconds), MAX_SLEEP_SECONDS))
        _agent_sleep_durations[agent_name] = capped
        _log_event(agent_name, "PLAN", f"Will sleep {capped}s -- {reason}")
        return f"Sleep set to {capped}s"

    def set_status(status: str) -> str:
        """Set your current status: WAITING, WORKING, DONE, or BLOCKED.

        - WAITING: you need something from another agent before you can work
        - WORKING: you are actively producing deliverables
        - DONE: you have delivered, but you can be woken up if needed
        - BLOCKED: you are stuck and need specialist help (use report_stuck instead)

        Note: once DONE, only wake_agent() from another agent can reactivate you.
        """
        try:
            new_state = AgentState(status.upper())
        except ValueError:
            return f"Invalid status '{status}'. Use: WAITING, WORKING, DONE, BLOCKED"
        current = _agent_states.get(agent_name)
        if current == AgentState.DONE and new_state != AgentState.DONE:
            _log_event(
                agent_name,
                "ACT",
                f"Ignored status change {current.value} -> {new_state.value} "
                "(must be woken by another agent)",
            )
            return "Cannot change status from DONE. Another agent must wake you with wake_agent()."
        _push_status(agent_name, new_state)
        _log_event(agent_name, "ACT", f"Status -> {new_state.value}")
        with _health_lock:
            if agent_name in _agent_health:
                _agent_health[agent_name]["status"] = new_state.value
        return f"Status set to {new_state.value}"

    def wake_agent(target_agent: str, reason: str) -> str:
        """Wake up another agent by sending them an urgent message.

        Use this when you need something from them or want to give them feedback.
        This resets their sleep timer so they act on your message quickly.
        """
        with _inbox_lock:
            if target_agent not in _agent_inboxes:
                return f"Unknown agent: {target_agent}"
            _agent_inboxes[target_agent].append(
                {
                    "from": agent_name,
                    "message": reason,
                    "urgent": True,
                    "timestamp": datetime.now().isoformat(),
                }
            )
        _agent_sleep_durations[target_agent] = 0
        _log_event(agent_name, "ACT", f"Woke up {target_agent}: {reason}")
        return f"{target_agent} woken up with message"

    def submit_verdict(
        verdict: str,
        reason: str,
        tests_pass: bool = False,
        launch_ok: bool = False,
        review_ok: bool = False,
        agents_confirmed: str = "",
    ) -> str:
        """REVIEWER ONLY -- Submit the final quality verdict.

        verdict must be 'PASS' or 'FAIL'.
        agents_confirmed is a comma-separated list of agent names that confirmed
        completion.  This writes the reviewer_verdict to the blackboard in the
        exact format the system expects.
        """
        from state import _memory as mem

        if agent_name != "reviewer":
            return "ERROR: Only the reviewer agent can submit a verdict."
        verdict_upper = verdict.strip().upper()
        if verdict_upper not in ("PASS", "FAIL"):
            return f"ERROR: verdict must be 'PASS' or 'FAIL', got '{verdict}'."
        confirmed_list = (
            [a.strip() for a in agents_confirmed.split(",") if a.strip()]
            if agents_confirmed
            else []
        )
        verdict_obj = {
            "tests_pass": bool(tests_pass),
            "launch_ok": bool(launch_ok),
            "review_ok": bool(review_ok),
            "verdict": verdict_upper,
            "reason": reason,
            "agents_confirmed": confirmed_list,
        }
        verdict_json = json.dumps(verdict_obj)
        with _bb_lock:
            _blackboard["reviewer_verdict"] = {
                "value": verdict_json,
                "author": "reviewer",
                "timestamp": datetime.now().isoformat(),
            }
        _log_event("reviewer", "ACT", f"Verdict submitted: {verdict_upper} -- {reason}")
        if mem:
            mem.store(
                f"[blackboard:reviewer_verdict] {verdict_json}",
                {"agent": "reviewer", "type": "blackboard", "key": "reviewer_verdict"},
            )
        return f"Verdict '{verdict_upper}' submitted successfully."

    return [
        _strands_tool(write_blackboard, owner=agent_name),
        _strands_tool(read_blackboard, owner=agent_name),
        _strands_tool(send_message, owner=agent_name),
        _strands_tool(check_inbox, owner=agent_name),
        _strands_tool(memory_store, owner=agent_name),
        _strands_tool(memory_search, owner=agent_name),
        _strands_tool(write_file, owner=agent_name),
        _strands_tool(read_file, owner=agent_name),
        _strands_tool(list_files, owner=agent_name),
        _strands_tool(run_command, owner=agent_name),
        _strands_tool(report_progress, owner=agent_name),
        _strands_tool(set_sleep_duration, owner=agent_name),
        _strands_tool(set_status, owner=agent_name),
        _strands_tool(wake_agent, owner=agent_name),
        _strands_tool(submit_verdict, owner=agent_name),
    ]


# ---------------------------------------------------------------------------
# Stuck Tool Factory
# ---------------------------------------------------------------------------


def make_stuck_tool(agent_name: str, loop: asyncio.AbstractEventLoop):
    """Create the report_stuck tool for an agent.

    The returned tool blocks the calling agent until the spawner creates
    and runs a specialist to resolve the reported issue.

    Args:
        agent_name: Identifier of the agent that owns this tool.
        loop: The running asyncio event loop for future resolution.

    Returns:
        Strands-decorated tool callable.
    """

    def report_stuck(
        task_description: str,
        what_i_tried: str,
        what_i_need: str,
        partial_work: str = "",
    ) -> str:
        """Call this when you cannot proceed after 2+ attempts.

        A specialist will be spawned to help you.  This call BLOCKS until the
        specialist finishes.  Returns the specialist's output -- use it to
        continue your task.
        """
        request_id = f"{agent_name}_{uuid.uuid4().hex[:8]}"
        future = loop.create_future()

        with _futures_lock:
            _stuck_futures[request_id] = future

        _log_event(agent_name, "PLAN", f"STUCK -- requesting specialist: {what_i_need}")

        asyncio.run_coroutine_threadsafe(
            _stuck_queue.put(
                {
                    "request_id": request_id,
                    "agent_name": agent_name,
                    "task_description": task_description,
                    "what_i_tried": what_i_tried,
                    "what_i_need": what_i_need,
                    "partial_work": partial_work,
                }
            ),
            loop,
        )

        result_container: dict = {"result": None, "error": None}

        def wait_for_result():
            try:
                future_result = asyncio.run_coroutine_threadsafe(
                    asyncio.wrap_future(future), loop
                ).result()
                result_container["result"] = future_result
            except Exception as e:
                result_container["error"] = str(e)

        wait_thread = threading.Thread(target=wait_for_result)
        wait_thread.start()
        wait_thread.join()

        if result_container["error"]:
            return (
                f"Specialist failed: {result_container['error']}. "
                "Try a different approach."
            )
        return (
            f"Specialist result:\n{result_container['result']}\n\n"
            "Continue your task using this."
        )

    return _strands_tool(report_stuck, owner=agent_name)
