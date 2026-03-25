"""Deployment agent module.

Provides an autonomous deployment agent that manages the full lifecycle
of shipping a project: environment detection, Git workflow, CI pipeline
setup, and platform deployment.

Usage::

    python deployement.py --specs "Your project description"
"""

import asyncio
import json
import os
import shutil
import subprocess
import argparse
import functools
import threading
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------- #
#                             LLM Provider Setup                              #
# --------------------------------------------------------------------------- #
try:
    from llm.llm_provider import LLMRouter, BaseLLMProvider

    _LLM_AVAILABLE = True
except ImportError:
    _LLM_AVAILABLE = False
    LLMRouter = None


SLIDING_WINDOW_THRESHOLD = 10

DEPLOYEUR_PROMPT = """You are the Deployeur — an autonomous deployment engineer.
You read the project, prepare it for deployment, manage the full Git workflow, and ship it.

### Execution model — IMPORTANT

**NEVER respond with plain text until your task is completed or can't be achieved.**
A text-only response terminates the agent loop immediately, so any remaining phases will be skipped.

**Execute one tool call at a time.**
**Assume that all tests have passed — no need to execute them again.**

### Your tools

You have exactly 4 tools:

| Tool | Purpose |
|---|---|
| list_files(directory="") | List all files in the project (or a subdirectory). Returns JSON array of relative paths. |
| read_file(filepath) | Read a file by its relative path and return its content. |
| write_file(filepath, content) | Write (create or overwrite) a file. Parent directories are created automatically. |
| run_command(command, cwd=".", timeout=120) | Run a shell command inside the project directory. Returns JSON with exit_code, stdout, stderr. On Windows use PowerShell syntax, on Linux/macOS use bash. |

You have NO other tools. Do NOT try to call anything else.

---

## Phase 1 — Understand the Environment

Use run_command() to detect:
- OS and shell (Windows → PowerShell, Linux/macOS → bash)
- Every relevant tool installed: git, docker, node, python, etc.
- Whether elevated privileges are available

---

## Phase 2 — Read the Project

Call list_files() to discover the project structure.
Then call read_file() on every relevant file:
- Language/runtime version files
- Dependency manifests (package.json, requirements.txt, etc.)
- Start/build scripts, Makefile, Procfile, README
- Environment variable files (.env, .env.example)

Fields to capture internally:

| Field | What to capture |
|---|---|
| Language & runtime | Language name and required version |
| Framework | What framework the app uses |
| Build command | How to compile or bundle |
| Start command | How to run in production |
| Port | What port the app listens on |
| Health check path | The HTTP path to probe, if any |
| External services | Databases, caches, queues, third-party APIs |
| Environment variables | Every variable the app requires |
| Existing deployment files | Every deployment-related file found and what strategy it implies |

---

## Phase 3 — Ensure Git Repo & CI Pipeline

### Step 1 — CI Pipeline

Use list_files(".github/workflows") to check if a CI YAML with a test step exists.
If not, use write_file(".github/workflows/ci.yml", content) to create one.

### Step 2 — Git and GitHub

Use run_command() for all git operations:
- If no local repo: run_command("git init") then commit.
- If no remote: run_command("gh repo create ... --source=. --push").
- Detect branch, ensure local/remote are in sync.
- If gh is not authenticated, stop and tell the user.

---

## Phase 4 — Deployment

### Step 1 — Choose strategy

Infer from Phase 2 findings, priority order:
1. Platform config file exists → use that platform (fly.toml → Fly.io, render.yaml → Render, etc.)
2. Dockerfile exists and Docker running → container-based
3. Managed PaaS CLI authenticated → use that PaaS
4. Cloud CLI authenticated (AWS, GCP, Azure) → VM-based
5. Default → Fly.io

### Step 2 — Pre-deployment checks

Use run_command() to verify platform CLI is installed and authenticated.
If not, stop and tell the user.

### Step 3 — Provision infrastructure (VM/cloud only)

Skip for PaaS. For cloud VM: provision via run_command() (VPC, subnet, security group, instance).

### Step 4 — Install and start

Adapt every run_command() call to the OS detected in Phase 1.
- PaaS: standard deploy command via run_command()
- VM/SSH: copy code, install deps, start app — all via run_command()
- Docker: build and run containers via run_command()

### Step 5 — Verify

Poll with run_command() until the app responds with HTTP 200 (up to 20 attempts, 10s apart).

On Windows:
```
run_command('for ($i=1;$i -le 20;$i++){try{$s=(Invoke-WebRequest -Uri "http://<host>:<port>" -UseBasicParsing).StatusCode;if($s -eq 200){Write-Host "Live";break}}catch{};Start-Sleep -Seconds 10}')
```

If after 20 attempts the app is not responding, use run_command() to check logs and report.

---

## Phase 5 — Final report

Only produce a text response once ALL of the following are confirmed:
- Local repo with at least one commit
- Remote repo with origin configured
- Current branch fully pushed
- CI pipeline file committed and pushed
- Deployment target running
- HTTP 200 confirmed from public address

Then output:
```
## Deployment complete

- **URL:** http://<public-address>:<port>
- **Platform:** <platform>
- **Branch:** <branch>
- **CI pipeline:** <file path>
```
"""


class DeploymentAgent:
    """Autonomous deployment agent.

    Copies the generated project into a deployment workspace, creates a
    Strands-based LLM agent equipped with file and shell tools, and
    iteratively drives it through environment detection, Git setup, CI
    pipeline creation, and platform deployment.

    Attributes:
        specs: Project specification string.
        workspace_root: Resolved root directory of the workspace.
        output_root: Resolved path to the current output directory.
        final_report: Deployment report produced upon successful completion.
    """

    def __init__(self, specs: str, workspace_root: str = "."):
        """Initialize the deployment agent.

        Args:
            specs: Project specification describing what was built.
            workspace_root: Path to the workspace root directory.
        """
        self.specs = specs
        self.workspace_root = Path(workspace_root).resolve()
        self.output_root = (self.workspace_root / "output").resolve()

        self._llm_provider = None
        self._event_log: list[dict] = []
        self._agent_health: dict[str, dict] = {}
        self._turn_count: int = 0
        self._agent_summaries: dict[str, str] = {}
        self.final_report: str | None = None

        self._trace_lock = threading.Lock()
        self._event_lock = threading.Lock()
        self._health_lock = threading.Lock()

    # ----------------------------------------------------------------------- #
    #                           LLM Initialization                            #
    # ----------------------------------------------------------------------- #

    def init_llm(self):
        """Initialize the LLM provider from environment configuration."""
        if self._llm_provider is not None:
            self._log_event(
                "system", "ACT", f"LLM provider (pre-set): {self._llm_provider}"
            )
            return
        if _LLM_AVAILABLE and LLMRouter:
            try:
                self._llm_provider = LLMRouter.from_env()
                self._log_event("system", "ACT", f"LLM provider: {self._llm_provider}")
            except Exception as e:
                print(f"[LLM] Failed to init: {e}")
        else:
            print("[LLM] llm_provider not available")

    def _get_strands_model(self):
        """Return a Strands-compatible model wrapper from the active provider."""
        if self._llm_provider is None:
            return None
        return self._llm_provider.to_strands_model()

    # ----------------------------------------------------------------------- #
    #                              Event Logging                              #
    # ----------------------------------------------------------------------- #

    def _log_to_file(self, category: str, source: str, message: str):
        """Append a structured log entry to the trace log file.

        Args:
            category: Log category identifier.
            source: Originating agent or system component.
            message: Log message content.
        """
        try:
            log_path = self.output_root / "agent_trace.log"
            if not log_path.parent.exists():
                log_path.parent.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().isoformat()
            with self._trace_lock:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"[{ts}] [{category}] [{source}] {message}\n")
        except Exception:
            pass

    def _log_event(self, agent: str, phase: str, message: str):
        """Record an event and broadcast it to all logging sinks.

        Args:
            agent: Name of the agent producing the event.
            phase: Current execution phase (e.g. ACT, PLAN, DONE).
            message: Human-readable event description.
        """
        self._log_to_file(phase, agent, message)
        ts = datetime.now().strftime("%H:%M:%S")
        with self._event_lock:
            self._event_log.append(
                {"ts": ts, "agent": agent, "phase": phase, "msg": message}
            )
        try:
            import ui.web_dashboard as web_dashboard

            web_dashboard.push_event(agent, phase, message)
        except Exception:
            pass
        print(f"[{ts}] {agent:<12} {phase:<6} {message}")
        with self._health_lock:
            if agent not in self._agent_health:
                self._agent_health[agent] = {
                    "status": "IDLE",
                    "phase": phase,
                    "last_msg": message,
                    "steps": 0,
                    "total": 6,
                }
            else:
                self._agent_health[agent].update(
                    {
                        "status": "RUNNING" if phase != "DONE" else "DONE",
                        "phase": phase,
                        "last_msg": message,
                    }
                )
                self._agent_health[agent]["steps"] = min(
                    self._agent_health[agent].get("steps", 0) + 1,
                    self._agent_health[agent].get("total", 6),
                )

    # ----------------------------------------------------------------------- #
    #                            Path Resolution                              #
    # ----------------------------------------------------------------------- #

    def _resolve_path(self, path: str, default_to_output: bool = True) -> Path:
        """Resolve a relative or symbolic path to an absolute workspace path.

        Args:
            path: Raw path string (may use ``repo/`` or ``output/`` prefixes).
            default_to_output: When ``True``, bare relative paths resolve
                under the output directory; otherwise under workspace root.

        Returns:
            Resolved absolute ``Path`` within the workspace boundary.

        Raises:
            ValueError: If the resolved path escapes the workspace root.
        """
        raw = (path or "").strip().replace("\\", "/")

        if not raw or raw == ".":
            target = self.output_root if default_to_output else self.workspace_root
        elif raw == "repo":
            target = self.workspace_root
        elif raw.startswith("repo/"):
            target = self.workspace_root / raw[len("repo/") :]
        elif raw == "output":
            target = self.workspace_root / "output"
        elif raw.startswith("output/"):
            target = self.workspace_root / raw
        elif raw.startswith("./"):
            target = (
                (self.output_root / raw[2:])
                if default_to_output
                else (self.workspace_root / raw[2:])
            )
        elif Path(raw).is_absolute():
            target = Path(raw)
        else:
            target = (
                (self.output_root / raw)
                if default_to_output
                else (self.workspace_root / raw)
            )

        resolved = target.resolve()
        try:
            resolved.relative_to(self.workspace_root)
        except ValueError:
            raise ValueError("Path escapes workspace")
        return resolved

    # ----------------------------------------------------------------------- #
    #                       Sliding Window Management                         #
    # ----------------------------------------------------------------------- #

    def _summarize_history(self, messages: list[dict]) -> str:
        """Produce a condensed summary of prior conversation messages.

        Args:
            messages: List of chat message dicts with ``role`` and ``content``.

        Returns:
            Summarized text suitable for injecting as conversation context.
        """
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

        if self._llm_provider:
            try:
                summary = self._llm_provider.complete(
                    user_prompt=f"Summarize this agent conversation history in 3-5 bullet points. Keep key decisions, deliverables produced, and pending issues:\n\n{conversation_text[:4000]}",
                    system_prompt="You are a concise summarizer. Output only bullet points, no preamble.",
                )
                return summary
            except Exception:
                pass

        return (
            f"[Previous work summary — {len(messages)} exchanges]: "
            + conversation_text[:1000]
        )

    def _manage_sliding_window(self, agent_obj) -> None:
        """Trim the agent conversation history to prevent context overflow.

        Older messages are summarized and replaced with a compact context
        block, retaining only the most recent exchanges.

        Args:
            agent_obj: Strands ``Agent`` instance whose message history
                will be compacted.
        """
        if self._turn_count <= SLIDING_WINDOW_THRESHOLD:
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

        existing_summary = self._agent_summaries.get("deployeur", "")
        summary = self._summarize_history(old_messages)

        full_summary = (
            f"{existing_summary}\n\n--- Later ---\n{summary}"
            if existing_summary
            else summary
        )
        self._agent_summaries["deployeur"] = full_summary

        summary_message = {
            "role": "user",
            "content": f"[CONTEXT SUMMARY — your previous work and conversations]:\n{full_summary}\n\n[End of summary — the following messages are your recent history]",
        }
        new_messages = [summary_message] + recent_messages

        for attr in ("messages", "_messages", "conversation", "_conversation"):
            if hasattr(agent_obj, attr) and isinstance(getattr(agent_obj, attr), list):
                setattr(agent_obj, attr, new_messages)
                break

        self._turn_count = len(recent_messages)
        self._log_event(
            "deployeur",
            "PLAN",
            f"Sliding window applied: {len(old_messages)} old messages summarized",
        )

    # ----------------------------------------------------------------------- #
    #                          Agent Tool Definitions                         #
    # ----------------------------------------------------------------------- #

    @staticmethod
    def _strands_tool(fn):
        """Wrap a function as a Strands tool, with a no-op fallback."""

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        try:
            from strands import tool

            return tool(wrapper)
        except (ImportError, Exception):
            return wrapper

    def _make_tools(self):
        """Build the set of tools exposed to the deployment LLM agent.

        Returns:
            List of Strands-wrapped tool callables (list_files, read_file,
            write_file, run_command).
        """
        agent = self

        def list_files(directory: str = "") -> str:
            """List all files in the project directory. Optionally specify a subdirectory."""
            try:
                base = agent._resolve_path(directory or "", default_to_output=True)
            except ValueError as e:
                return json.dumps({"error": str(e)})
            if not base.exists():
                return json.dumps([])
            files = sorted(
                str(p.relative_to(base)).replace("\\", "/")
                for p in base.rglob("*")
                if p.is_file()
                and ".git" not in p.parts
                and "node_modules" not in p.parts
                and "__pycache__" not in p.parts
            )
            if len(files) > 100:
                return json.dumps(
                    {
                        "message": f"Too many files ({len(files)}). Returning first 100.",
                        "files": files[:100],
                    }
                )
            return json.dumps(files)

        def read_file(filepath: str) -> str:
            """Read a file from the project by its relative path."""
            try:
                out = agent._resolve_path(filepath, default_to_output=True)
            except ValueError as e:
                return f"Invalid path: {e}"
            if not out.exists():
                return f"File not found: {filepath}"
            try:
                return out.read_text(encoding="utf-8")
            except Exception as e:
                return f"Error reading file: {e}"

        def write_file(filepath: str, content: str) -> str:
            """Write content to a file in the project. Parent directories are created automatically."""
            try:
                out = agent._resolve_path(filepath, default_to_output=True)
            except ValueError as e:
                return f"Invalid path: {e}"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(content, encoding="utf-8")
            rel = str(out.relative_to(agent.workspace_root)).replace("\\", "/")
            agent._log_event("deployeur", "ACT", f"Wrote file: {rel}")
            return f"Written: {rel}"

        def run_command(command: str, cwd: str = ".", timeout: int = 120) -> str:
            """Run a shell command. cwd defaults to the project root directory.
            Adapt syntax to the detected OS: PowerShell on Windows, bash on Linux/macOS.
            """
            try:
                if cwd == "." or not cwd:
                    workdir = agent.output_root
                else:
                    workdir = agent._resolve_path(cwd, default_to_output=True)
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

            agent._log_event("deployeur", "ACT", f"Run command: {command}")

            try:
                result = subprocess.run(
                    command,
                    cwd=str(workdir),
                    shell=True,
                    capture_output=True,
                    timeout=timeout,
                )
                stdout = result.stdout.decode("utf-8", errors="replace")
                stderr = result.stderr.decode("utf-8", errors="replace")
                return json.dumps(
                    {
                        "cwd": str(workdir),
                        "exit_code": result.returncode,
                        "stdout": stdout.strip(),
                        "stderr": stderr.strip(),
                    },
                    ensure_ascii=False,
                )
            except subprocess.TimeoutExpired:
                return json.dumps(
                    {
                        "cwd": str(workdir),
                        "exit_code": -1,
                        "stdout": "",
                        "stderr": f"Process killed: exceeded timeout of {timeout}s.",
                    },
                    ensure_ascii=False,
                )
            except Exception as e:
                return json.dumps(
                    {
                        "cwd": str(workdir),
                        "exit_code": -1,
                        "stdout": "",
                        "stderr": str(e),
                    },
                    ensure_ascii=False,
                )

        return [
            self._strands_tool(list_files),
            self._strands_tool(read_file),
            self._strands_tool(write_file),
            self._strands_tool(run_command),
        ]

    # ----------------------------------------------------------------------- #
    #                       Deployment Workspace Setup                        #
    # ----------------------------------------------------------------------- #

    def _copy_to_deploy(self):
        """Copy the output directory into a dedicated deployment workspace.

        Creates ``output_deploy/`` as a clean copy of ``output/``, excluding
        build artifacts and caches, then redirects ``output_root`` to it.
        """
        src_dir = self.workspace_root / "output"
        dst_dir = self.workspace_root / "output_deploy"

        self._log_event(
            "system",
            "ACT",
            f"Copying {src_dir} → {dst_dir} (preserving existing files)",
        )
        shutil.copytree(
            str(src_dir),
            str(dst_dir),
            ignore=shutil.ignore_patterns(
                "__pycache__",
                "*.pyc",
                ".pytest_cache",
                "node_modules",
                "agent_trace.log",
                ".coverage",
                "*.db",
            ),
            dirs_exist_ok=True,
        )
        self.output_root = dst_dir.resolve()
        self._log_event(
            "system", "DONE", f"Project copied — output_root → {self.output_root}"
        )

    # ----------------------------------------------------------------------- #
    #                          Deployment Execution                           #
    # ----------------------------------------------------------------------- #

    async def run(self, max_cycles: int = 30):
        """Execute the full deployment pipeline.

        Validates the workspace, initializes the LLM, copies the project,
        creates a Strands agent, and iterates through deployment cycles
        until completion or the cycle limit is reached.

        Args:
            max_cycles: Maximum number of agent interaction cycles.
        """
        src_dir = self.workspace_root / "output"
        if not src_dir.exists():
            print(
                "[ERROR] output/ directory not found. Run the swarm first to generate code."
            )
            return

        self.init_llm()
        self._copy_to_deploy()

        try:
            from strands import Agent

            strands_model = self._get_strands_model()
            tools = self._make_tools()

            agent_kwargs = {"system_prompt": DEPLOYEUR_PROMPT, "tools": tools}
            if strands_model:
                agent_kwargs["model"] = strands_model

            agent = Agent(**agent_kwargs)
        except ImportError:
            print("[ERROR] strands-agents not installed. pip install strands-agents")
            return

        self._log_event(
            "deployeur", "ACT", "Deployeur agent created — starting deployment"
        )

        cycle = 0
        while cycle < max_cycles:
            cycle += 1
            self._turn_count = cycle

            if cycle == 1:
                task_prompt = (
                    f"PROJECT SPECS: {self.specs}\n\n"
                    f"YOUR TASK: Deploy this project. The code is complete and all tests pass.\n"
                    f"Follow your phases: Environment → Read Project → Git & CI → Deploy → Verify → Report.\n"
                    f"Start with Phase 1 — detect the environment."
                )
            else:
                task_prompt = (
                    f"== DEPLOYMENT CYCLE {cycle} ==\n"
                    f"Continue your deployment workflow from where you left off.\n"
                    f"If you have completed all phases and confirmed HTTP 200, "
                    f"produce your final deployment report as plain text."
                )

            self._log_event("deployeur", "ACT", f"Deployment cycle {cycle}")

            try:
                self._log_to_file("AGENT_START", "deployeur", task_prompt[:300])
                result = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: str(agent(task_prompt))
                )
                self._log_to_file("AGENT_FINISH", "deployeur", result[:500])
                self._log_event("deployeur", "DONE", f"Cycle {cycle} complete")

                result_lower = result.lower()
                if (
                    "deployment complete" in result_lower
                    or "déploiement terminé" in result_lower
                ):
                    self._log_event(
                        "deployeur", "DONE", "Deployment report produced — done"
                    )
                    self.final_report = result
                    break

                self._manage_sliding_window(agent)

            except Exception as e:
                self._log_event("deployeur", "ERROR", f"Cycle {cycle} error: {e}")
                await asyncio.sleep(5)

        self._print_summary(cycle, max_cycles)

    def _print_summary(self, cycle: int, max_cycles: int):
        """Print a human-readable deployment summary to stdout.

        Args:
            cycle: Number of cycles executed.
            max_cycles: Maximum cycle limit configured.
        """
        dst_dir = self.workspace_root / "output_deploy"
        deploy_files = (
            [p for p in dst_dir.rglob("*") if p.is_file()] if dst_dir.exists() else []
        )

        print(f"\n{'=' * 60}")
        if self.final_report:
            print("DEPLOYMENT COMPLETE")
            print("=" * 60)
            print(self.final_report)
        elif cycle >= max_cycles:
            print(f"DEPLOYMENT STOPPED — reached {max_cycles} cycle limit")
            print("=" * 60)
            print("Check output_deploy/agent_trace.log for details.")
        else:
            print("DEPLOYMENT ENDED")
            print("=" * 60)

        print(f"\nFiles in output_deploy/: {len(deploy_files)}")
        print(f"Trace log: output_deploy/agent_trace.log")


# --------------------------------------------------------------------------- #
#                              CLI Entry Point                                #
# --------------------------------------------------------------------------- #


def main():
    """Parse command-line arguments and launch the deployment agent."""
    parser = argparse.ArgumentParser(
        description="Isolated Deployment Agent — test Phase 2"
    )
    parser.add_argument(
        "--specs", type=str, required=True, help="Project specs (what was built)"
    )
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"  DEPLOYMENT AGENT — ISOLATED TEST")
    print(f"  Specs: {args.specs[:80]}")
    print(f"{'=' * 60}\n")

    agent = DeploymentAgent(specs=args.specs)
    asyncio.run(agent.run())


if __name__ == "__main__":
    main()
