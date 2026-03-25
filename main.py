"""Multi-agent software delivery orchestrator — slim entry point.

Coordinates a team of six persistent AI agents (analyste, architecte,
developpeur, devops, qa, reviewer) running in parallel via asyncio.

Usage:
    python main.py --specs "Project description"
    python main.py --web-ui
    python main.py --mcp --mcp-transport sse
"""

import argparse
import asyncio
import os
import sys
import threading
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Re-exports for backward-compatibility (mcp_server.py imports from `main`)
# ---------------------------------------------------------------------------
from state import (
    _blackboard,  # noqa: F401
    _bb_lock,  # noqa: F401
    _agent_states,  # noqa: F401
    _event_log,  # noqa: F401
    _event_lock,  # noqa: F401
    _memory,  # noqa: F401 — reassigned below in run_swarm
    AgentState,  # noqa: F401
)
import state as _state_mod

from agent_config import AGENT_CONFIGS  # noqa: F401

load_dotenv()

# Module-level flags — mcp_server.py sets these directly on the module object
_use_rich_display: bool = True
_use_web_ui: bool = False

# ---------------------------------------------------------------------------
# LLM Provider
# ---------------------------------------------------------------------------
try:
    from llm_provider import LLMRouter, BaseLLMProvider

    _LLM_AVAILABLE = True
except ImportError:
    _LLM_AVAILABLE = False
    LLMRouter = None

_llm_provider: Optional[object] = None
_frontend_blocks: list[dict] | None = None

# ---------------------------------------------------------------------------
# Browser Tool
# ---------------------------------------------------------------------------
_browser_instance = None


def _get_browser_tool():
    """Return a shared headless browser tool, creating it on first call."""
    global _browser_instance
    if _browser_instance is None:
        os.environ.setdefault("STRANDS_BROWSER_HEADLESS", "true")
        from strands_tools.browser import LocalChromiumBrowser

        _browser_instance = LocalChromiumBrowser(launch_options={"channel": "chrome"})
    return _browser_instance.browser


# ---------------------------------------------------------------------------
# LLM Strands Model Helper
# ---------------------------------------------------------------------------


def get_strands_model():
    """Build and return a Strands-compatible model from the active LLM provider."""
    if _llm_provider is None:
        return None
    return _llm_provider.to_strands_model()


# ---------------------------------------------------------------------------
# Frontend File Loader
# ---------------------------------------------------------------------------


def load_frontend_file(path: str) -> list[dict]:
    """Load an image or PDF file and return Strands ContentBlock dicts.

    Args:
        path: Filesystem path to the reference file.

    Returns:
        List of content-block dictionaries compatible with Strands message format.

    Raises:
        FileNotFoundError: If the specified path does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Frontend file not found: {path}")

    ext = p.suffix.lower()

    if ext == ".pdf":
        return [
            {
                "document": {
                    "format": "pdf",
                    "name": p.stem,
                    "source": {"bytes": p.read_bytes()},
                }
            }
        ]

    fmt = {
        ".png": "png",
        ".jpg": "jpeg",
        ".jpeg": "jpeg",
        ".gif": "gif",
        ".webp": "webp",
    }.get(ext, "png")

    return [
        {
            "image": {
                "format": fmt,
                "source": {"bytes": p.read_bytes()},
            }
        }
    ]


# ---------------------------------------------------------------------------
# OpenTelemetry Tracing
# ---------------------------------------------------------------------------
os.environ.setdefault(
    "OTEL_SEMCONV_STABILITY_OPT_IN",
    "gen_ai_latest_experimental,gen_ai_tool_definitions",
)

_tracing_enabled = False  # propagated to state module below


def _tracing_endpoint_configured() -> bool:
    """Return True when OTLP tracing has been explicitly configured."""
    return bool(os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip())


def _setup_tracing():
    """Configure OpenTelemetry OTLP exporters and meters for Jaeger integration."""
    global _tracing_enabled
    try:
        from strands.telemetry import StrandsTelemetry

        telemetry = StrandsTelemetry()
        telemetry.setup_otlp_exporter()
        telemetry.setup_console_exporter()
        telemetry.setup_meter(
            enable_otlp_exporter=True,
            enable_console_exporter=False,
        )
        _tracing_enabled = True
        # Also patch state module so agents.py can read it
        _state_mod._tracing_enabled = True
        print("  Tracing: OpenTelemetry -> Jaeger (http://localhost:16686)")
    except Exception as e:
        print(f"  Tracing: DISABLED ({e})")
        _tracing_enabled = False


# ---------------------------------------------------------------------------
# Swarm Orchestrator
# ---------------------------------------------------------------------------


async def run_swarm(specs: str, use_rich: bool = True):
    """Launch the full multi-agent swarm and block until completion.

    Args:
        specs: Project specification string driving all agent work.
        use_rich: Enable the Rich terminal dashboard.
    """
    import asyncio as _asyncio
    import threading as _threading

    from memory import HybridMemory
    import sys as _sys

    _this_module = _sys.modules[__name__]
    _use_web_ui = _this_module._use_web_ui
    _WEB_UI_AVAILABLE = _state_mod._WEB_UI_AVAILABLE

    from state import (
        _agent_health,
        _agent_sleep_durations,
        _agent_summaries,
        _agent_turn_count,
        _push_status,
        _log_event,
        _reviewer_said_pass,
        _all_agents_done,
        _bb_lock,
        _blackboard,
        _health_lock,
    )
    from agent_config import AGENT_CONFIGS
    from agents import run_agent
    from spawner import spawner_loop
    from dashboard import SwarmDashboard
    from deployement import DeploymentAgent

    # Make state-module globals accessible to mcp_server via `main._memory` etc.
    global _memory  # noqa: PLW0603

    # Reset state module globals for this run
    _state_mod._stuck_queue = _asyncio.Queue()
    _state_mod._swarm_done = _asyncio.Event()
    _state_mod._memory = HybridMemory(persist_dir="./memory_store")
    _memory = _state_mod._memory  # keep re-export in sync

    _state_mod._memory.store(
        f"Project specs: {specs}", {"agent": "system", "type": "specs"}
    )

    Path("./output/src").mkdir(parents=True, exist_ok=True)
    Path("./output/tests").mkdir(parents=True, exist_ok=True)
    Path("./output/adr").mkdir(parents=True, exist_ok=True)
    Path("./output/diagrams").mkdir(parents=True, exist_ok=True)
    Path("./output/devops").mkdir(parents=True, exist_ok=True)

    loop = _asyncio.get_event_loop()

    if _use_web_ui and _WEB_UI_AVAILABLE:
        import web_dashboard

        async def _web_ui_sync_loop():
            while not _state_mod._swarm_done.is_set():
                with _bb_lock:
                    bb_copy = dict(_blackboard)
                with _state_mod._inbox_lock:
                    inbx_copy = {
                        k: list(v) for k, v in _state_mod._agent_inboxes.items()
                    }
                web_dashboard.push_state(bb_copy, inbx_copy)
                await _asyncio.sleep(1.0)

        loop.create_task(_web_ui_sync_loop())

    if _frontend_blocks:
        AGENT_CONFIGS["analyste"]["system_prompt"] += (
            "\n\n== FRONTEND REFERENCE ==\n"
            "A frontend reference file (image or PDF) was provided at startup. You will "
            "receive it in your first message. Use it as a primary input when deriving the "
            "feature list, layout assumptions, and ambiguity notes. Do not claim the image "
            "is unavailable if it is attached."
        )
        AGENT_CONFIGS["architecte"]["system_prompt"] += (
            "\n\n== FRONTEND REFERENCE ==\n"
            "A frontend reference file (image or PDF) was provided at startup. You will "
            "receive it in your first message. Use it as a primary input when deriving the "
            "architecture, component boundaries, and diagram assumptions. Do not claim the "
            "image is unavailable if it is attached."
        )
        AGENT_CONFIGS["reviewer"]["system_prompt"] += (
            "\n\n== FRONTEND VALIDATION ==\n"
            "A frontend reference file (image or PDF) was provided at startup (you will "
            "receive it in your first message).\n"
            "Your quality gate MUST include a visual compliance check:\n"
            "- Does the generated frontend match the layout and structure of the reference?\n"
            "- Are the colors, typography, and component placement consistent?\n"
            "- If a chart: is the chart type, scale, and styling correct?\n"
            "- If a multi-page PDF: verify ALL pages are addressed.\n"
            "Add a 'frontend_match' boolean and 'frontend_notes' string to your "
            "reviewer_verdict JSON. A PASS requires frontend_match: true."
        )
        AGENT_CONFIGS["developpeur"]["system_prompt"] += (
            "\n\n== FRONTEND REFERENCE ==\n"
            "A UI/chart reference file was provided. You will receive it in your "
            "first message. Study it carefully and reproduce it faithfully in the frontend code."
        )

    for name in AGENT_CONFIGS:
        _push_status(name, AgentState.WAITING)
        _agent_sleep_durations[name] = 0
        _agent_turn_count[name] = 0
        _agent_summaries.pop(name, None)

    for name in list(AGENT_CONFIGS.keys()) + ["spawner"]:
        _agent_health[name] = {
            "status": "IDLE",
            "phase": "IDLE",
            "last_msg": "Waiting to start",
            "steps": 0,
            "total": 6,
            "progress": 0.0,
        }

    dashboard = None
    if use_rich:
        dashboard = SwarmDashboard()
        dash_thread = _threading.Thread(target=dashboard.run, daemon=True)
        dash_thread.start()

    spawner_task = _asyncio.create_task(spawner_loop(loop))

    try:
        _log_event(
            "spawner",
            "PLAN",
            "All persistent agents launched -- stops on reviewer PASS + all agents DONE",
        )

        agent_names = list(AGENT_CONFIGS.keys())
        results = await _asyncio.gather(
            *[run_agent(name, specs, loop) for name in agent_names],
            return_exceptions=True,
        )

        for name, result in zip(agent_names, results):
            if isinstance(result, Exception):
                _push_status(name, AgentState.BLOCKED)
                _log_event(name, "ERROR", f"Agent task crashed: {result!r}")

    except _asyncio.CancelledError:
        pass

    finally:
        _state_mod._swarm_done.set()
        spawner_task.cancel()
        try:
            await spawner_task
        except _asyncio.CancelledError:
            pass

        if _use_web_ui and _WEB_UI_AVAILABLE:
            import web_dashboard

            web_dashboard.set_pipeline_status("done")

        if dashboard is not None:
            dashboard.stop()
        await _asyncio.sleep(0.5)
        if dashboard is not None:
            dashboard.show_summary()

    if _reviewer_said_pass() and _all_agents_done():
        deploy_agent = DeploymentAgent(
            specs=specs, workspace_root=str(_state_mod._workspace_root)
        )
        if _llm_provider is not None:
            deploy_agent._llm_provider = _llm_provider
        await deploy_agent.run()


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def main():
    """CLI entry point: parse arguments, configure providers, and run the swarm."""
    global _llm_provider, _frontend_blocks

    parser = argparse.ArgumentParser(
        description="hackathon-ibm -- Multi-Agent Software Delivery Swarm"
    )
    parser.add_argument(
        "--specs",
        type=str,
        default=None,
        help="Project specifications (optional if --web-ui is used)",
    )
    parser.add_argument(
        "--demo", action="store_true", help="Demo mode -- simulate agents without LLM"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help=(
            "LLM provider: snowflake, ollama, anthropic, openrouter, openai, "
            "gemini, bedrock, mistral, groq, cohere, litellm"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (e.g. openai-gpt-4.1 for Snowflake, deepseek-r1:14b for Ollama)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models per provider and exit",
    )
    parser.add_argument(
        "--no-rich",
        action="store_true",
        help="Disable Rich dashboard and print plain logs to terminal",
    )
    parser.add_argument(
        "--frontend-img",
        type=str,
        default=None,
        help=(
            "Path to a frontend reference file (image or PDF). "
            "The reviewer and developer will match it."
        ),
    )
    parser.add_argument(
        "--no-tracing",
        action="store_true",
        help="Disable OpenTelemetry tracing (Jaeger export)",
    )
    parser.add_argument(
        "--web-ui",
        action="store_true",
        help="Enable modern web UI at http://127.0.0.1:8500",
    )
    parser.add_argument(
        "--mcp",
        action="store_true",
        help="Expose the swarm as an MCP server instead of running interactively",
    )
    parser.add_argument(
        "--mcp-transport",
        type=str,
        default="stdio",
        choices=["stdio", "sse", "streamable-http"],
        help="MCP transport (default: stdio)",
    )
    parser.add_argument(
        "--mcp-host",
        type=str,
        default="127.0.0.1",
        help="MCP server host for sse/streamable-http (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--mcp-port",
        type=int,
        default=8000,
        help="MCP server port for sse/streamable-http (default: 8000)",
    )
    args = parser.parse_args()

    # In stdio MCP mode stdout must carry only JSON-RPC — redirect all prints to stderr
    if args.mcp and args.mcp_transport == "stdio":
        sys.stdout = sys.stderr

    global _use_rich_display, _use_web_ui
    _use_web_ui = args.web_ui
    _use_rich_display = not args.no_rich and not args.web_ui
    _state_mod._use_web_ui = _use_web_ui
    _state_mod._use_rich_display = _use_rich_display
    _state_mod._frontend_blocks = _frontend_blocks

    if _state_mod._use_web_ui and _state_mod._WEB_UI_AVAILABLE:
        import web_dashboard

        web_dashboard.start_dashboard_server()

    if not args.no_tracing and not args.demo and _tracing_endpoint_configured():
        os.environ["SWARM_SESSION_ID"] = uuid.uuid4().hex[:12]
        _setup_tracing()
    elif not args.no_tracing and not args.demo:
        print("  Tracing: DISABLED (no OTLP endpoint configured)")
    else:
        print("  Tracing: DISABLED (--no-tracing or --demo)")

    if args.model:
        os.environ["SWARM_MODEL"] = args.model

    if args.mcp:
        from mcp_server import start_mcp_server

        start_mcp_server(
            transport=args.mcp_transport,
            host=args.mcp_host,
            port=args.mcp_port,
            provider=args.provider,
            model=args.model,
        )
        return

    if args.demo:
        sys.modules["strands"] = None  # type: ignore

    if hasattr(args, "list_models") and args.list_models:
        if _LLM_AVAILABLE:
            models = LLMRouter.list_all()
            for prov, mlist in models.items():
                print(f"\n{prov}:")
                for m in mlist:
                    print(f"  - {m}")
        else:
            print("llm_provider.py not found")
        return

    if not args.demo and _LLM_AVAILABLE:
        if args.provider:
            _llm_provider = LLMRouter.create(args.provider, model=args.model)
        else:
            _llm_provider = LLMRouter.from_env()

        try:
            _llm_provider.to_strands_model()
        except Exception as e:
            print(f"LLM configuration error: {e}")
            return

        print(f"  LLM: {type(_llm_provider).__name__} / {_llm_provider.model}")
        _state_mod._llm_provider = _llm_provider

    if args.frontend_img:
        try:
            _frontend_blocks = load_frontend_file(args.frontend_img)
            kind = (
                "PDF document"
                if args.frontend_img.lower().endswith(".pdf")
                else "image"
            )
            print(
                f"  Frontend ref loaded: {args.frontend_img} "
                f"({kind}, {len(_frontend_blocks)} block(s))"
            )
            _state_mod._frontend_blocks = _frontend_blocks
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return

    specs = args.specs
    waiting_for_ui = False

    if specs is None and args.web_ui and _state_mod._WEB_UI_AVAILABLE:
        waiting_for_ui = True
    elif specs is None:
        specs = "Application de gestion de taches (To-Do) avec auth, dashboard filtrable, CRUD taches."

    print(f"\n{'='*60}")
    print("  hackathon-ibm -- Multi-Agent Swarm")
    print(f"  Model: {args.model if args.model else 'default per provider'}")
    print(f"  Specs: {specs if specs else '(waiting for Web UI input)'}")
    print(f"  Mode: {'DEMO' if args.demo else 'LIVE'}")
    print(
        f"  Display: "
        f"{'WEB UI' if args.web_ui else ('PLAIN LOGS' if args.no_rich else 'RICH DASHBOARD')}"
    )
    print(f"  Frontend ref: {args.frontend_img or 'none'}")
    print(
        f"  Tracing: "
        f"{'Jaeger @ http://localhost:16686' if _tracing_enabled else 'DISABLED'}"
    )
    print("  Memory: Hybrid BM25 + Semantic (ChromaDB)")
    print(f"{'='*60}\n")

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    if waiting_for_ui:
        import web_dashboard

        _ui_specs_event = threading.Event()
        _ui_specs: list = [None, None]  # [specs, frontend_img_path]

        def _on_start(user_specs: str, img_path: str | None):
            _ui_specs[0] = user_specs
            _ui_specs[1] = img_path
            _ui_specs_event.set()

        web_dashboard.register_start_callback(_on_start)
        print("  Web UI ready at http://127.0.0.1:8500")
        print("  Waiting for project specs from Web UI...")
        try:
            _ui_specs_event.wait()
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            return

        specs = _ui_specs[0]
        if _ui_specs[1]:
            try:
                _frontend_blocks = load_frontend_file(_ui_specs[1])
                _state_mod._frontend_blocks = _frontend_blocks
                print(
                    f"  Frontend ref loaded from upload: {_ui_specs[1]} "
                    f"({len(_frontend_blocks)} block(s))"
                )
            except Exception as e:
                print(f"  Warning: could not load uploaded file: {e}")

        print(f"  Specs received: {specs[:120]}...")

    try:
        asyncio.run(run_swarm(specs, use_rich=not args.no_rich))
    except KeyboardInterrupt:
        print("\nInterrupted by user.")


if __name__ == "__main__":
    main()
