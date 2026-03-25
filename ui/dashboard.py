"""Rich terminal dashboard module.

Defines agent display constants (AGENT_COLORS, AGENT_ICONS, PHASE_STYLES),
panel builder functions (build_agent_panel, build_event_bus_panel,
build_artifacts_panel, build_memory_panel, build_steering_panel),
and the SwarmDashboard class that drives the Live terminal UI with
optional user-steering input.
"""

import sys
import time
import threading
from pathlib import Path

from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from core.state import (
    _agent_health,
    _event_log,
    _event_lock,
    _health_lock,
    _memory,
)
from agents.agent_config import AGENT_CONFIGS
from agents.steering import _steering_log, _steering_log_lock, send_steering

# ---------------------------------------------------------------------------
# Display Constants
# ---------------------------------------------------------------------------

AGENT_COLORS = {
    "analyste": "#9b7ecf",
    "architecte": "#5ea8aa",
    "devops": "#4caf7a",
    "developpeur": "#c9a54e",
    "qa": "#c46b6b",
    "reviewer": "#6b9fd4",
    "spawner": "#7986cb",
    "user": "#bfad6e",
    "steering": "#bfad6e",
}

AGENT_ICONS = {
    "analyste": "🔍",
    "architecte": "🏗",
    "devops": "⚙",
    "developpeur": "💻",
    "qa": "✅",
    "reviewer": "🧪",
    "spawner": "⚡",
}

PHASE_STYLES = {
    "PLAN": "#6b9fd4",
    "ACT": "#c9a54e",
    "REASON": "#5ea8aa",
    "DONE": "#4caf7a",
    "IDLE": "dim",
    "ERROR": "#c46b6b",
    "STEER": "#bfad6e",
}

# ---------------------------------------------------------------------------
# Panel Builders
# ---------------------------------------------------------------------------


def build_agent_panel(agent_name: str) -> Panel:
    """Render a Rich Panel showing an agent's status, recent events, and progress."""
    color = AGENT_COLORS.get(agent_name, "white")
    icon = AGENT_ICONS.get(agent_name, "•")

    with _health_lock:
        health = _agent_health.get(agent_name, {})
    with _event_lock:
        agent_events = [e for e in _event_log if e["agent"] == agent_name][-8:]

    status = health.get("status", "IDLE")
    phase = health.get("phase", "IDLE")
    progress = health.get("progress", 0.0)

    phase_style = PHASE_STYLES.get(phase, "white")
    badge = f"[{phase_style}][{phase}][/{phase_style}]"

    table = Table.grid(padding=(0, 1))
    table.add_column(style="dim", width=8)
    table.add_column(width=8)
    table.add_column()

    for ev in agent_events:
        ps = PHASE_STYLES.get(ev["phase"], "white")
        table.add_row(
            Text(ev["ts"], style="dim"),
            Text(f"[{ev['phase']}]", style=ps),
            Text(ev["msg"], style="white"),
        )

    filled = int(progress * 20)
    bar = "━" * filled + "╌" * (20 - filled)
    pct = f"{int(progress * 100)}%"
    prog_text = f"[{color}]{bar}[/{color}] [dim]{pct}[/dim]"

    content = Table.grid()
    content.add_row(table)
    content.add_row(Text.from_markup(f"\n{prog_text}"))

    border_style = color if status != "IDLE" else "dim"
    title = f"{icon} {agent_name.upper()} {badge}"

    return Panel(content, title=title, border_style=border_style, box=box.SQUARE)


def build_event_bus_panel() -> Panel:
    """Render the live event bus panel showing the most recent system events."""
    with _event_lock:
        recent = list(reversed(_event_log[-12:]))

    table = Table.grid(padding=(0, 1))
    table.add_column(style="dim", width=8)
    table.add_column(width=12)
    table.add_column(width=9)
    table.add_column()

    for ev in recent:
        ps = PHASE_STYLES.get(ev["phase"], "white")
        color = AGENT_COLORS.get(ev["agent"], "white")
        table.add_row(
            Text(ev["ts"], style="dim"),
            Text(ev["agent"], style=f"bold {color}"),
            Text(f"[{ev['phase']}]", style=ps),
            Text(ev["msg"]),
        )

    return Panel(table, title="Event Bus", border_style="#5a6480", box=box.SQUARE)


def build_artifacts_panel() -> Panel:
    """Render a panel listing generated artifact files with sizes."""
    output_dir = Path("./output")
    files = []
    if output_dir.exists():
        for p in sorted(output_dir.rglob("*")):
            if p.is_file():
                size = p.stat().st_size
                rel = str(p.relative_to(output_dir))
                files.append(f"[green]{rel}[/green] [dim]({size}b)[/dim]")

    content = "\n".join(files[:15]) or "[dim]No files yet[/dim]"
    return Panel(
        Text.from_markup(content),
        title="Artifacts",
        border_style="#4caf7a",
        box=box.SQUARE,
    )


def build_memory_panel() -> Panel:
    """Render a panel showing the latest entries from long-term memory."""
    from core.state import _memory as mem

    if not mem:
        return Panel(
            "[dim]Memory not initialized[/dim]", title="Memory", box=box.SQUARE
        )
    entries = mem.get_all(limit=5)
    lines = []
    for e in reversed(entries):
        agent = e["metadata"].get("agent", "?")
        color = AGENT_COLORS.get(agent.replace("specialist:", ""), "white")
        lines.append(f"[{color}]{agent}[/{color}]: [dim]{e['text']}[/dim]")
    content = "\n".join(lines) or "[dim]No memories yet[/dim]"
    return Panel(
        Text.from_markup(content),
        title="Long-term Memory",
        border_style="#9b7ecf",
        box=box.SQUARE,
    )


def build_steering_panel() -> Panel:
    """Render a panel displaying recent steering messages and the input prompt."""
    with _steering_log_lock:
        recent = list(reversed(_steering_log[-6:]))
    lines = []
    for entry in recent:
        tgt = entry["target"]
        tgt_label = (
            f"[#5ea8aa]{tgt.upper()}[/#5ea8aa]"
            if tgt != "all"
            else "[white]ALL[/white]"
        )
        lines.append(f"[dim]{entry['ts']}[/dim] {tgt_label}: {entry['message'][:80]}")
    if not lines:
        lines.append("[dim]No steering messages yet[/dim]")
    lines.append("")
    lines.append("[#bfad6e]Press S to steer agents[/#bfad6e]")
    content = "\n".join(lines)
    return Panel(
        Text.from_markup(content),
        title="Steering",
        border_style="#bfad6e",
        box=box.SQUARE,
    )


# ---------------------------------------------------------------------------
# Dashboard Class
# ---------------------------------------------------------------------------


class SwarmDashboard:
    """Rich Live terminal dashboard for real-time swarm monitoring."""

    def __init__(self):
        self._console = Console()
        self._start_time = time.time()
        self._running = True

    def _build_layout(self) -> Layout:
        """Compose the full dashboard layout with header, agent panels, and footer."""
        from core.state import _memory as mem

        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="agents", size=22),
            Layout(name="bottom", size=16),
        )

        elapsed = int(time.time() - self._start_time)
        m, s = divmod(elapsed, 60)
        with _health_lock:
            active = sum(
                1 for h in _agent_health.values() if h.get("status") == "WORKING"
            )
            done = sum(1 for h in _agent_health.values() if h.get("status") == "DONE")
        header_text = (
            f"[bold white]Multi-Agent Swarm[/bold white]  "
            f"[dim]|[/dim]  [#5ea8aa]{m:02d}:{s:02d}[/#5ea8aa]  "
            f"[dim]|[/dim]  [#4caf7a]Active {active}[/#4caf7a]  "
            f"[dim]|[/dim]  [#6b9fd4]Done {done}/{len(AGENT_CONFIGS)}[/#6b9fd4]  "
            f"[dim]|  Memory: {len(mem._docs) if mem else 0}[/dim]"
        )
        layout["header"].update(
            Panel(
                Text.from_markup(header_text),
                box=box.HORIZONTALS,
            )
        )

        layout["agents"].split_row(
            Layout(name="row1"),
            Layout(name="row2"),
        )
        layout["row1"].split_column(
            Layout(name="ag_top"),
            Layout(name="ag_bot"),
        )
        layout["ag_top"].split_row(
            Layout(build_agent_panel("analyste")),
            Layout(build_agent_panel("architecte")),
            Layout(build_agent_panel("devops")),
        )
        layout["ag_bot"].split_row(
            Layout(build_agent_panel("developpeur")),
            Layout(build_agent_panel("qa")),
            Layout(build_agent_panel("reviewer")),
            Layout(build_agent_panel("spawner")),
        )
        layout["row2"].update(build_event_bus_panel())

        layout["bottom"].split_row(
            Layout(build_artifacts_panel()),
            Layout(build_memory_panel()),
            Layout(build_steering_panel()),
        )

        return layout

    def _check_keyboard(self) -> bool:
        """Poll for a steering key-press (platform-aware)."""
        try:
            import msvcrt

            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch.lower() == "s":
                    return True
        except ImportError:
            import select

            if select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)
                if ch.lower() == "s":
                    return True
        return False

    def _prompt_steering(self, live):
        """Pause the live display, collect steering input, and resume."""
        live.stop()
        self._console.print("\n[bold yellow]== User Steering ==[/bold yellow]")
        valid_names = ", ".join(AGENT_CONFIGS.keys())
        self._console.print(f"Target agent ({valid_names}) or 'all': ", end="")
        try:
            target = input().strip() or "all"
            if target != "all" and target not in AGENT_CONFIGS:
                self._console.print(
                    f"[red]Unknown agent '{target}'. "
                    f"Valid: {valid_names} or 'all'[/red]"
                )
                input("Press Enter to continue...")
                live.start()
                return
            self._console.print("Message: ", end="")
            message = input().strip()
            if message:
                send_steering(message, target)
                self._console.print(f"[green]Steering sent to {target}[/green]")
            else:
                self._console.print("[dim]Empty message -- cancelled[/dim]")
            time.sleep(0.5)
        except (EOFError, KeyboardInterrupt):
            pass
        live.start()

    def run(self):
        """Start the Rich Live loop, refreshing panels until stopped."""
        with Live(
            self._build_layout(),
            console=self._console,
            refresh_per_second=3,
            screen=True,
        ) as live:
            while self._running:
                time.sleep(0.35)
                if self._check_keyboard():
                    self._prompt_steering(live)
                live.update(self._build_layout())

    def stop(self):
        """Signal the dashboard loop to terminate."""
        self._running = False

    def show_summary(self):
        """Print a final summary of generated artifacts and memory stats."""
        from core.state import _memory as mem

        self._console.print("\n")
        self._console.rule("[bold green]Swarm Complete[/bold green]")
        output_dir = Path("./output")
        if output_dir.exists():
            files = list(output_dir.rglob("*"))
            file_list = [p for p in files if p.is_file()]
            self._console.print(
                f"\n[green]Generated {len(file_list)} files in ./output/[/green]\n"
            )
            for f in sorted(file_list)[:20]:
                self._console.print(f"  [dim]->[/dim] {f.relative_to(output_dir)}")
        if mem:
            self._console.print(
                f"\n[magenta]Long-term memory: {len(mem._docs)} entries stored[/magenta]"
            )
            self._console.print("[dim]Hybrid BM25 + semantic search available[/dim]")
