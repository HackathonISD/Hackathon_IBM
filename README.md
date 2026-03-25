# hackathon-ibm

A production-grade multi-agent system that automates software delivery.
5 specialized agents work **in true parallel**, with a **Spawner** that creates
specialists on demand when any agent gets stuck.

# Demo video

Please refer to this link https://drive.google.com/file/d/1s51B8Cw1yv4BqOMNjJawXXVSyxQMqxzN/view?usp=sharing

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  TEAM (                parallel                                                )                     │
│                                                         │
│  [Analyste] [Architecte] [DevOps] [Développeur] [QA]   │
│      │           │          │           │         │     │
│      └───────────┴──────────┴───────────┴─────────┘     │
│                       │ report_stuck()                  │
│                       ▼                                 │
│                       │                                 │
│                       ▼                                 │
│          [Spawner] ← always listening                   │
│         spawn_specialist() → run → return result        │
└─────────────────────────────────────────────────────────┘
```

### 4 Communication Channels
| Channel | How | When |
|---------|-----|------|
| Return value | Future resolved by Spawner | Specialist → stuck agent |
| Blackboard | Shared dict + lock | Broadcast to all agents |
| Inbox | Per-agent message queue | Targeted agent-to-agent |
| Filesystem | `output/` directory | Code artifacts |

### Hybrid Long-term Memory
```
Query → [BM25 keyword (0.3)] ──┐
      → [Semantic embed (0.7)] ─┤→ RRF fusion → Top-K results
```
- **BM25** via `bm25s` — fast, no GPU needed, great for exact matches
- **Semantic** via ChromaDB + `all-MiniLM-L6-v2` — meaning-based retrieval
- **RRF** (Reciprocal Rank Fusion, k=60) — combines both with no tuning

---

## Install

```bash
git clone <your-repo-url> hackathon-ibm
cd hackathon-ibm

pip install -r requirements.txt

copy env.example .env
# Edit .env with your model choice
```

On macOS/Linux, replace `copy` with:

```bash
cp env.example .env
```

For local LLM (recommended):
```bash
ollama pull deepseek-r1:14b
# or
ollama pull qwen2.5-coder:32b
```

---

## Run

```bash
# Demo mode (no LLM required — simulates all agents)
python main.py --demo

# Live mode with default specs
python main.py

# Live mode with custom specs
python main.py --specs "Build a REST API for inventory management in Go with Gin framework"

# Override model
python main.py --model ollama/qwen2.5-coder:32b --specs "..."

# OpenRouter
set OPENROUTER_API_KEY=sk-or-...
python main.py --provider openrouter --model anthropic/claude-3.5-sonnet --specs "..."

# Web UI dashboard
python main.py --web-ui --specs "Build a REST API for inventory management in Go with Gin framework"
```

You can also select OpenRouter entirely from `.env`:

```bash
OPENROUTER_API_KEY=sk-or-...
SWARM_MODEL=openrouter/anthropic/claude-3.5-sonnet
```

Under the hood, the project routes OpenRouter through LiteLLM, so any LiteLLM-compatible OpenRouter model path should work.

If you want the browser dashboard, start with `--web-ui` and open:

```text
http://127.0.0.1:8500
```

The dashboard shows prompts, responses, tool calls, agent communication, blackboard writes, and lifecycle events.

## Monitoring via OpenTelemetry

Live runs support OpenTelemetry tracing and metrics export.

- Tracing is enabled by default in live mode
- Tracing is disabled in demo mode
- You can disable it explicitly with `--no-tracing`
- The default OTLP endpoint is `http://localhost:4318`
- The traces are intended to be viewed in Jaeger at `http://localhost:16686`

```bash
# Run with OpenTelemetry enabled (default in live mode)
python main.py --specs "Build a REST API for inventory management in Go with Gin framework"

# Run with web UI + OpenTelemetry
python main.py --web-ui --specs "Build a REST API for inventory management in Go with Gin framework"

# Disable tracing explicitly
python main.py --no-tracing --specs "Build a REST API for inventory management in Go with Gin framework"
```

If you need a different OTLP collector, set the endpoint before launching:

```bash
set OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
python main.py --specs "..."
```

## MCP Server

The solution can also be served as an MCP server instead of running as an interactive CLI application.

```bash
# MCP over stdio
python main.py --mcp

# MCP over SSE
python main.py --mcp --mcp-transport sse --mcp-host 127.0.0.1 --mcp-port 8000

# MCP over streamable HTTP
python main.py --mcp --mcp-transport streamable-http --mcp-host 127.0.0.1 --mcp-port 8000
```

When running in MCP mode, the server exposes tools to:

- run the swarm from a project specification
- inspect blackboard state and agent status
- list and read generated output files
- query the hybrid memory store
- retrieve recent agent event logs

## Repository notes

- Generated deliverables, local memory, uploaded files, virtual environments, databases, and logs are ignored via `.gitignore`.
- Keep `env.example` committed, but do not commit `.env`.
- The `output/` directory is runtime-generated and should be treated as disposable.

---

## Output

All generated files land in `./output/`:
```
output/
├── feature_list.json       ← Analyste
├── adr/
│   └── ADR-001.md
├── diagrams/               ← Architecte
│   ├── c4_context.mmd
│   ├── c4_containers.mmd
│   └── c4_components.mmd
├── shared_dependencies.json
├── devops/                 ← DevOps
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── .github/workflows/ci.yml
├── src/                    ← Développeur
│   └── main.py (+ all modules)
├── tests/                  ← Développeur + QA
│   └── test_*.py
└── qa_report.md            ← QA
```

Long-term memory is persisted in `./memory_store/` (survives restarts).

---

## How the Spawner works

When a team agent calls `report_stuck()`:

1. Agent describes: `task_description`, `what_i_tried`, `what_i_need`, `partial_work`
2. Request goes on the `StuckQueue`
3. Agent's coroutine **blocks** (waits on a Future)
4. Spawner Agent (LLM) receives the request and decides what specialist to create
5. Spawner calls `spawn_specialist_tool(name, system_prompt, task, tools, context)`
6. Specialist Agent runs, writes its output to blackboard + files
7. Spawner resolves the Future with the specialist's result
8. Stuck agent **unblocks** and continues with the specialist's output

All other team agents keep running in parallel — only the stuck agent is blocked.

---

## Memory search example

```python
from main import HybridMemory

mem = HybridMemory("./memory_store")

# Store
mem.store("FastAPI auth service implemented with JWT", {"agent": "developpeur"})

# Hybrid search (BM25 + semantic)
results = mem.search("authentication token")
# Finds "FastAPI auth service implemented with JWT" via both keyword + meaning
```
