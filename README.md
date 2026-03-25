# NexusClaw

`NexusClaw` is a multi-agent software delivery orchestrator built during a 3-day hackathon. It runs a persistent team of AI agents in parallel to analyze requirements, design an architecture, generate code, prepare delivery assets, execute QA, and publish a reviewer verdict.

The system is designed for fast end-to-end software generation with collaborative agents, shared state, long-term memory, optional user steering, multimodal frontend reference support, and MCP exposure for external clients.

## Demo video

Please refer to this link: https://drive.google.com/file/d/1s51B8Cw1yv4BqOMNjJawXXVSyxQMqxzN/view?usp=sharing

## Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Core agents](#core-agents)
- [Communication channels](#communication-channels)
- [Hybrid memory](#hybrid-memory)
- [Agent tools](#agent-tools)
- [Spawner and specialist agents](#spawner-and-specialist-agents)
- [Skills](#skills)
- [User steering](#user-steering)
- [Image and PDF support](#image-and-pdf-support)
- [MCP server](#mcp-server)
- [Installation](#installation)
- [Configuration](#configuration)
- [Run modes](#run-modes)
- [Monitoring and observability](#monitoring-and-observability)
- [Generated outputs](#generated-outputs)
- [Known problems and future fixes](#known-problems-and-future-fixes)

## Overview

The runtime is organized around six persistent agents:

- `analyste`: parses the project request, derives scope, modules, journeys, and ADRs
- `architecte`: designs the system and shared contracts
- `developpeur`: writes source code and tests
- `devops`: prepares execution, packaging, CI/CD, and environment assets
- `qa`: executes tests, checks coverage, and reports failures
- `reviewer`: acts as the final quality gate and publishes the verdict

In addition, the `spawner` can create temporary specialist agents when a core agent is blocked or when a project requires extra expertise such as data science, data engineering, ML engineering, or security.

## Architecture
![Architecture](./documentation/Archi.drawio.svg)


### Architectural responsibilities

- `main.py` is the entry point and orchestrates runtime setup, output folders, memory initialization, optional tracing, dashboards, and lifecycle shutdown.
- `agents.py` instantiates persistent Strands agents, applies plugins, attaches tools, and runs each role in an iterative loop.
- `state.py` centralizes runtime coordination with the blackboard, inboxes, event log, lifecycle states, and request queues.
- `memory.py` provides persistent hybrid retrieval using BM25 and semantic embeddings with Reciprocal Rank Fusion.
- `agent_tools.py` exposes the operational tool surface used by agents.
- `spawner.py` manages temporary specialists for blocking issues or missing domain expertise.
- `steering.py` lets a user inject live remarks into running agent conversations.
- `mcp_server.py` exposes the swarm as an MCP-compatible tool server.

## Core agents

| Agent | Primary responsibility | Main outputs |
|------|------|------|
| `analyste` | Interpret the specification, derive scope, acceptance criteria, stack assumptions, and ADRs | `output/feature_list.json`, `output/adr/ADR-001.md` |
| `architecte` | Create the system architecture and shared contracts | `output/diagrams/*`, `output/shared_dependencies.json` |
| `developpeur` | Generate implementation and tests | `output/src/*`, `output/tests/*` |
| `devops` | Prepare execution and delivery assets | `output/devops/*` |
| `qa` | Execute validation and publish testing status | `output/qa_report.md`, blackboard QA results |
| `reviewer` | Re-run critical checks and issue final PASS/FAIL verdict | `output/review_report.md`, `reviewer_verdict` |

## Communication channels

The swarm communicates through four primary channels plus long-term memory.

| Channel | Backing implementation | Purpose | Typical producer | Typical consumer |
|------|------|------|------|------|
| Blackboard | Shared dict guarded by `_bb_lock` | Broadcast structured outputs and status for all agents | Any agent | All agents |
| Inbox | Per-agent queue guarded by `_inbox_lock` | Direct targeted communication and wake-up messages | Any agent | Specific target agent |
| Filesystem | `output/` directory tree | Persist generated artifacts, diagrams, code, tests, and reports | Any producing agent | All reviewing agents |
| Future return path | Async future resolution from the spawner | Return specialist results to the requesting agent | Spawner / specialist | Blocked or requesting agent |
| Event log | Shared runtime event stream | Operational trace for dashboards and debugging | Runtime and all agents | Dashboards and MCP consumers |

### Channel semantics

- Use the blackboard for durable, shared facts such as `feature_list`, `architecture`, `code_ready`, `qa_status`, and `reviewer_verdict`.
- Use inbox messages when one agent needs another agent to act, respond, or wake up immediately.
- Use the filesystem for anything too large or structured for the blackboard, especially code, tests, diagrams, reports, and deployment assets.
- Use the spawner return path when a specialist has been requested and the requester needs a direct result payload before proceeding.

## Hybrid memory

The swarm includes persistent long-term memory stored under `memory_store/`.

```text
Query -> BM25 keyword retrieval (weight 0.3)
Query -> Semantic embedding retrieval (weight 0.7)
Results -> Reciprocal Rank Fusion (k=60)
Top-K -> Returned to the agent
```

### Memory design

- Keyword layer: `bm25s` for exact or near-exact lexical matches
- Semantic layer: ChromaDB persistent collection with `all-MiniLM-L6-v2`
- Fusion layer: Reciprocal Rank Fusion for robust blended retrieval
- Persistence: documents are saved in `memory_store/docs.json` and Chroma files

### What gets stored

- original project specs
- blackboard entries
- agent completions and summaries
- any explicit `memory_store(...)` calls performed by agents

This lets later iterations reuse previous reasoning, known errors, prior deliverables, and historical decisions.

## Agent tools

All persistent agents receive a common tool base, and some roles receive additional tools.

### Shared tools

| Tool | Purpose |
|------|------|
| `write_blackboard` | Publish shared facts to the blackboard |
| `read_blackboard` | Read one or all blackboard entries |
| `send_message` | Send direct agent-to-agent messages |
| `check_inbox` | Consume queued messages |
| `memory_store` | Persist information to long-term memory |
| `memory_search` | Query long-term memory |
| `write_file` | Write files to `output/` or workspace paths |
| `read_file` | Read files with line ranges |
| `list_files` | Enumerate output or workspace files |
| `run_command` | Execute shell commands in the workspace |
| `report_progress` | Update health and dashboard progress |
| `set_sleep_duration` | Sleep until timeout or wake-up |
| `set_status` | Move between `WAITING`, `WORKING`, `DONE`, and `BLOCKED` |
| `wake_agent` | Wake another agent with an urgent message |

### Specialized tools

| Tool | Availability | Purpose |
|------|------|------|
| `report_stuck` | All persistent agents | Request specialist help when blocked |
| `request_specialist` | `analyste` | Proactively ask the spawner for a domain specialist |
| `submit_verdict` | `reviewer` | Publish the structured PASS/FAIL verdict |
| `tavily_search` | Agents when available | Web research for best practices, errors, and competitive benchmarks |
| browser tool | Agents when available | Read documentation pages and web content |
| `skills(...)` | Agents when local skills are enabled | Activate local AgentSkills instructions on demand |

## Spawner and specialist agents

The spawner listens for two kinds of requests:

- blocked-agent requests via `report_stuck(...)`
- proactive specialist requests via `request_specialist(...)`

When a request arrives, the spawner decides what temporary specialist to instantiate, provides a constrained tool subset, runs the specialist, and returns a summary to the requesting agent.

Typical temporary specialists include:

- language experts
- framework experts
- debugging experts
- DevOps experts
- documentation experts
- database experts
- domain experts such as data scientist, data engineer, ML engineer, analytics engineer, and security specialist

This mechanism keeps the core swarm stable while still allowing the workflow to adapt to project-specific needs.

## Skills

Local AgentSkills are loaded from the `skills/` directory when it exists.

- skills are enabled by default
- set `SWARM_ENABLE_SKILLS=0` to disable them
- skills are exposed through the Strands `skills(...)` tool
- agents can progressively load skill instructions only when needed

The repository currently includes:

- a testing skill at `skills/unit-testing/` (unit, integration, and end-to-end strategy)
- a frontend design skill at `skills/frontend-design/` (professional UI quality, consistent color systems, and no emoji-heavy interfaces)

## User steering

The runtime supports live user steering via `steering.py`.

### What steering does

- injects real-time user remarks into running agent conversations
- broadcasts a remark to all agents or targets a specific agent
- logs steering events so the dashboard and event log can display them

This is useful when the user wants to clarify scope, change direction, or add constraints while a swarm run is already in progress.

## Image and PDF support

The swarm can accept a frontend reference file at startup.

### Supported formats

- `.pdf`
- `.png`
- `.jpg`
- `.jpeg`
- `.gif`
- `.webp`

### How it is used

- the file is loaded into Strands content blocks
- the analyst, architect, developer, and reviewer receive additional instructions to use it
- the reviewer is explicitly told to validate frontend visual fidelity against the provided reference

This is intended for frontend mockups, screenshots, UI references, or multi-page PDF specifications.

## MCP server

The project can run as an MCP server instead of an interactive CLI.

### Supported transports

- `stdio`
- `sse`
- `streamable-http`

### Exposed MCP capabilities

- launch the swarm from a free-text specification
- inspect blackboard state
- inspect agent statuses
- list generated files
- read generated files
- query hybrid memory
- retrieve recent event logs

This allows external MCP-compatible clients to orchestrate or observe the swarm without directly using the CLI or web dashboard.

## Installation

### Prerequisites

- Python 3.11+
- a configured LLM backend for live mode
- optional: Ollama for local models
- optional: Docker, depending on generated outputs
- optional: Jaeger or another OTLP collector for tracing

### Setup

```bash
git clone <your-repo-url> hackathon-ibm
cd hackathon-ibm
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy env.example .env
```

On macOS/Linux, replace the activation and copy commands with the equivalent shell commands.

### Optional local model setup

```bash
ollama pull deepseek-r1:14b
# or
ollama pull qwen2.5-coder:32b
```

## Configuration

Configuration is primarily driven by environment variables and CLI flags.

### Common configuration knobs

- `SWARM_MODEL`: default model selection
- `OPENROUTER_API_KEY`: required for OpenRouter usage
- `OTEL_EXPORTER_OTLP_ENDPOINT`: OTLP endpoint for tracing
- `SWARM_ENABLE_SKILLS`: set to `0` to disable local skills

### Example `.env` usage

```bash
OPENROUTER_API_KEY=sk-or-...
SWARM_MODEL=openrouter/anthropic/claude-3.5-sonnet
```

## Run modes

### Demo mode

Runs a simulated swarm without live LLM calls.

```bash
python main.py --demo
```

### Live mode (For better UI quality, run with `--web-ui` in live mode)

Runs the full swarm with the configured provider.

```bash
python main.py --specs "Build a REST API for inventory management in Go with Gin framework"
```

### Live mode with model override

```bash
python main.py --model ollama/qwen2.5-coder:32b --specs "Build a REST API for inventory management in Go with Gin framework"
```

### OpenRouter example

```bash
set OPENROUTER_API_KEY=sk-or-...
python main.py --provider openrouter --model anthropic/claude-3.5-sonnet --specs "Build a REST API for inventory management in Go with Gin framework"
```

### Web UI mode (RECOMMENDED)

```bash
python main.py --web-ui --specs "Build a REST API for inventory management in Go with Gin framework"
```

Open the dashboard at:

```text
http://127.0.0.1:8500
```

### MCP mode

```bash
python main.py --mcp
python main.py --mcp --mcp-transport sse --mcp-host 127.0.0.1 --mcp-port 8000
python main.py --mcp --mcp-transport streamable-http --mcp-host 127.0.0.1 --mcp-port 8000
```

## Monitoring and observability

The project exposes several monitoring surfaces.

### Dashboards

- Rich terminal dashboard
- Web dashboard at `http://127.0.0.1:8500`

The dashboards surface prompts, responses, tool calls, blackboard writes, inter-agent messages, and lifecycle events.

### OpenTelemetry

Live runs support OpenTelemetry tracing and metrics export.

- tracing is enabled by default in live mode when an OTLP endpoint is configured
- tracing is disabled in demo mode
- you can disable tracing explicitly with `--no-tracing`
- the default OTLP endpoint is `http://localhost:4318`
- traces are intended to be viewed in Jaeger at `http://localhost:16686`

Example:

```bash
set OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
python main.py --specs "Build a REST API for inventory management in Go with Gin framework"
```

### MCP inspection tools

If the system is running in MCP mode, clients can inspect:

- blackboard contents
- agent statuses
- output files
- hybrid memory
- recent event logs

## Generated outputs

The swarm writes generated artifacts under `output/`.

Typical subdirectories include:

- `output/src/`
- `output/tests/`
- `output/adr/`
- `output/diagrams/`
- `output/devops/`

Reports commonly include:

- `output/qa_report.md`
- `output/review_report.md`

## Known problems and future fixes

This codebase was produced under hackathon pressure in roughly three days. It already works for demos and experimentation, but there are known runtime issues that should be fixed properly in a future stabilization pass.

### Known issues

- sometimes the swarm does not stop even though all core agents are `DONE` and the reviewer verdict is `PASS`
- sometimes individual agents do not transition to `DONE` reliably, even when their work is effectively finished

### Likely improvement areas

- make completion detection more deterministic and less dependent on timing between the watcher, reviewer, and agent loops
- tighten lifecycle state transitions so `DONE`, `WAITING`, and wake-up behavior are easier to reason about
- add stronger automated tests around swarm shutdown, reviewer verdict submission, and stuck/specialist request flows
- improve resilience around background tasks and race conditions in asyncio plus thread interactions

These issues are explicitly acknowledged here because the project was built fast, under real hackathon constraints, and now needs a more deliberate hardening phase.
