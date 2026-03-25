"""Real-time SSE-based web dashboard for multi-agent swarm monitoring.

Provides a FastAPI server with Server-Sent Events streaming, agent lifecycle
tracking, blackboard/inbox state synchronization, and a dark-themed UI for
visualization of the orchestration pipeline.
"""

import asyncio
import base64
import json
import os
import time
import threading
import logging
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import (
    HTMLResponse,
    StreamingResponse,
    JSONResponse,
    FileResponse,
)
import uvicorn

logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

app = FastAPI(title="Agent Swarm Dashboard")

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

_history: list[dict] = []
_history_lock = threading.Lock()
_subscribers: list[asyncio.Queue] = []
_sub_lock = threading.Lock()

_current_blackboard: dict = {}
_current_inboxes: dict = {}
_bb_lock = threading.Lock()

_agents: dict[str, dict] = {}
_agents_lock = threading.Lock()

_pipeline_status = {"status": "idle"}

_server_loop: asyncio.AbstractEventLoop | None = None

_start_callback: Callable[[str, str | None], None] | None = None
_start_lock = threading.Lock()
_uploaded_file_path: str | None = None

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")

_next_event_id = 0
_event_id_lock = threading.Lock()
_replay_buffer: list[dict] = []
_replay_lock = threading.Lock()


def _broadcast(event: dict):
    """Assign a monotonic ID, buffer for replay, and broadcast.

    Thread-safe: uses ``call_soon_threadsafe`` to put events into the
    asyncio subscriber queues from orchestrator threads.
    """
    global _next_event_id
    with _event_id_lock:
        _next_event_id += 1
        event["id"] = _next_event_id
    with _replay_lock:
        _replay_buffer.append(event)
        if len(_replay_buffer) > 8000:
            _replay_buffer[:] = _replay_buffer[-6000:]
    with _sub_lock:
        dead = []
        for q in _subscribers:
            try:
                if _server_loop and _server_loop.is_running():
                    _server_loop.call_soon_threadsafe(q.put_nowait, event)
                else:
                    q.put_nowait(event)
            except Exception:
                dead.append(q)
        for q in dead:
            _subscribers.remove(q)


# ---------------------------------------------------------------------------
# Public API (called from orchestrator threads)
# ---------------------------------------------------------------------------


def push_event(agent: str, phase: str, message: str):
    """Record and broadcast an agent event.

    Args:
        agent: Name of the emitting agent.
        phase: Event phase label (e.g. TOOL, MSG, PROMPT, RESPONSE).
        message: Human-readable event description.
    """
    evt = {
        "type": "event",
        "agent": agent,
        "phase": phase,
        "message": message,
        "timestamp": time.time(),
    }
    with _history_lock:
        _history.append(evt)
        if len(_history) > 5000:
            _history[:] = _history[-4000:]

    # Update agent tracking
    with _agents_lock:
        if agent not in _agents:
            _agents[agent] = {
                "name": agent,
                "status": "idle",
                "events": 0,
                "tools": 0,
                "messages": 0,
            }
        a = _agents[agent]
        a["events"] = a.get("events", 0) + 1
        if phase == "TOOL":
            a["tools"] = a.get("tools", 0) + 1
        if phase == "MSG":
            a["messages"] = a.get("messages", 0) + 1
        if phase in ("PROMPT", "THINK", "RESPONSE"):
            a["llm"] = a.get("llm", 0) + 1

    _broadcast(evt)


def push_state(blackboard_dict: dict, inboxes_dict: dict):
    """Synchronize blackboard and inbox snapshots to the dashboard.

    Args:
        blackboard_dict: Current blackboard key-value mapping.
        inboxes_dict: Per-agent inbox message lists.
    """
    global _current_blackboard, _current_inboxes
    with _bb_lock:
        _current_blackboard = blackboard_dict
        _current_inboxes = inboxes_dict
    evt = {
        "type": "state_sync",
        "timestamp": time.time(),
        "blackboard": blackboard_dict,
        "inboxes": inboxes_dict,
    }
    _broadcast(evt)


def set_pipeline_status(status: str):
    """Update and broadcast the global pipeline status.

    Args:
        status: One of 'idle', 'running', 'done', or 'error'.
    """
    global _pipeline_status
    _pipeline_status = {"status": status, "timestamp": time.time()}
    _broadcast({"type": "pipeline_status", **_pipeline_status})


def push_agent_status(agent: str, status: str):
    """Broadcast an agent lifecycle status update.

    Args:
        agent: Agent name.
        status: Lifecycle state (WAITING, WORKING, DONE, BLOCKED).
    """
    with _agents_lock:
        if agent not in _agents:
            _agents[agent] = {
                "name": agent,
                "status": status.lower(),
                "events": 0,
                "tools": 0,
                "messages": 0,
            }
        else:
            _agents[agent]["status"] = status.lower()
    evt = {
        "type": "agent_status",
        "agent": agent,
        "status": status.lower(),
        "timestamp": time.time(),
    }
    _broadcast(evt)


# ---------------------------------------------------------------------------
# FastAPI routes
# ---------------------------------------------------------------------------


@app.get("/api/events")
async def sse_events(request: Request):
    """Open an SSE stream delivering real-time swarm events.

    Supports ``Last-Event-ID`` header so the browser can resume after
    a disconnect without losing events.
    """
    last_id = 0
    raw = request.headers.get("Last-Event-ID") or request.query_params.get(
        "lastEventId"
    )
    if raw:
        try:
            last_id = int(raw)
        except (ValueError, TypeError):
            pass

    q: asyncio.Queue = asyncio.Queue()
    with _sub_lock:
        _subscribers.append(q)

    async def stream():
        try:
            # Replay events the client missed during the disconnect gap
            if last_id > 0:
                with _replay_lock:
                    missed = [e for e in _replay_buffer if e.get("id", 0) > last_id]
                for evt in missed:
                    yield f"id: {evt['id']}\ndata: {json.dumps(evt, default=str)}\n\n"

            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield f"id: {event.get('id', '')}\ndata: {json.dumps(event, default=str)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            with _sub_lock:
                if q in _subscribers:
                    _subscribers.remove(q)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/history")
async def get_history(last_n: int = 5000):
    """Return the most recent *last_n* events from history."""
    with _history_lock:
        return JSONResponse(_history[-last_n:])


@app.get("/api/agents")
async def get_agents():
    """Return the current list of registered agents."""
    with _agents_lock:
        return JSONResponse(list(_agents.values()))


@app.get("/api/blackboard")
async def get_blackboard():
    """Return the current blackboard snapshot."""
    with _bb_lock:
        return JSONResponse(_current_blackboard)


@app.get("/api/inboxes")
async def get_inboxes():
    """Return per-agent inbox message queues."""
    with _bb_lock:
        return JSONResponse(_current_inboxes)


@app.get("/api/status")
async def get_status():
    """Return the current pipeline status."""
    return JSONResponse(_pipeline_status)


@app.get("/api/deliverables")
async def list_deliverables():
    """List all generated deliverable files with relative paths and sizes."""
    if not os.path.isdir(OUTPUT_DIR):
        return JSONResponse([])
    files = []
    for root, dirs, filenames in os.walk(OUTPUT_DIR):
        for fn in filenames:
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, OUTPUT_DIR)
            files.append(
                {"path": rel.replace("\\", "/"), "size": os.path.getsize(full)}
            )
    return JSONResponse(files)


@app.get("/api/deliverables/{filepath:path}")
async def download_deliverable(filepath: str):
    """Serve a deliverable file with path-traversal protection."""
    safe = os.path.join(OUTPUT_DIR, filepath.replace("..", ""))
    real_output = os.path.realpath(OUTPUT_DIR)
    real_safe = os.path.realpath(safe)
    if not real_safe.startswith(real_output) or not os.path.isfile(safe):
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(safe, filename=os.path.basename(safe))


# ---------------------------------------------------------------------------
# Swarm start / upload API
# ---------------------------------------------------------------------------


def register_start_callback(cb: Callable[[str, str | None], None]):
    """Register the callback invoked when the user launches the swarm.

    Args:
        cb: Callable receiving (specs_text, optional_image_path).
    """
    global _start_callback
    with _start_lock:
        _start_callback = cb


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Accept and persist an uploaded file (image or PDF)."""
    global _uploaded_file_path
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    safe_name = os.path.basename(file.filename or "upload.bin").replace("..", "")
    dest = os.path.join(UPLOAD_DIR, safe_name)
    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)
    _uploaded_file_path = dest
    return JSONResponse({"ok": True, "path": safe_name, "size": len(content)})


@app.post("/api/start")
async def start_swarm(request: Request):
    """Trigger the swarm pipeline with user-provided specifications."""
    body = await request.json()
    specs = body.get("specs", "").strip()
    if not specs:
        return JSONResponse({"error": "No specs provided"}, status_code=400)
    with _start_lock:
        cb = _start_callback
    if cb is None:
        return JSONResponse({"error": "Backend not ready"}, status_code=503)
    try:
        img_path = _uploaded_file_path
        cb(specs, img_path)
        return JSONResponse({"ok": True})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/steer")
async def steer_agents(request: Request):
    """Inject a steering message directed at one or all agents."""
    body = await request.json()
    message = body.get("message", "").strip()
    target = body.get("target", "all")
    if not message:
        return JSONResponse({"error": "Empty message"}, status_code=400)
    push_event("user", "STEER", f"[{target}] {message}")
    return JSONResponse({"ok": True})


# ---------------------------------------------------------------------------
# HTML serving
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the single-page monitoring dashboard."""
    return HTMLResponse(HTML_TEMPLATE)


# ---------------------------------------------------------------------------
# Server startup
# ---------------------------------------------------------------------------


def start_dashboard_server(host="127.0.0.1", port=8500):
    """Launch the dashboard server on a background daemon thread.

    Args:
        host: Bind address.
        port: Listening port.
    """
    config = uvicorn.Config(app, host=host, port=port, log_level="error")
    server = uvicorn.Server(config)

    def run_server():
        global _server_loop
        _server_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_server_loop)
        _server_loop.run_until_complete(server.serve())

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    print(f"\n  Web Dashboard: http://{host}:{port}\n")


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Agent Swarm Monitor</title>
<style>
/* ── VARIABLES ── */
:root {
  --bg:        #06080f;
  --bg2:       #0b0e1b;
  --surface:   #0e1221;
  --surface2:  #121829;
  --surface3:  #181f35;
  --border:    rgba(255,255,255,0.07);
  --border2:   rgba(255,255,255,0.13);
  --text:      #eef1ff;
  --text2:     #7d8baa;
  --text3:     #404d6a;
  --red:       #e8604a;
  --red2:      #ff7b5c;
  --red-bg:    rgba(232,96,74,0.13);
  --green:     #2dd4a0;
  --green-bg:  rgba(45,212,160,0.11);
  --orange:    #f5a841;
  --orange-bg: rgba(245,168,65,0.11);
  --blue:      #6babff;
  --blue-bg:   rgba(107,171,255,0.11);
  --pink:      #f472b6;
  --cyan:      #4ecdc4;
  --teal:      #4ecdc4;
  --accent:    #e8604a;
  --accent2:   #ff7b5c;
  --r:         14px;
  --r-sm:      9px;
  --r-xs:      5px;
}

*{margin:0;padding:0;box-sizing:border-box;}

body{
  font-family:'Inter','SF Pro Display',-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;
  background:var(--bg);color:var(--text);
  height:100vh;overflow:hidden;
  background-image:
    radial-gradient(ellipse 70% 50% at 8% 55%,  rgba(232,96,74,.06)  0%,transparent 70%),
    radial-gradient(ellipse 55% 45% at 92% 25%,  rgba(78,205,196,.06) 0%,transparent 70%),
    radial-gradient(ellipse 40% 30% at 50% 100%, rgba(107,171,255,.04) 0%,transparent 60%);
}

/* subtle starfield */
body::before{
  content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
  background-image:
    radial-gradient(1px 1px at 18%  22%, rgba(255,255,255,.18) 0%,transparent 100%),
    radial-gradient(1px 1px at 72%  8%,  rgba(255,255,255,.14) 0%,transparent 100%),
    radial-gradient(1px 1px at 43%  58%, rgba(255,255,255,.12) 0%,transparent 100%),
    radial-gradient(1px 1px at 88%  44%, rgba(255,255,255,.16) 0%,transparent 100%),
    radial-gradient(1px 1px at 30%  80%, rgba(255,255,255,.10) 0%,transparent 100%),
    radial-gradient(1px 1px at 60%  33%, rgba(255,255,255,.12) 0%,transparent 100%),
    radial-gradient(1px 1px at 92%  72%, rgba(255,255,255,.14) 0%,transparent 100%),
    radial-gradient(1px 1px at 5%   60%, rgba(255,255,255,.09) 0%,transparent 100%),
    radial-gradient(1px 1px at 55%  90%, rgba(255,255,255,.10) 0%,transparent 100%);
}

::-webkit-scrollbar{width:4px;}
::-webkit-scrollbar-track{background:transparent;}
::-webkit-scrollbar-thumb{background:rgba(255,255,255,.1);border-radius:4px;}
::-webkit-scrollbar-thumb:hover{background:rgba(255,255,255,.2);}

@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
@keyframes slideIn{from{transform:translateX(100%);opacity:0}to{transform:translateX(0);opacity:1}}
@keyframes fadeIn{from{opacity:0;transform:translateY(7px)}to{opacity:1;transform:translateY(0)}}
@keyframes shimmer{0%{background-position:200% center}100%{background-position:-200% center}}

/* ── HEADER ── */
.header{
  background:rgba(10,13,25,.82);
  backdrop-filter:blur(22px);-webkit-backdrop-filter:blur(22px);
  border-bottom:1px solid var(--border);
  padding:0 24px;display:flex;align-items:center;justify-content:space-between;
  height:58px;position:relative;z-index:100;
}
.header::after{
  content:'';position:absolute;bottom:0;left:8%;right:8%;height:1px;
  background:linear-gradient(90deg,transparent,var(--accent) 35%,var(--teal) 65%,transparent);
  opacity:.3;
}
.header-left{display:flex;align-items:center;gap:16px;}
.logo{
  width:38px;height:38px;border-radius:11px;
  background:linear-gradient(135deg,var(--accent) 0%,var(--teal) 100%);
  display:flex;align-items:center;justify-content:center;
  font-size:19px;font-weight:900;color:white;
  box-shadow:0 0 22px rgba(232,96,74,.35),0 3px 10px rgba(0,0,0,.5);
}
.brand{
  font-size:18px;font-weight:800;letter-spacing:-.4px;
  background:linear-gradient(90deg,var(--accent) 0%,#ff9f7a 45%,var(--teal) 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}
.tagline{
  font-size:9px;color:var(--text3);letter-spacing:2.8px;
  text-transform:uppercase;margin-top:2px;font-weight:600;
}
.pip-badge{
  padding:4px 14px;border-radius:20px;font-size:10px;
  font-weight:700;text-transform:uppercase;letter-spacing:.9px;
}
.badge-idle   {background:rgba(255,255,255,.05);color:var(--text3);border:1px solid var(--border);}
.badge-running{background:var(--red-bg);color:var(--red);border:1px solid rgba(232,96,74,.28);animation:pulse 1.5s infinite;}
.badge-done   {background:var(--green-bg);color:var(--green);border:1px solid rgba(45,212,160,.28);}
.badge-error  {background:rgba(232,96,74,.18);color:var(--red2);border:1px solid rgba(232,96,74,.38);}
.header-stats{display:flex;gap:22px;font-size:11px;color:var(--text2);}
.header-stats .sv{color:var(--text);font-weight:700;margin-left:5px;}

/* ── LAYOUT ── */
.layout{
  display:grid;grid-template-columns:235px 1fr 295px;
  height:calc(100vh - 58px - 68px);position:relative;z-index:1;
}

/* ── LEFT PANEL ── */
.panel-left{
  background:rgba(12,16,30,.72);border-right:1px solid var(--border);
  display:flex;flex-direction:column;overflow:hidden;
  backdrop-filter:blur(10px);
}
.ph{
  padding:14px 16px 10px;font-size:9.5px;font-weight:700;
  text-transform:uppercase;letter-spacing:1.6px;color:var(--text3);
  border-bottom:1px solid var(--border);display:flex;align-items:center;gap:7px;
}
.ph .dot-h{
  width:6px;height:6px;border-radius:50%;
  background:linear-gradient(135deg,var(--accent),var(--teal));
  box-shadow:0 0 6px rgba(232,96,74,.5);
}
.agents-list{flex:1;overflow-y:auto;padding:10px;}
.agent-card{
  background:rgba(255,255,255,.03);border:1px solid var(--border);
  border-radius:var(--r);padding:12px 14px;margin-bottom:8px;
  cursor:pointer;transition:all .2s cubic-bezier(.4,0,.2,1);position:relative;overflow:hidden;
}
.agent-card::after{
  content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.12),transparent);
  opacity:0;transition:opacity .2s;
}
.agent-card:hover{border-color:var(--border2);background:rgba(255,255,255,.055);transform:translateY(-1px);}
.agent-card:hover::after{opacity:1;}
.agent-card.selected{
  border-color:rgba(232,96,74,.45);background:rgba(232,96,74,.08);
  box-shadow:0 0 22px rgba(232,96,74,.14),inset 0 1px 0 rgba(255,255,255,.06);
}
.adot{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:8px;flex-shrink:0;}
.adot-idle   {background:var(--text3);}
.adot-working{background:var(--accent);animation:pulse .9s infinite;box-shadow:0 0 8px rgba(232,96,74,.6);}
.adot-done   {background:var(--green);box-shadow:0 0 6px rgba(45,212,160,.4);}
.adot-error  {background:var(--red2);}
.adot-waiting{background:var(--orange);animation:pulse 1.6s infinite;box-shadow:0 0 6px rgba(240,169,85,.5);}
.adot-blocked{background:var(--red);animation:pulse 1.2s infinite;box-shadow:0 0 8px rgba(232,93,93,.5);}
.aname{font-weight:700;font-size:13px;letter-spacing:-.1px;}
.astatus{font-size:9px;font-weight:700;text-transform:uppercase;margin-top:5px;letter-spacing:.6px;}
.acounts{font-size:9px;color:var(--text3);margin-top:5px;display:flex;gap:10px;}

/* input area */
.input-section{
  background:rgba(10,13,24,.9);border-top:1px solid var(--border);
  padding:11px 12px;display:flex;gap:8px;align-items:flex-end;
}
.input-section textarea{
  flex:1;background:rgba(255,255,255,.04);color:var(--text);
  border:1px solid var(--border);border-radius:var(--r-sm);padding:8px 12px;
  font-size:11.5px;resize:vertical;min-height:38px;max-height:100px;
  font-family:inherit;transition:border-color .18s;
}
.input-section textarea::placeholder{color:var(--text3);}
.input-section textarea:focus{outline:none;border-color:rgba(232,96,74,.45);}
.btn-send{
  background:linear-gradient(135deg,var(--accent) 0%,var(--accent2) 100%);
  color:white;border:none;padding:9px 16px;border-radius:var(--r-sm);
  cursor:pointer;font-weight:700;font-size:11px;white-space:nowrap;
  box-shadow:0 3px 14px rgba(232,96,74,.32);transition:all .18s;
}
.btn-send:hover{transform:translateY(-1px);box-shadow:0 5px 20px rgba(232,96,74,.45);}

/* ── CENTER PANEL ── */
.panel-center{display:flex;flex-direction:column;overflow:hidden;}
.main-tabs{
  display:flex;background:rgba(11,14,25,.8);border-bottom:1px solid var(--border);
  padding:0 20px;flex-shrink:0;backdrop-filter:blur(10px);
}
.mtab{
  padding:15px 15px;font-size:11px;font-weight:600;color:var(--text3);
  cursor:pointer;border-bottom:2px solid transparent;transition:all .2s;
  display:flex;align-items:center;gap:6px;letter-spacing:.15px;
}
.mtab:hover{color:var(--text2);}
.mtab.active{color:var(--text);border-bottom-color:var(--accent);}
.tc{
  font-size:9px;background:rgba(255,255,255,.07);color:var(--text3);
  padding:1px 6px;border-radius:10px;min-width:20px;text-align:center;font-weight:700;
}
.mtab.active .tc{background:var(--red-bg);color:var(--accent);}

/* filter bar */
.filter-bar{
  padding:8px 18px;background:rgba(10,13,24,.6);border-bottom:1px solid var(--border);
  display:flex;align-items:center;gap:10px;flex-shrink:0;
}
.filter-bar label{
  font-size:9.5px;color:var(--text3);font-weight:700;
  letter-spacing:.8px;text-transform:uppercase;
}
.filter-bar select,.filter-bar input{
  background:rgba(255,255,255,.05);color:var(--text);
  border:1px solid var(--border);border-radius:var(--r-sm);
  padding:5px 11px;font-size:11px;font-family:inherit;transition:border-color .18s;
}
.filter-bar select:focus,.filter-bar input:focus{outline:none;border-color:rgba(232,96,74,.45);}
.filter-bar input::placeholder{color:var(--text3);}
.fbtn{
  background:rgba(255,255,255,.05);color:var(--text3);
  border:1px solid var(--border);border-radius:var(--r-sm);
  padding:5px 11px;font-size:10px;cursor:pointer;font-weight:600;transition:all .18s;
}
.fbtn.on{background:var(--red-bg);color:var(--accent);border-color:rgba(232,96,74,.35);}

/* tab content */
.tab-content{flex:1;overflow:hidden;position:relative;}
.tpage{display:none;height:100%;overflow-y:auto;padding:0;}
.tpage.active{display:block;}

/* log table */
.ltbl{
  width:100%;font-size:11.5px;border-collapse:collapse;
  font-family:'SF Mono','Cascadia Code','Fira Code',Consolas,monospace;
}
.ltbl tr{border-bottom:1px solid rgba(255,255,255,.04);}
.ltbl tr:hover{background:rgba(255,255,255,.025);}
.ltbl td{padding:5px 10px;vertical-align:top;white-space:nowrap;}
.ltbl td:last-child{white-space:normal;word-break:break-word;width:100%;line-height:1.5;}
.ltbl .ts{color:var(--text3);font-size:10px;letter-spacing:.2px;}
.ltbl .tag{
  display:inline-block;padding:2px 9px;border-radius:8px;
  font-size:9px;font-weight:700;letter-spacing:.4px;
}
.ltbl .et{
  display:inline-block;padding:2px 8px;border-radius:8px;
  font-size:9px;font-weight:700;letter-spacing:.4px;
}

/* detail cards */
.dcard{
  background:rgba(255,255,255,.03);border:1px solid var(--border);
  border-radius:var(--r);margin:10px 14px;overflow:hidden;
  animation:fadeIn .25s ease-out;transition:border-color .2s,background .2s;
}
.dcard:hover{border-color:var(--border2);background:rgba(255,255,255,.04);}
.dcard-h{
  padding:10px 15px;display:flex;justify-content:space-between;align-items:center;
  font-size:11px;border-bottom:1px solid var(--border);
}
.dcard-h .dl{display:flex;align-items:center;gap:10px;flex-wrap:wrap;}
.dcard-h .dt{color:var(--text3);font-size:10px;}
.dcard-h .dlbl{font-weight:700;font-size:12px;letter-spacing:-.1px;}
.dcard-h .dm{color:var(--text2);font-size:10px;}
.dcard-b{
  padding:11px 15px;font-family:'SF Mono','Cascadia Code','Fira Code',Consolas,monospace;
  font-size:11.5px;line-height:1.65;white-space:pre-wrap;word-break:break-word;
  max-height:360px;overflow-y:auto;color:var(--text);background:rgba(0,0,0,.22);
}
.dcard.br-red   {border-left:3px solid var(--red);}
.dcard.br-green {border-left:3px solid var(--green);}
.dcard.br-orange{border-left:3px solid var(--orange);}
.dcard.br-blue  {border-left:3px solid var(--blue);}
.dcard.br-pink  {border-left:3px solid var(--pink);}
.dcard.br-cyan  {border-left:3px solid var(--cyan);}
.dcard.br-accent{border-left:3px solid var(--accent);}

/* ── RIGHT PANEL ── */
.panel-right{
  background:rgba(12,16,30,.72);border-left:1px solid var(--border);
  display:flex;flex-direction:column;overflow:hidden;backdrop-filter:blur(10px);
}
.rtabs{display:flex;border-bottom:1px solid var(--border);flex-shrink:0;}
.rtab{
  flex:1;padding:13px 6px;font-size:9.5px;font-weight:700;color:var(--text3);
  text-align:center;cursor:pointer;border-bottom:2px solid transparent;
  transition:all .2s;text-transform:uppercase;letter-spacing:1px;
}
.rtab:hover{color:var(--text2);}
.rtab.active{color:var(--accent);border-bottom-color:var(--accent);}
.rcontent{flex:1;overflow-y:auto;}
.rpage{display:none;padding:10px;}
.rpage.active{display:block;}

/* blackboard cards */
.bb-card{
  background:rgba(255,255,255,.03);border:1px solid var(--border);
  border-radius:var(--r);padding:13px 15px;margin-bottom:9px;
  transition:all .2s;border-left:3px solid rgba(232,96,74,.55);
}
.bb-card:hover{border-color:var(--border2);background:rgba(255,255,255,.05);}
.bb-key{font-weight:700;color:var(--accent);font-size:12px;margin-bottom:5px;letter-spacing:-.1px;}
.bb-author{
  font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.7px;
  padding:2px 8px;border-radius:10px;
  background:rgba(255,255,255,.06);color:var(--text2);
}
.bb-val{
  font-size:11.5px;color:var(--text);margin-top:9px;
  font-family:'SF Mono','Cascadia Code','Fira Code',Consolas,monospace;
  background:rgba(0,0,0,.28);padding:9px 11px;border-radius:var(--r-sm);
  max-height:130px;overflow-y:auto;white-space:pre-wrap;word-break:break-word;
  line-height:1.5;border:1px solid var(--border);
}
.bb-time{font-size:9px;color:var(--text3);margin-top:6px;}
.bb-count{
  background:var(--red-bg);color:var(--accent);font-size:10px;font-weight:700;
  padding:2px 10px;border-radius:10px;border:1px solid rgba(232,96,74,.25);
}

/* inbox cards */
.inbox-group{
  background:rgba(255,255,255,.03);border:1px solid var(--border);
  border-radius:var(--r);margin-bottom:9px;overflow:hidden;
}
.inbox-hdr{
  padding:11px 14px;cursor:pointer;display:flex;align-items:center;
  justify-content:space-between;font-size:12px;font-weight:600;
  border-bottom:1px solid transparent;transition:all .18s;
}
.inbox-hdr:hover{background:rgba(255,255,255,.04);}
.inbox-hdr.open{border-bottom-color:var(--border);}
.inbox-cnt{
  font-size:9px;font-weight:700;padding:2px 8px;border-radius:10px;
  background:var(--red-bg);color:var(--accent);border:1px solid rgba(232,96,74,.25);
}
.inbox-body{display:none;padding:8px 12px;max-height:200px;overflow-y:auto;}
.inbox-body.open{display:block;}
.inbox-msg{
  background:rgba(0,0,0,.22);border-radius:var(--r-sm);
  padding:9px 11px;margin-bottom:6px;font-size:10.5px;
  border-left:2px solid rgba(232,96,74,.4);line-height:1.5;
}
.inbox-msg .im-from{
  font-weight:700;color:var(--text2);font-size:9px;margin-bottom:4px;
  text-transform:uppercase;letter-spacing:.5px;
}
.inbox-msg .im-body{color:var(--text);white-space:pre-wrap;word-break:break-word;}

/* file list */
.frow{
  display:flex;justify-content:space-between;align-items:center;
  padding:9px 13px;border-bottom:1px solid var(--border);font-size:11px;
  transition:background .15s;
}
.frow:hover{background:rgba(255,255,255,.03);}
.frow a{color:var(--blue);text-decoration:none;font-weight:500;}
.frow a:hover{color:var(--teal);}
.frow .fsz{color:var(--text3);font-size:10px;font-weight:600;}

/* overlay */
.overlay{
  display:none;position:fixed;top:0;right:0;width:62%;height:100vh;
  background:rgba(8,11,22,.97);border-left:1px solid var(--border);
  z-index:200;overflow:hidden;
  box-shadow:-24px 0 70px rgba(0,0,0,.8);
  animation:slideIn .28s cubic-bezier(.16,1,.3,1);
  flex-direction:column;backdrop-filter:blur(28px);
}
.overlay::before{
  content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,var(--accent) 0%,#ff9f7a 45%,var(--teal) 100%);
  z-index:1;
}
.overlay.open{display:flex;}
.ov-header{
  border-bottom:1px solid var(--border);
  padding:18px 22px;display:flex;justify-content:space-between;align-items:center;flex-shrink:0;
}
.ov-header h2{
  font-size:18px;font-weight:800;display:flex;align-items:center;gap:12px;letter-spacing:-.3px;
}
.ov-close{
  background:rgba(255,255,255,.07);border:1px solid var(--border);color:var(--text2);
  padding:6px 16px;border-radius:var(--r-sm);cursor:pointer;font-size:11px;font-weight:600;
  transition:all .18s;
}
.ov-close:hover{background:rgba(255,255,255,.13);color:var(--text);}
.ov-tabs{
  display:flex;border-bottom:1px solid var(--border);padding:0 22px;flex-shrink:0;
}
.ov-tab{
  padding:13px 15px;font-size:10px;font-weight:700;color:var(--text3);cursor:pointer;
  border-bottom:2px solid transparent;text-transform:uppercase;letter-spacing:.9px;transition:all .2s;
}
.ov-tab:hover{color:var(--text2);}
.ov-tab.active{color:var(--accent);border-bottom-color:var(--accent);}
.ov-body{flex:1;overflow-y:auto;padding:0;}
.ov-sec{display:none;}
.ov-sec.active{display:block;padding:6px 0;}
.empty{
  color:var(--text3);font-size:12px;padding:36px;text-align:center;
  letter-spacing:.4px;font-weight:500;
}

/* ── BOTTOM BAR ── */
.bottom-bar{
  position:relative;z-index:50;
  background:rgba(10,13,25,.88);
  backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px);
  border-top:1px solid var(--border);
  padding:12px 24px;display:flex;align-items:flex-end;gap:10px;
}
.bottom-bar::before{
  content:'';position:absolute;top:0;left:8%;right:8%;height:1px;
  background:linear-gradient(90deg,transparent,rgba(232,96,74,.25),rgba(78,205,196,.25),transparent);
}
.bottom-bar textarea{
  flex:1;background:rgba(255,255,255,.04);color:var(--text);
  border:1px solid var(--border);border-radius:12px;padding:12px 16px;
  font-size:13px;resize:none;min-height:44px;max-height:120px;
  font-family:inherit;transition:border-color .18s;line-height:1.4;
}
.bottom-bar textarea::placeholder{color:var(--text3);}
.bottom-bar textarea:focus{outline:none;border-color:rgba(232,96,74,.45);}
.bb-actions{display:flex;gap:8px;align-items:center;flex-shrink:0;}
.bb-upload{
  background:rgba(255,255,255,.06);border:1px solid var(--border);
  color:var(--text2);border-radius:10px;padding:10px 14px;cursor:pointer;
  font-size:12px;transition:all .18s;display:flex;align-items:center;gap:6px;
}
.bb-upload:hover{background:rgba(255,255,255,.1);color:var(--text);}
.bb-upload.has-file{border-color:rgba(45,212,160,.4);color:var(--green);}
.bb-send{
  background:linear-gradient(135deg,var(--accent) 0%,var(--accent2) 100%);
  color:white;border:none;padding:10px 22px;border-radius:10px;
  cursor:pointer;font-weight:700;font-size:13px;white-space:nowrap;
  box-shadow:0 3px 14px rgba(232,96,74,.32);transition:all .18s;
  display:flex;align-items:center;gap:7px;
}
.bb-send:hover{transform:translateY(-1px);box-shadow:0 5px 20px rgba(232,96,74,.45);}
.bb-send:disabled{opacity:.4;cursor:not-allowed;transform:none;box-shadow:none;}
.upload-name{
  font-size:10px;color:var(--teal);max-width:130px;overflow:hidden;
  text-overflow:ellipsis;white-space:nowrap;
}
</style>
</head>
<body>

<!-- HEADER -->
<div class="header">
  <div class="header-left">
    <div class="logo">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/>
      </svg>
    </div>
    <div>
      <div class="brand">NexusClaw</div>
      <div class="tagline">The AI that actually builds things.</div>
    </div>
    <div id="pip-badge" class="pip-badge badge-idle">IDLE</div>
  </div>
  <div class="header-stats">
    <span>Events <span class="sv" id="s-events">0</span></span>
    <span>Agents <span class="sv" id="s-agents">0</span></span>
    <span>Files <span class="sv" id="s-files">0</span></span>
    <span>Errors <span class="sv" id="s-errors" style="color:var(--red)">0</span></span>
  </div>
</div>

<!-- MAIN LAYOUT -->
<div class="layout">

  <!-- LEFT: AGENTS -->
  <div class="panel-left">
    <div class="ph"><span class="dot-h"></span> Agents</div>
    <div class="agents-list" id="agents-list"></div>
  </div>

  <!-- CENTER: TABS + LOG -->
  <div class="panel-center">
    <div class="main-tabs">
      <div class="mtab active" onclick="switchTab('full-log',this)">Full Log <span class="tc" id="tc-log">0</span></div>
      <div class="mtab" onclick="switchTab('llm-tab',this)">LLM <span class="tc" id="tc-llm">0</span></div>
      <div class="mtab" onclick="switchTab('tools-tab',this)">Tools <span class="tc" id="tc-tools">0</span></div>
      <div class="mtab" onclick="switchTab('comms-tab',this)">Comms <span class="tc" id="tc-comms">0</span></div>
      <div class="mtab" onclick="switchTab('errors-tab',this)">Errors <span class="tc" id="tc-errors">0</span></div>
    </div>
    <div class="filter-bar">
      <label>Agent:</label>
      <select id="f-agent" onchange="applyFilters()"><option value="all">All</option></select>
      <label>Search:</label>
      <input id="f-text" type="text" placeholder="filter..." oninput="applyFilters()" style="width:120px">
      <div class="fbtn on" id="f-auto" onclick="toggleAuto()">Auto-scroll</div>
    </div>
    <div class="tab-content">
      <div class="tpage active" id="p-full-log"><table class="ltbl"><tbody id="log-body"></tbody></table></div>
      <div class="tpage" id="p-llm-tab"><div id="llm-list"></div></div>
      <div class="tpage" id="p-tools-tab"><div id="tools-list"></div></div>
      <div class="tpage" id="p-comms-tab"><div id="comms-list"></div></div>
      <div class="tpage" id="p-errors-tab"><div id="errors-list"></div></div>
    </div>
  </div>

  <!-- RIGHT: BLACKBOARD + FILES -->
  <div class="panel-right">
    <div class="rtabs">
      <div class="rtab active" onclick="switchRight('r-bb',this)">Blackboard</div>
      <div class="rtab" onclick="switchRight('r-inbox',this)">Inboxes</div>
      <div class="rtab" onclick="switchRight('r-files',this)">Files</div>
    </div>
    <div class="rcontent">
      <div class="rpage active" id="r-bb">
        <div class="ph" style="border-bottom:none;padding:8px 8px 4px;">
          Keys <span class="bb-count" id="bb-count">0</span>
        </div>
        <div id="bb-list"><div class="empty">Blackboard is empty</div></div>
      </div>
      <div class="rpage" id="r-inbox"><div id="inbox-list"><div class="empty">No messages yet</div></div></div>
      <div class="rpage" id="r-files"><div id="files-list"><div class="empty">No files yet</div></div></div>
    </div>
  </div>
</div>

<!-- BOTTOM INPUT BAR -->
<div class="bottom-bar" id="bottom-bar">
  <label class="bb-upload" id="upload-btn" title="Attach image or PDF">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19a4 4 0 015.66 5.66l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48"/></svg>
    <span id="upload-label">Attach</span>
    <input type="file" id="file-input" accept="image/*,.pdf" style="display:none" onchange="handleFileUpload(this)">
  </label>
  <span class="upload-name" id="upload-name" style="display:none"></span>
  <textarea id="main-input" placeholder="Describe your project specifications... (Enter to launch)" rows="1"
    oninput="autoGrow(this)"
    onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();handleSubmit();}"></textarea>
  <button class="bb-send" id="send-btn" onclick="handleSubmit()">
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M22 2L11 13"/><path d="M22 2l-7 20-4-9-9-4 20-7z"/></svg>
    <span id="send-label">Launch</span>
  </button>
</div>

<!-- OVERLAY -->
<div id="overlay" class="overlay">
  <div class="ov-header">
    <h2><span id="ov-name">Agent</span> <span id="ov-status" class="pip-badge badge-idle" style="font-size:10px"></span></h2>
    <button class="ov-close" onclick="closeOverlay()">Close</button>
  </div>
  <div class="ov-tabs">
    <div class="ov-tab active" onclick="switchOvTab('ov-timeline',this)">Timeline</div>
    <div class="ov-tab" onclick="switchOvTab('ov-llm',this)">Prompts &amp; Answers</div>
    <div class="ov-tab" onclick="switchOvTab('ov-tools',this)">Tools</div>
    <div class="ov-tab" onclick="switchOvTab('ov-comms',this)">Comms</div>
  </div>
  <div class="ov-body">
    <div id="ov-timeline" class="ov-sec active"></div>
    <div id="ov-llm" class="ov-sec"></div>
    <div id="ov-tools" class="ov-sec"></div>
    <div id="ov-comms" class="ov-sec"></div>
  </div>
</div>

<script>
/* ═══════════════════════════════════════════════════════════════
   STATE
   ═══════════════════════════════════════════════════════════════ */
var AGENT_COLORS={
  analyste:'#a87cf0',architecte:'#40d4cf',devops:'#22c997',
  developpeur:'#f0a955',qa:'#e85d5d',reviewer:'#5ba4f5',spawner:'#e879a8',deployeur:'#f06292'
};
var AGENT_ICONS={
  analyste:'\u{1F50D}',architecte:'\u{1F3D7}',devops:'\u{2699}',
  developpeur:'\u{1F4BB}',qa:'\u{2705}',reviewer:'\u{1F9EA}',spawner:'\u{26A1}',deployeur:'\u{1F680}'
};
var agents={};
var selectedAgent=null;
var autoScroll=true;
var allLogs=[];
var counters={events:0,errors:0,files:0,tools:0,comms:0,llm:0};
var eventSource=null;
var swarmRunning=false;
var renderCache={
  agents:'',
  agentOptions:'',
  blackboard:'',
  inboxes:'',
  files:'',
  overlayTimeline:'',
  overlayTools:'',
  overlayLlm:'',
  overlayComms:''
};

/* helpers */
function esc(s){var d=document.createElement('div');d.textContent=String(s||'');return d.innerHTML;}
function trunc(s,n){s=String(s||'');return s.length>n?s.substring(0,n)+'\u2026':s;}
function ts(t){var d=new Date(t*1000);return d.toLocaleTimeString('en-GB',{hour:'2-digit',minute:'2-digit',second:'2-digit'});}
function color(a){return AGENT_COLORS[a]||'#e84040';}

/* ═══════════════════════════════════════════════════════════════
   AGENTS PANEL
   ═══════════════════════════════════════════════════════════════ */
function renderAgents(){
  var el=document.getElementById('agents-list'),html='';
  var names=Object.keys(agents);
  document.getElementById('s-agents').textContent=names.length;
  var sel=document.getElementById('f-agent'),cur=sel.value;
  var opts='<option value="all">All</option>';
  for(var i=0;i<names.length;i++) opts+='<option value="'+esc(names[i])+'">'+esc(names[i])+'</option>';
  if(renderCache.agentOptions!==opts){
    sel.innerHTML=opts;
    renderCache.agentOptions=opts;
  }
  sel.value=cur;

  for(var i=0;i<names.length;i++){
    var n=names[i],a=agents[n];
    var st=a.status||'idle';
    var c=color(n);
    var selCls=selectedAgent===n?' selected':'';
    html+='<div class="agent-card'+selCls+'" onclick="openOverlay(\''+esc(n)+'\')">'
      +'<span class="adot adot-'+st+'"></span>'
      +'<span class="aname">'+(AGENT_ICONS[n]||'\u{1F916}')+' '+esc(n)+'</span>'
      +'<div class="astatus" style="color:'+(st==='working'?'var(--accent)':st==='done'?'var(--green)':st==='error'?'var(--red)':st==='waiting'?'var(--orange)':st==='blocked'?'var(--red)':'var(--text3)')+'">'+esc(st.toUpperCase())+'</div>'
      +'<div class="acounts"><span>Evts:'+(a.events||0)+'</span><span>Tools:'+(a.tools||0)+'</span><span>Msgs:'+(a.messages||0)+'</span></div>'
      +'</div>';
  }
  html=html||'<div class="empty">Waiting for agents...</div>';
  if(renderCache.agents!==html){
    el.innerHTML=html;
    renderCache.agents=html;
  }
}

/* ═══════════════════════════════════════════════════════════════
   LOG TABLE
   ═══════════════════════════════════════════════════════════════ */
function applyFilters(){
  var body=document.getElementById('log-body');
  body.innerHTML='';
  var frag=document.createDocumentFragment();
  var fA=document.getElementById('f-agent').value;
  var fT=document.getElementById('f-text').value.toLowerCase();
  for(var i=0;i<allLogs.length;i++){
    var e=allLogs[i];
    if(fA!=='all'&&e.agent!==fA) continue;
    if(fT&&e.message.toLowerCase().indexOf(fT)<0&&e.agent.toLowerCase().indexOf(fT)<0) continue;
    frag.appendChild(_buildLogTR(e));
  }
  body.appendChild(frag);
  while(body.children.length>1500) body.removeChild(body.firstChild);
}
function toggleAuto(){autoScroll=!autoScroll;document.getElementById('f-auto').classList.toggle('on',autoScroll);}

/* ═══════════════════════════════════════════════════════════════
   TAB SWITCHING
   ═══════════════════════════════════════════════════════════════ */
function switchTab(id,el){
  document.querySelectorAll('.mtab').forEach(function(t){t.classList.remove('active');});
  document.querySelectorAll('.tpage').forEach(function(p){p.classList.remove('active');});
  el.classList.add('active');
  document.getElementById('p-'+id).classList.add('active');
}
function switchRight(id,el){
  document.querySelectorAll('.rtab').forEach(function(t){t.classList.remove('active');});
  document.querySelectorAll('.rpage').forEach(function(p){p.classList.remove('active');});
  el.classList.add('active');
  document.getElementById(id).classList.add('active');
}
function switchOvTab(id,el){
  document.querySelectorAll('.ov-tab').forEach(function(t){t.classList.remove('active');});
  document.querySelectorAll('.ov-sec').forEach(function(s){s.classList.remove('active');});
  el.classList.add('active');
  document.getElementById(id).classList.add('active');
}

/* ═══════════════════════════════════════════════════════════════
   CENTER TAB BUILDERS
   ═══════════════════════════════════════════════════════════════ */
function addToolEntry(agent,phase,message,timestamp){
  counters.tools++;
  document.getElementById('tc-tools').textContent=counters.tools;
  var el=document.getElementById('tools-list');
  var div=document.createElement('div');
  div.className='dcard br-cyan';
  div.innerHTML='<div class="dcard-h"><div class="dl">'
    +'<span class="dlbl" style="color:var(--cyan)">\u{1F527} '+esc(phase)+'</span>'
    +'<span class="dm">'+esc(agent)+'</span></div>'
    +'<span class="dt">'+ts(timestamp||Date.now()/1000)+'</span></div>'
    +'<div class="dcard-b">'+esc(message)+'</div>';
  el.appendChild(div);
  while(el.children.length>300) el.removeChild(el.firstChild);
}

function addCommEntry(agent,message,timestamp){
  counters.comms++;
  document.getElementById('tc-comms').textContent=counters.comms;
  var el=document.getElementById('comms-list');
  var div=document.createElement('div');
  div.className='dcard br-accent';
  div.innerHTML='<div class="dcard-h"><div class="dl">'
    +'<span class="dlbl" style="color:var(--accent)">\u{1F4AC} Message</span>'
    +'<span class="dm">'+esc(agent)+'</span></div>'
    +'<span class="dt">'+ts(timestamp||Date.now()/1000)+'</span></div>'
    +'<div class="dcard-b">'+esc(message)+'</div>';
  el.appendChild(div);
  while(el.children.length>200) el.removeChild(el.firstChild);
}

function addErrorEntry(agent,message,timestamp){
  counters.errors++;
  document.getElementById('tc-errors').textContent=counters.errors;
  document.getElementById('s-errors').textContent=counters.errors;
  var el=document.getElementById('errors-list');
  var div=document.createElement('div');
  div.className='dcard br-red';
  div.innerHTML='<div class="dcard-h"><div class="dl">'
    +'<span class="dlbl" style="color:var(--red)">\u26A0 ERROR</span>'
    +'<span class="dm">'+esc(agent)+'</span></div>'
    +'<span class="dt">'+ts(timestamp||Date.now()/1000)+'</span></div>'
    +'<div class="dcard-b" style="color:var(--red)">'+esc(message)+'</div>';
  el.appendChild(div);
}

function addLLMEntry(agent,phase,message,timestamp){
  counters.llm++;
  document.getElementById('tc-llm').textContent=counters.llm;
  var el=document.getElementById('llm-list');
  var div=document.createElement('div');
  var isPrompt=phase==='PROMPT';
  var isThink=phase==='THINK';
  div.className='dcard '+(isPrompt?'br-pink':isThink?'br-orange':'br-cyan');
  var icon=isPrompt?'\u{1F4E4} Prompt':isThink?'\u{1F9E0} Thinking':'\u{1F4E5} Response';
  var clr=isPrompt?'var(--pink)':isThink?'var(--orange)':'var(--cyan)';
  div.innerHTML='<div class="dcard-h"><div class="dl">'
    +'<span class="dlbl" style="color:'+clr+'">'+icon+'</span>'
    +'<span class="dm">'+esc(agent)+'</span></div>'
    +'<span class="dt">'+ts(timestamp||Date.now()/1000)+'</span></div>'
    +'<div class="dcard-b">'+esc(message)+'</div>';
  el.appendChild(div);
  while(el.children.length>300) el.removeChild(el.firstChild);
}

/* ═══════════════════════════════════════════════════════════════
   BLACKBOARD
   ═══════════════════════════════════════════════════════════════ */
function renderBlackboard(bb){
  var el=document.getElementById('bb-list');
  var keys=Object.keys(bb);
  document.getElementById('bb-count').textContent=keys.length;
  if(!keys.length){
    if(renderCache.blackboard!=='<div class="empty">Blackboard is empty</div>'){
      el.innerHTML='<div class="empty">Blackboard is empty</div>';
      renderCache.blackboard='<div class="empty">Blackboard is empty</div>';
    }
    return;
  }
  var html='';
  for(var i=0;i<keys.length;i++){
    var k=keys[i],d=bb[k];
    var author=d.author||'system';
    var val=typeof d==='string'?d:(d.value||JSON.stringify(d));
    var t=d.timestamp?new Date(d.timestamp).toLocaleTimeString():'';
    html+='<div class="bb-card">'
      +'<div style="display:flex;justify-content:space-between;align-items:center;">'
      +'<span class="bb-key">'+esc(k)+'</span>'
      +'<span class="bb-author" style="color:'+color(author)+'">'+esc(author)+'</span>'
      +'</div>'
      +'<div class="bb-val">'+esc(val)+'</div>'
      +(t?'<div class="bb-time">'+esc(t)+'</div>':'')
      +'</div>';
  }
  if(renderCache.blackboard!==html){
    el.innerHTML=html;
    renderCache.blackboard=html;
  }
}

/* ═══════════════════════════════════════════════════════════════
   INBOXES
   ═══════════════════════════════════════════════════════════════ */
function renderInboxes(inboxes){
  var el=document.getElementById('inbox-list');
  var html='';var any=false;
  for(var agent in inboxes){
    var msgs=inboxes[agent];
    if(!msgs||!msgs.length) continue;
    any=true;
    var c=color(agent);
    html+='<div class="inbox-group">'
      +'<div class="inbox-hdr" onclick="this.classList.toggle(\'open\');this.nextElementSibling.classList.toggle(\'open\');">'
      +'<span style="color:'+c+'">'+(AGENT_ICONS[agent]||'')+' '+esc(agent)+'</span>'
      +'<span class="inbox-cnt">'+msgs.length+'</span>'
      +'</div><div class="inbox-body">';
    for(var j=0;j<msgs.length;j++){
      var m=msgs[j];
      var sender=m.sender||m.from||'?';
      var body=m.content||m.message||JSON.stringify(m);
      var mt=m.timestamp?new Date(m.timestamp).toLocaleTimeString():'';
      html+='<div class="inbox-msg">'
        +'<div class="im-from">From: '+esc(sender)+(mt?' | '+esc(mt):'')+'</div>'
        +'<div class="im-body">'+esc(body)+'</div></div>';
    }
    html+='</div></div>';
  }
  html=any?html:'<div class="empty">No messages yet</div>';
  if(renderCache.inboxes!==html){
    el.innerHTML=html;
    renderCache.inboxes=html;
  }
}

/* ═══════════════════════════════════════════════════════════════
   FILES
   ═══════════════════════════════════════════════════════════════ */
function refreshFiles(){
  fetch('/api/deliverables').then(function(r){return r.json();}).then(function(files){
    var el=document.getElementById('files-list');
    if(!files.length){
      if(renderCache.files!=='<div class="empty">No files yet</div>'){
        el.innerHTML='<div class="empty">No files yet</div>';
        renderCache.files='<div class="empty">No files yet</div>';
      }
      return;
    }
    document.getElementById('s-files').textContent=files.length;
    var html=files.map(function(f){
      return '<div class="frow"><a href="/api/deliverables/'+encodeURIComponent(f.path)+'" target="_blank">'+esc(f.path)+'</a><span class="fsz">'+(f.size/1024).toFixed(1)+'KB</span></div>';
    }).join('');
    if(renderCache.files!==html){
      el.innerHTML=html;
      renderCache.files=html;
    }
  }).catch(function(){});
}

/* ═══════════════════════════════════════════════════════════════
   OVERLAY
   ═══════════════════════════════════════════════════════════════ */
function openOverlay(name){
  selectedAgent=name;
  renderAgents();
  document.getElementById('ov-name').textContent=(AGENT_ICONS[name]||'')+' '+name;
  document.getElementById('overlay').classList.add('open');
  renderOverlayData(name);
}
function closeOverlay(){
  selectedAgent=null;
  renderAgents();
  document.getElementById('overlay').classList.remove('open');
}

function renderOverlayData(name){
  var info=agents[name]||{};
  var st=document.getElementById('ov-status');
  st.textContent=(info.status||'idle');
  st.className='pip-badge badge-'+(info.status||'idle');

  /* gather events for this agent from allLogs */
  var evts=[];
  for(var i=0;i<allLogs.length;i++){
    if(allLogs[i].agent===name) evts.push(allLogs[i]);
  }
  evts=evts.slice(-200);

  /* timeline */
  var tlEl=document.getElementById('ov-timeline');
  if(!evts.length){
    if(renderCache.overlayTimeline!=='<div class="empty">No events yet</div>'){
      tlEl.innerHTML='<div class="empty">No events yet</div>';
      renderCache.overlayTimeline='<div class="empty">No events yet</div>';
    }
  }
  else{
    var html='';
    for(var i=0;i<evts.length;i++){
      var e=evts[i];
      var phClr=e.phase==='TOOL'?'var(--cyan)':e.phase==='ERROR'?'var(--red)':e.phase==='DONE'?'var(--green)':e.phase==='ACT'?'var(--orange)':e.phase==='MSG'?'var(--accent)':e.phase==='PROMPT'?'var(--pink)':e.phase==='THINK'?'var(--orange)':e.phase==='RESPONSE'?'var(--cyan)':'var(--blue)';
      var bc=e.phase==='TOOL'?'br-cyan':e.phase==='ERROR'?'br-red':e.phase==='DONE'?'br-green':e.phase==='MSG'?'br-accent':e.phase==='PROMPT'?'br-pink':e.phase==='THINK'?'br-orange':e.phase==='RESPONSE'?'br-cyan':'br-blue';
      html+='<div class="dcard '+bc+'"><div class="dcard-h"><div class="dl">'
        +'<span class="dlbl" style="color:'+phClr+'">'+esc(e.phase)+'</span>'
        +'</div><span class="dt">'+ts(e.time)+'</span></div>'
        +'<div class="dcard-b">'+esc(e.message)+'</div></div>';
    }
    if(renderCache.overlayTimeline!==html){
      tlEl.innerHTML=html;
      renderCache.overlayTimeline=html;
    }
  }

  /* tools */
  var toolEl=document.getElementById('ov-tools');
  var toolEvts=evts.filter(function(e){return e.phase==='TOOL';});
  if(!toolEvts.length){
    if(renderCache.overlayTools!=='<div class="empty">No tool calls</div>'){
      toolEl.innerHTML='<div class="empty">No tool calls</div>';
      renderCache.overlayTools='<div class="empty">No tool calls</div>';
    }
  }
  else{
    var html='';
    for(var i=0;i<toolEvts.length;i++){
      var e=toolEvts[i];
      html+='<div class="dcard br-cyan"><div class="dcard-h"><div class="dl">'
        +'<span class="dlbl" style="color:var(--cyan)">\u{1F527} Tool</span>'
        +'</div><span class="dt">'+ts(e.time)+'</span></div>'
        +'<div class="dcard-b">'+esc(e.message)+'</div></div>';
    }
    if(renderCache.overlayTools!==html){
      toolEl.innerHTML=html;
      renderCache.overlayTools=html;
    }
  }

  /* llm prompts & answers */
  var llmEl=document.getElementById('ov-llm');
  var llmEvts=evts.filter(function(e){return e.phase==='PROMPT'||e.phase==='THINK'||e.phase==='RESPONSE';});
  if(!llmEvts.length){
    if(renderCache.overlayLlm!=='<div class="empty">No LLM calls yet</div>'){
      llmEl.innerHTML='<div class="empty">No LLM calls yet</div>';
      renderCache.overlayLlm='<div class="empty">No LLM calls yet</div>';
    }
  }
  else{
    var html='';
    for(var i=0;i<llmEvts.length;i++){
      var e=llmEvts[i];
      var isP=e.phase==='PROMPT';
      var isT=e.phase==='THINK';
      var bc=isP?'br-pink':isT?'br-orange':'br-cyan';
      var clr=isP?'var(--pink)':isT?'var(--orange)':'var(--cyan)';
      var icon=isP?'\u{1F4E4} Prompt':isT?'\u{1F9E0} Thinking':'\u{1F4E5} Response';
      html+='<div class="dcard '+bc+'"><div class="dcard-h"><div class="dl">'
        +'<span class="dlbl" style="color:'+clr+'">'+icon+'</span>'
        +'</div><span class="dt">'+ts(e.time)+'</span></div>'
        +'<div class="dcard-b">'+esc(e.message)+'</div></div>';
    }
    if(renderCache.overlayLlm!==html){
      llmEl.innerHTML=html;
      renderCache.overlayLlm=html;
    }
  }

  /* comms */
  var commEl=document.getElementById('ov-comms');
  var commEvts=evts.filter(function(e){return e.phase==='MSG'||e.phase==='STEER';});
  if(!commEvts.length){
    if(renderCache.overlayComms!=='<div class="empty">No communications</div>'){
      commEl.innerHTML='<div class="empty">No communications</div>';
      renderCache.overlayComms='<div class="empty">No communications</div>';
    }
  }
  else{
    var html='';
    for(var i=0;i<commEvts.length;i++){
      var e=commEvts[i];
      html+='<div class="dcard br-accent"><div class="dcard-h"><div class="dl">'
        +'<span class="dlbl" style="color:var(--accent)">\u{1F4AC} '+esc(e.phase)+'</span>'
        +'</div><span class="dt">'+ts(e.time)+'</span></div>'
        +'<div class="dcard-b">'+esc(e.message)+'</div></div>';
    }
    if(renderCache.overlayComms!==html){
      commEl.innerHTML=html;
      renderCache.overlayComms=html;
    }
  }
}
/* overlay refresh is now handled by the batched throttle timer */

/* ═══════════════════════════════════════════════════════════════
   EVENT BATCHING  (requestAnimationFrame-based)
   All SSE events are queued and applied in batches each frame,
   preventing DOM thrashing under high event throughput.
   ═══════════════════════════════════════════════════════════════ */
var _eventQueue = [];
var _rafScheduled = false;
var _dirtyAgents = false;
var _dirtyBlackboard = null;
var _dirtyInboxes = null;
var _lastAgentRender = 0;
var THROTTLE_MS = 250;

function _scheduleFlush(){
  if(!_rafScheduled){
    _rafScheduled = true;
    requestAnimationFrame(_flushBatch);
  }
}

function _flushBatch(){
  _rafScheduled = false;
  var batch = _eventQueue;
  _eventQueue = [];
  if(!batch.length) return;

  var agentsDirty = false;
  var needLogScroll = false;
  var logFrag = document.createDocumentFragment();
  var fA = document.getElementById('f-agent').value;
  var fT = document.getElementById('f-text').value.toLowerCase();

  for(var i = 0; i < batch.length; i++){
    var d = batch[i];

    if(d.type === 'event'){
      var agent=d.agent, phase=d.phase, msg=d.message, t=d.timestamp;
      if(!agents[agent]) agents[agent]={name:agent,status:'idle',events:0,tools:0,messages:0};
      var a=agents[agent];
      a.events=(a.events||0)+1;
      if(phase==='ERROR') a.status='error';
      if(phase==='TOOL') a.tools=(a.tools||0)+1;
      if(phase==='MSG') a.messages=(a.messages||0)+1;
      if(phase==='PROMPT'||phase==='THINK'||phase==='RESPONSE') a.llm=(a.llm||0)+1;
      agentsDirty = true;

      /* build log entry */
      var entry={agent:agent,phase:phase,message:msg,time:t||Date.now()/1000};
      allLogs.push(entry);
      counters.events++;

      /* filtered row into fragment */
      if((fA==='all'||agent===fA) && (!fT||msg.toLowerCase().indexOf(fT)>=0||agent.toLowerCase().indexOf(fT)>=0)){
        logFrag.appendChild(_buildLogTR(entry));
        needLogScroll = true;
      }

      if(phase==='TOOL') addToolEntry(agent,phase,msg,t);
      if(phase==='MSG'||phase==='STEER') addCommEntry(agent,msg,t);
      if(phase==='ERROR') addErrorEntry(agent,msg,t);
      if(phase==='PROMPT'||phase==='THINK'||phase==='RESPONSE') addLLMEntry(agent,phase,msg,t);

      if(!swarmRunning){
        swarmRunning=true;
        document.getElementById('main-input').placeholder='Send remark to agents... (Shift+Enter for newline)';
        document.getElementById('send-label').textContent='Send';
        document.getElementById('upload-btn').style.display='none';
        var un=document.getElementById('upload-name');if(un)un.style.display='none';
      }
      var badge=document.getElementById('pip-badge');
      if(badge.classList.contains('badge-idle')){badge.className='pip-badge badge-running';badge.textContent='RUNNING';}
    }
    else if(d.type==='state_sync'){
      _dirtyBlackboard = d.blackboard||{};
      _dirtyInboxes = d.inboxes||{};
    }
    else if(d.type==='agent_status'){
      var ag=d.agent, st=d.status;
      if(!agents[ag]) agents[ag]={name:ag,status:st,events:0,tools:0,messages:0};
      else agents[ag].status=st;
      agentsDirty = true;
    }
    else if(d.type==='pipeline_status'){
      var badge=document.getElementById('pip-badge');
      if(d.status==='running'){badge.className='pip-badge badge-running';badge.textContent='RUNNING';}
      else if(d.status==='done'){badge.className='pip-badge badge-done';badge.textContent='COMPLETE';}
      else if(d.status==='error'){badge.className='pip-badge badge-error';badge.textContent='ERROR';}
      else{badge.className='pip-badge badge-idle';badge.textContent='IDLE';}
    }
  }

  /* --- Apply batched DOM mutations --- */

  /* Log table: append fragment in one shot */
  if(logFrag.childNodes.length){
    var body = document.getElementById('log-body');
    body.appendChild(logFrag);
    while(body.children.length > 1500) body.removeChild(body.firstChild);
    if(allLogs.length > 4000) allLogs = allLogs.slice(-3000);
    document.getElementById('s-events').textContent = counters.events;
    document.getElementById('tc-log').textContent = allLogs.length;
    if(autoScroll && needLogScroll) document.getElementById('p-full-log').scrollTop = 999999;
  }

  /* Agents panel: throttle to THROTTLE_MS */
  if(agentsDirty){
    var now = Date.now();
    if(now - _lastAgentRender >= THROTTLE_MS){
      renderAgents();
      _lastAgentRender = now;
      _dirtyAgents = false;
    } else {
      _dirtyAgents = true;
    }
  }

  /* Blackboard & inboxes: apply latest snapshot */
  if(_dirtyBlackboard !== null){
    renderBlackboard(_dirtyBlackboard);
    _dirtyBlackboard = null;
  }
  if(_dirtyInboxes !== null){
    renderInboxes(_dirtyInboxes);
    _dirtyInboxes = null;
  }
}

/* Build a single <tr> without inserting it (for DocumentFragment) */
function _buildLogTR(e){
  var tr = document.createElement('tr');
  var c = color(e.agent);
  var phClr = 'var(--blue)';
  if(e.phase==='TOOL') phClr='var(--cyan)';
  else if(e.phase==='ERROR') phClr='var(--red)';
  else if(e.phase==='DONE') phClr='var(--green)';
  else if(e.phase==='ACT') phClr='var(--orange)';
  else if(e.phase==='PLAN') phClr='var(--pink)';
  else if(e.phase==='MSG') phClr='var(--accent)';
  else if(e.phase==='PROMPT') phClr='var(--pink)';
  else if(e.phase==='THINK') phClr='var(--orange)';
  else if(e.phase==='RESPONSE') phClr='var(--cyan)';
  tr.innerHTML='<td class="ts">'+ts(e.time)+'</td>'
    +'<td><span class="tag" style="background:'+c+'18;color:'+c+'">'+esc(e.agent)+'</span></td>'
    +'<td><span class="et" style="background:'+phClr+'20;color:'+phClr+'">'+esc(e.phase)+'</span></td>'
    +'<td style="color:var(--text)">'+esc(e.message)+'</td>';
  return tr;
}

/* Catch-up timer for throttled renders */
setInterval(function(){
  if(_dirtyAgents){renderAgents();_dirtyAgents=false;_lastAgentRender=Date.now();}
  if(selectedAgent) renderOverlayData(selectedAgent);
}, THROTTLE_MS * 2);

/* ═══════════════════════════════════════════════════════════════
   SSE CONNECT  (browser-native reconnection with Last-Event-ID)
   ═══════════════════════════════════════════════════════════════ */
function connectSSE(){
  if(eventSource) eventSource.close();
  eventSource = new EventSource('/api/events');
  eventSource.onmessage = function(e){
    var d;
    try{ d = JSON.parse(e.data); }catch(ex){ return; }
    _eventQueue.push(d);
    _scheduleFlush();
  };
  eventSource.onerror = function(){
    /* EventSource.CLOSED === 2 means server rejected; must reconnect manually.
       Otherwise the browser auto-reconnects and sends Last-Event-ID header
       so the server replays missed events. */
    if(eventSource.readyState === EventSource.CLOSED){
      setTimeout(connectSSE, 5000);
    }
  };
}

/* ═══════════════════════════════════════════════════════════════
   BOTTOM BAR: START / STEER / UPLOAD
   ═══════════════════════════════════════════════════════════════ */
function autoGrow(el){el.style.height='auto';el.style.height=Math.min(el.scrollHeight,120)+'px';}

function handleFileUpload(input){
  if(!input.files||!input.files.length) return;
  var file=input.files[0];
  var fd=new FormData();
  fd.append('file',file);
  var label=document.getElementById('upload-name');
  label.textContent='Uploading...';label.style.display='inline';
  document.getElementById('upload-btn').classList.add('has-file');
  fetch('/api/upload',{method:'POST',body:fd}).then(function(r){return r.json();}).then(function(d){
    if(d.ok){label.textContent=file.name;}
    else{label.textContent='Failed';document.getElementById('upload-btn').classList.remove('has-file');}
  }).catch(function(){label.textContent='Error';document.getElementById('upload-btn').classList.remove('has-file');});
}

function handleSubmit(){
  var input=document.getElementById('main-input');
  var text=input.value.trim();
  if(!text) return;
  var btn=document.getElementById('send-btn');
  btn.disabled=true;

  if(!swarmRunning){
    /* --- LAUNCH MODE --- */
    fetch('/api/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({specs:text})})
      .then(function(r){return r.json();})
      .then(function(d){
        if(d.ok){
          input.value='';input.style.height='auto';
        } else { alert(d.error||'Failed to start'); }
        btn.disabled=false;
      }).catch(function(e){alert('Network error: '+e);btn.disabled=false;});
  } else {
    /* --- STEERING MODE --- */
    fetch('/api/steer',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:text})})
      .then(function(r){return r.json();})
      .then(function(){
        input.value='';input.style.height='auto';
        btn.disabled=false;
      }).catch(function(){btn.disabled=false;});
  }
}

document.addEventListener('keydown',function(e){if(e.key==='Escape') closeOverlay();});

/* ═══════════════════════════════════════════════════════════════
   STARTUP
   ═══════════════════════════════════════════════════════════════ */
connectSSE();

/* Load initial agent/state data for pages opened mid-run */
fetch('/api/agents').then(function(r){return r.json();}).then(function(a){
  for(var i=0;i<a.length;i++) agents[a[i].name]=a[i];
  renderAgents();
}).catch(function(){});

/* History replay in non-blocking chunks (50 events per frame) */
fetch('/api/history?last_n=10000').then(function(r){return r.json();}).then(function(events){
  var idx = 0;
  var CHUNK = 50;
  function replayChunk(){
    var end = Math.min(idx + CHUNK, events.length);
    for(; idx < end; idx++){
      var d = events[idx];
      _eventQueue.push(d);
    }
    _scheduleFlush();
    if(idx < events.length) requestAnimationFrame(replayChunk);
  }
  if(events.length) requestAnimationFrame(replayChunk);
}).catch(function(){});

fetch('/api/blackboard').then(function(r){return r.json();}).then(function(bb){renderBlackboard(bb);}).catch(function(){});
fetch('/api/inboxes').then(function(r){return r.json();}).then(function(ib){renderInboxes(ib);}).catch(function(){});
refreshFiles();
/* File list poll (no SSE channel for file changes) */
setInterval(refreshFiles, 10000);
</script>
</body>
</html>
"""
