#!/usr/bin/env python3
"""GenieX Chat API Server — Flask backend for chat_ui.html and dashboard.html."""

import os
import re
import time
import uuid
import threading
import psutil
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = "ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4"
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
LOG_DIR    = os.path.join(BASE_DIR, "04_logs")
LOG_FILE   = os.path.join(LOG_DIR, "log_performance.txt")

os.makedirs(LOG_DIR, exist_ok=True)

app = Flask(__name__, static_folder=None)
CORS(app)

# ---------------------------------------------------------------------------
# Model state
# ---------------------------------------------------------------------------
_model     = None
_tokenizer = None
_model_ready = threading.Event()
_model_error = None
_infer_lock  = threading.Lock()   # one inference at a time on the NPU


def _load_model():
    global _model, _tokenizer, _model_error
    try:
        from QEfficient import QEFFAutoModelForCausalLM
        from transformers import AutoTokenizer

        _model = QEFFAutoModelForCausalLM.from_pretrained(MODEL_NAME)
        _model.compile(
            num_devices=2,
            num_cores=16,
            prefill_seq_len=128,
            ctx_len=4096,
            batch_size=1,
            mxint8_kv_cache=True,
        )
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception as exc:
        _model_error = str(exc)
    finally:
        _model_ready.set()


threading.Thread(target=_load_model, daemon=True).start()

# ---------------------------------------------------------------------------
# Session store
# ---------------------------------------------------------------------------
_sessions: dict = {}   # {sid: {id, title, history, updated_at}}
_memory:   dict = {}   # global memory: {user_name, ...}
_sess_lock = threading.Lock()


def _session(sid: str) -> dict:
    """Return existing session or create one."""
    if sid not in _sessions:
        _sessions[sid] = {
            "id": sid,
            "title": "Novo chat",
            "history": [],
            "updated_at": _now(),
        }
    return _sessions[sid]


def _now() -> str:
    return datetime.now().strftime("%d/%m/%Y %H:%M")

# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _build_prompt(history: list, message: str, mode: str) -> str:
    system = (
        "Você é GenieX, um assistente inteligente e prestativo. "
        "Responda em português, de forma clara e objetiva."
        if mode == "chat"
        else
        "Você é GenieX, um assistente analítico avançado. "
        "Faça uma análise profunda e detalhada, explorando múltiplos ângulos do problema."
    )
    messages = [{"role": "system", "content": system}]
    for msg in history[-10:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": message})

    if hasattr(_tokenizer, "apply_chat_template"):
        return _tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # Fallback: plain text format
    parts = [f"<|system|>\n{system}\n"]
    for msg in messages[1:]:
        parts.append(f"<|{msg['role']}|>\n{msg['content']}\n")
    parts.append("<|assistant|>\n")
    return "".join(parts)


def _infer(prompt: str, mode: str) -> tuple[str, dict]:
    """Run one inference pass. Returns (reply, metrics)."""
    if not _model_ready.is_set():
        raise RuntimeError("Modelo ainda carregando, aguarde...")
    if _model_error:
        raise RuntimeError(f"Falha ao carregar modelo: {_model_error}")

    with _infer_lock:
        t0 = time.perf_counter()
        raw = _model.generate(
            tokenizer=_tokenizer,
            prompts=[prompt],
            device_id=[0, 1],
        )
        t1 = time.perf_counter()

    # QEfficient may return list or string
    reply = raw[0] if isinstance(raw, (list, tuple)) else str(raw)
    # Strip echoed prompt if present
    if reply.startswith(prompt):
        reply = reply[len(prompt):].strip()

    elapsed   = t1 - t0
    tok_prompt = max(1, len(prompt) // 4)
    tok_comp   = max(1, len(reply)  // 4)
    return reply, {
        "latency": elapsed * 1000,
        "ttft":    elapsed * 150,   # ~15 % of total, rough estimate
        "tps":     tok_comp / elapsed if elapsed > 0 else 0,
        "tok_prompt": tok_prompt,
        "tok_comp":   tok_comp,
    }


def _log(sid: str, mode: str, m: dict, context_chars: int, summary_chars: int = 0):
    proc     = psutil.Process()
    ram_mb   = proc.memory_info().rss / 1024 / 1024
    cpu_pct  = psutil.cpu_percent(interval=None) / (psutil.cpu_count() or 1)
    tok_total = m["tok_prompt"] + m["tok_comp"]
    line = (
        f"time={datetime.now().strftime('%H:%M:%S')} | "
        f"execution_target=npu | npu_status=enabled | mode={mode} | session_id={sid} | "
        f"tps={m['tps']:.2f} | latency={m['latency']:.0f} | ttft={m['ttft']:.0f} | "
        f"tok_total={tok_total} | tok_prompt={m['tok_prompt']} | tok_comp={m['tok_comp']} | "
        f"ram_mb={ram_mb:.1f} | cpu_pct={cpu_pct:.1f} | "
        f"context_chars={context_chars} | summary_chars={summary_chars}"
    )
    with open(LOG_FILE, "a") as fh:
        fh.write(line + "\n")

# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.post("/chat")
def chat():
    data    = request.get_json(force=True)
    message = (data.get("message") or "").strip()
    mode    = data.get("mode", "chat")
    sid     = data.get("session_id", "default")

    if not message:
        return jsonify({"error": "empty message"}), 400

    with _sess_lock:
        sess = _session(sid)
        sess["history"].append({"role": "user", "content": message})
        history_snapshot = list(sess["history"][:-1])  # history before this message

    prompt       = _build_prompt(history_snapshot, message, mode)
    context_chars = len(prompt)

    try:
        reply, metrics = _infer(prompt, mode)
    except RuntimeError as exc:
        reply   = str(exc)
        metrics = {"latency": 0, "ttft": 0, "tps": 0, "tok_prompt": 0, "tok_comp": 0}

    with _sess_lock:
        sess = _session(sid)
        sess["history"].append({"role": "assistant", "content": reply})
        sess["updated_at"] = _now()
        if sess["title"] == "Novo chat" and len(sess["history"]) >= 2:
            sess["title"] = message[:40] + ("..." if len(message) > 40 else "")
        # Persist user name if mentioned
        m = re.search(
            r"(?:me chamo|meu nome é|sou o|sou a)\s+(\w+)", message, re.IGNORECASE
        )
        if m:
            _memory["user_name"] = m.group(1)

    _log(sid, mode, metrics, context_chars)

    return jsonify({"response": reply, "mode": mode})


@app.get("/history")
def history():
    return jsonify({"memory": _memory, "history": []})


@app.get("/sessions")
def list_sessions():
    with _sess_lock:
        items = sorted(_sessions.values(), key=lambda s: s["updated_at"], reverse=True)
        return jsonify({
            "sessions": [
                {"id": s["id"], "title": s["title"], "updated_at": s["updated_at"]}
                for s in items
            ]
        })


@app.post("/session/new")
def new_session():
    sid = str(uuid.uuid4())[:8]
    with _sess_lock:
        _session(sid)
    return jsonify({"session_id": sid})


@app.get("/session/<sid>")
def get_session(sid):
    with _sess_lock:
        sess = _session(sid)
        return jsonify({"title": sess["title"], "history": sess["history"]})


@app.post("/session/<sid>/rename")
def rename_session(sid):
    data  = request.get_json(force=True)
    title = (data.get("title") or "").strip()
    if not title:
        return jsonify({"error": "empty title"}), 400
    with _sess_lock:
        _session(sid)["title"] = title
    return jsonify({"ok": True})


@app.post("/session/<sid>/reset")
def reset_session(sid):
    with _sess_lock:
        if sid in _sessions:
            _sessions[sid]["history"]    = []
            _sessions[sid]["updated_at"] = _now()
    return jsonify({"ok": True})


@app.get("/model-status")
def model_status():
    return jsonify({
        "ready":   _model_ready.is_set() and _model_error is None,
        "loading": not _model_ready.is_set(),
        "error":   _model_error,
    })


# ---------------------------------------------------------------------------
# Static files
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return send_from_directory(BASE_DIR, "chat_ui.html")


@app.get("/dashboard")
def dashboard():
    return send_from_directory(BASE_DIR, "dashboard.html")


@app.get("/04_logs/<path:filename>")
def serve_log(filename):
    return send_from_directory(LOG_DIR, filename)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== GenieX Server iniciando na porta 5000 ===")
    print(f"  Chat UI : http://127.0.0.1:5000/")
    print(f"  Dashboard: http://127.0.0.1:5000/dashboard")
    print(f"  Status modelo: http://127.0.0.1:5000/model-status")
    print("  (modelo carregando em background...)")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
