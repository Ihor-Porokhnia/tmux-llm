import asyncio
import json
import os
import re
import subprocess
import time
import traceback
import sys
from pathlib import Path
from textwrap import dedent

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from loguru import logger as log
from openai import OpenAI
from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit, VSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame, TextArea

DEFAULT_FILTER_PATTERNS = [
    r"\bsk-[A-Za-z0-9]{16,}\b",
    r"\bsk-proj-[A-Za-z0-9]{16,}\b",
    r"(?i)(OPENAI_API_KEY\s*=\s*)([^\s\'\"]+)",
    r"(?i)(Authorization:\s*Bearer\s+)([^\s]+)",
]

DEFAULT_INSTRUCTIONS = dedent(
    """
    You are a DevOps/SRE terminal assistant helping diagnose and resolve a wide range of System Administrator and DevOps problems.
    Output EXACTLY two blocks in this order with these headers on their own lines: HUMAN: then BASH:.
    The HUMAN block is for the human operator only, MUST be written in Russian, and MUST NOT be pasted into the terminal.
    The BASH block MUST be safe-to-paste and executable in a bash shell as-is: ONLY bash code and shell commands, NO Markdown, NO explanations, NO numbering, NO prompt symbols, NO surrounding backticks.
    Work ONE STEP AT A TIME: each response must contain a single next action consisting of either (a) minimal diagnostic commands to collect information needed for the next decision, or (b) one minimal corrective change with commands to apply it.
    Diagnostic command execution is allowed and encouraged.
    If a helper script must be written to a file from bash (via heredoc or echo), then immediately after writing the script inside the BASH block you MUST include, in subsequent lines of the same BASH block: (1) a command that makes the script executable, (2) a command that runs the script, and (3) ensure that the script itself prints its result to STDOUT so that its execution output is shown on the console, (4) remove the script file right after execution.
    If absolutely no commands are needed, leave the BASH block empty.
    Do not provide multiple alternative branches in one response; choose the single most likely next step.
    Commands in BASH must not emit user-facing reports by default.
    When a command may produce excessively large output, it is allowed to apply filtering, truncation, or structured marking to keep the output within reasonable bounds for model processing.
    Formatting or marking of truncated/filtered output is permitted to preserve structure.
    BASH may include creating files and executing them, but must remain runnable without manual editing unless explicitly unavoidable; if unavoidable, mark assumptions in HUMAN and keep BASH runnable.
    """
).strip()

DEFAULT_LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
DEFAULT_LOGGING_CONFIG = {
    "level": "INFO",
    "stderr_level": "WARNING",
    "rotation": "5 MB",
    "retention": 10,
    "error_rotation": "2 MB",
    "error_retention": 15,
    "format": DEFAULT_LOG_FORMAT,
    "stderr": True,
}


def sh(args, input_text=None, check=True):
    p = subprocess.run(args, input=input_text, text=True, capture_output=True)
    if check and p.returncode != 0:
        raise RuntimeError(
            f"cmd failed rc={p.returncode} cmd={' '.join(args)}\nstdout:\n{p.stdout}\nstderr:\n{p.stderr}"
        )
    return p.stdout


def strip_ansi(s):
    return re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", s)


def write_file(p, s):
    Path(p).write_text(s, encoding="utf-8")


def read_file(p):
    try:
        return Path(p).read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def ensure_empty_file(p):
    try:
        pp = Path(p)
        pp.parent.mkdir(parents=True, exist_ok=True)
        if pp.exists() and pp.stat().st_size > 0:
            pp.write_text("", encoding="utf-8")
        if not pp.exists():
            pp.write_text("", encoding="utf-8")
    except Exception:
        try:
            Path(p).write_text("", encoding="utf-8")
        except Exception:
            pass


def resolve_paths(app_name="tmux-llm"):
    base = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local/share"))) / app_name
    run_dir = base / "run"
    hist_dir = base / "history"
    logs_dir = base / "logs"
    tmp_dir = base / "tmp"
    run_dir.mkdir(parents=True, exist_ok=True)
    hist_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return {"base": base, "run": run_dir, "hist": hist_dir, "logs": logs_dir, "tmp": tmp_dir}


def open_log(paths, logging_cfg=None):
    log_path = paths["logs"] / "tmux-llm.log"
    err_path = paths["logs"] / "tmux-llm.error.log"
    cfg = {**DEFAULT_LOGGING_CONFIG, **(logging_cfg or {})}
    fmt = cfg.get("format", DEFAULT_LOG_FORMAT)
    log.remove()
    log.add(
        log_path,
        rotation=cfg.get("rotation", DEFAULT_LOGGING_CONFIG["rotation"]),
        retention=cfg.get("retention", DEFAULT_LOGGING_CONFIG["retention"]),
        enqueue=True,
        backtrace=True,
        diagnose=False,
        level=cfg.get("level", DEFAULT_LOGGING_CONFIG["level"]),
        encoding="utf-8",
        format=fmt,
    )
    log.add(
        err_path,
        rotation=cfg.get("error_rotation", DEFAULT_LOGGING_CONFIG["error_rotation"]),
        retention=cfg.get("error_retention", DEFAULT_LOGGING_CONFIG["error_retention"]),
        enqueue=True,
        backtrace=True,
        diagnose=False,
        level="ERROR",
        encoding="utf-8",
        format=fmt,
    )
    if cfg.get("stderr", True):
        log.add(
            sys.stderr,
            level=cfg.get("stderr_level", DEFAULT_LOGGING_CONFIG["stderr_level"]),
            backtrace=False,
            diagnose=False,
            format=fmt,
        )
    bound_logger = log.bind(component="tmux-llm")

    def log_exc(prefix, e):
        bound_logger.opt(exception=e).error("{} {}", prefix, e)

    return bound_logger, log_path, log_exc


def load_config():
    cfg_path = _config_path()
    cfg_raw = _read_config(cfg_path)
    if cfg_raw is None:
        cfg_raw = {"filters": DEFAULT_FILTER_PATTERNS, "instructions": DEFAULT_INSTRUCTIONS, "logging": DEFAULT_LOGGING_CONFIG}
        cfg_path.write_text(_render_config(cfg_raw), encoding="utf-8")
    filters = _normalize_filters(cfg_raw.get("filters"))
    instructions = _normalize_instructions(cfg_raw.get("instructions"))
    logging_cfg = _normalize_logging(cfg_raw.get("logging"))
    return {"filters": filters, "instructions": instructions, "logging": logging_cfg, "path": cfg_path}


def _config_path():
    base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "tmux-llm" / "config"
    cfg_dir = base
    cfg_dir.mkdir(parents=True, exist_ok=True)
    return cfg_dir / "config.toml"


def _normalize_filters(filters):
    if isinstance(filters, list):
        return [str(f) for f in filters if isinstance(f, str) and f.strip()]
    return DEFAULT_FILTER_PATTERNS


def _normalize_instructions(instructions):
    if isinstance(instructions, str) and instructions.strip():
        return instructions
    return DEFAULT_INSTRUCTIONS


def _normalize_logging(logging_cfg):
    if not isinstance(logging_cfg, dict):
        return DEFAULT_LOGGING_CONFIG
    cfg = {**DEFAULT_LOGGING_CONFIG}
    for k, v in logging_cfg.items():
        if v is None:
            continue
        cfg[k] = v
    return cfg


def _read_config(cfg_path):
    try:
        if cfg_path.exists():
            return tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return {"filters": DEFAULT_FILTER_PATTERNS, "instructions": DEFAULT_INSTRUCTIONS, "logging": DEFAULT_LOGGING_CONFIG}
    return None


def _render_config(cfg):
    filters = _normalize_filters(cfg.get("filters"))
    instructions = _normalize_instructions(cfg.get("instructions")).strip("\n")
    logging_cfg = _normalize_logging(cfg.get("logging"))
    filters_toml = ",\n".join([f"  {json.dumps(f, ensure_ascii=False)}" for f in filters])
    retention = logging_cfg.get("retention", DEFAULT_LOGGING_CONFIG["retention"])
    error_retention = logging_cfg.get("error_retention", DEFAULT_LOGGING_CONFIG["error_retention"])
    retention_line = f"retention = {retention}\n" if isinstance(retention, (int, float)) else f'retention = "{retention}"\n'
    error_retention_line = (
        f"error_retention = {error_retention}\n"
        if isinstance(error_retention, (int, float))
        else f'error_retention = "{error_retention}"\n'
    )
    return (
        "# tmux-llm configuration\n"
        "# Edit values and restart tmux-llm to apply changes.\n"
        "filters = [\n"
        f"{filters_toml}\n"
        "]\n\n"
        f'instructions = """\n{instructions}\n"""\n\n'
        "[logging]\n"
        f'level = "{logging_cfg.get("level", DEFAULT_LOGGING_CONFIG["level"])}"\n'
        f'stderr_level = "{logging_cfg.get("stderr_level", DEFAULT_LOGGING_CONFIG["stderr_level"])}"\n'
        f'stderr = {"true" if logging_cfg.get("stderr", DEFAULT_LOGGING_CONFIG["stderr"]) else "false"}\n'
        f'rotation = "{logging_cfg.get("rotation", DEFAULT_LOGGING_CONFIG["rotation"])}"\n'
        f"{retention_line}"
        f'error_rotation = "{logging_cfg.get("error_rotation", DEFAULT_LOGGING_CONFIG["error_rotation"])}"\n'
        f"{error_retention_line}"
        f'format = "{logging_cfg.get("format", DEFAULT_LOG_FORMAT)}"\n'
    )


class TextFilters:
    def __init__(self, api_key="", patterns=None):
        self.api_key = (api_key or "").strip()
        self._filters = [self._redact_sensitive_values]
        self._api_key_patterns = [re.compile(p) for p in (patterns or [])]
        if self.api_key:
            self._api_key_patterns.append(re.compile(re.escape(self.api_key)))

    def add(self, fn):
        self._filters.append(fn)
        return self

    def apply(self, s):
        if not s:
            return s
        out = s
        for fn in self._filters:
            out = fn(out)
        return out

    def apply_obj(self, o):
        if o is None:
            return o
        if isinstance(o, str):
            return self.apply(o)
        if isinstance(o, list):
            return [self.apply_obj(x) for x in o]
        if isinstance(o, dict):
            return {k: self.apply_obj(v) for k, v in o.items()}
        return o

    def _redact_sensitive_values(self, s):
        out = s
        for rx in self._api_key_patterns:
            def _repl(m):
                return (m.group(1) if m.lastindex else "") + "<REDACTED>"
            out = rx.sub(_repl, out)
        return out


def load_history(hist_path, m, filters):
    rows = []
    p = Path(hist_path)
    if p.exists():
        with p.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(filters.apply_obj(json.loads(line)))
                except Exception:
                    continue
    return rows[-m:]


def append_history(hist_path, rec, filters):
    p = Path(hist_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    safe = filters.apply_obj(rec)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(safe, ensure_ascii=False) + "\n")


def build_ctx(hist_rows, filters):
    ctx = []
    for r in hist_rows:
        q = (r.get("question") or "").strip()
        h = (r.get("answer_human") or "").strip()
        b = (r.get("answer_bash") or "").strip()
        if q:
            ctx.append({"role": "user", "content": q})
        if h or b:
            ctx.append({"role": "assistant", "content": ("HUMAN:\n" + h + "\n\nBASH:\n" + b).strip()})
    return filters.apply_obj(ctx)


def call_model_blocking(api_key, model, pane_capture, notes, hist_path, m, paths, filters, instructions):
    run_dir = paths["run"]
    human_path = str(run_dir / "human.txt")
    bash_path = str(run_dir / "bash.sh")
    hist_rows = load_history(hist_path, m, filters)
    ctx = build_ctx(hist_rows, filters)
    cap_s = filters.apply(pane_capture)
    notes_s = filters.apply(notes)
    user_in = "TERMINAL OUTPUT:\n" + cap_s + "\n\nUSER NOTES:\n" + (notes_s if notes_s else "(none)") + "\n"
    client = OpenAI(api_key=api_key)
    resp = client.responses.create(model=model, instructions=instructions, input=ctx + [{"role": "user", "content": user_in}])
    txt = (resp.output_text or "").replace("\r\n", "\n").strip() + "\n"
    txt = filters.apply(txt)
    mobj = re.search(r"(?s)^\s*HUMAN:\s*(.*?)\n\s*BASH:\s*(.*)\Z", txt)
    human = ""
    bash = ""
    if mobj:
        human = mobj.group(1).strip() + "\n" if mobj.group(1).strip() else ""
        bash = mobj.group(2)
    else:
        human = "(FORMAT ERROR) RAW OUTPUT:\n" + txt
        bash = ""
    write_file(human_path, human)
    write_file(bash_path, bash)
    rec = {"ts": int(time.time()), "model": model, "question": user_in, "answer_human": human, "answer_bash": bash}
    append_history(hist_path, rec, filters)
    return human, bash


def tmux_paste_and_enter(pane_id, bash_text):
    sh(["tmux", "send-keys", "-t", pane_id, "-l", bash_text], check=True)
    if not bash_text.endswith("\n"):
        sh(["tmux", "send-keys", "-t", pane_id, "Enter"], check=True)


def capture_clean(pane_id, n_lines):
    raw = sh(["tmux", "capture-pane", "-t", pane_id, "-pS", f"-{n_lines}"], check=True)
    return strip_ansi(raw)


async def run_tmux_llm():
    config = load_config()
    paths = resolve_paths("tmux-llm")
    logger, log_path, log_exc = open_log(paths, config.get("logging"))
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is missing in environment")
    filters = TextFilters(api_key=api_key, patterns=config["filters"])
    model = os.environ.get("OPENAI_MODEL", "gpt-5-nano")
    pane_id = sh(["tmux", "display-message", "-p", "#{pane_id}"]).strip()
    if not pane_id:
        raise SystemExit("EMPTY PANE_ID")
    run_dir = paths["run"]
    hist_path = str(paths["hist"] / "history.jsonl")
    cap_path = str(run_dir / "capture.txt")
    notes_path = str(run_dir / "notes.txt")
    human_path = str(run_dir / "human.txt")
    bash_path = str(run_dir / "bash.sh")
    ensure_empty_file(notes_path)
    ensure_empty_file(human_path)
    ensure_empty_file(bash_path)
    n_lines = int(os.environ.get("TMUX_LLM_LINES", "400"))
    m_turns = int(os.environ.get("TMUX_LLM_TURNS", "6"))
    logger.info(
        "tmux-llm session starting pane_id={} model={} lines={} turns={}",
        pane_id,
        model,
        n_lines,
        m_turns,
    )
    pane_capture = filters.apply(capture_clean(pane_id, n_lines))
    write_file(cap_path, pane_capture)
    style = Style.from_dict({"frame.border": "ansiblue", "title": "ansicyan bold"})
    kb = KeyBindings()
    header = FormattedTextControl(text=[("class:title", "tmux-llm | Tab/Shift-Tab: focus | F5: send | F6: paste | F10/Esc: quit")])
    status = FormattedTextControl(text=[("", "Ready.")])

    def set_status(s):
        status.text = [("", s)]

    preview = TextArea(text=(" %d lines from %s\n" % (n_lines, pane_id)) + pane_capture, read_only=True, scrollbar=True, wrap_lines=False)
    notes = TextArea(text="", multiline=True, scrollbar=True, wrap_lines=True)
    bash = TextArea(text="", read_only=True, scrollbar=True, wrap_lines=False)
    human = TextArea(text="", read_only=True, scrollbar=True, wrap_lines=True)
    sending = {"busy": False}
    refreshing = {"busy": False}
    w70 = Dimension(weight=7)
    w30 = Dimension(weight=3)
    top = VSplit([Frame(preview, title="Preview", width=w70), Frame(notes, title="Prompt", width=w30)])
    bottom = VSplit([Frame(bash, title="BASH", width=w70), Frame(human, title="HUMAN", width=w30)])
    container = HSplit([Window(header, height=1), top, bottom, Window(status, height=1)])
    app = Application(layout=Layout(container, focused_element=notes), key_bindings=kb, full_screen=True, style=style)

    @kb.add("tab")
    def _(event):
        event.app.layout.focus_next()

    @kb.add("s-tab")
    def _(event):
        event.app.layout.focus_previous()

    async def refresh_preview_multi(delays):
        if refreshing["busy"]:
            return
        refreshing["busy"] = True
        try:
            logger.info("Refreshing preview for pane {} with {} windows", pane_id, len(delays))
            for d in delays:
                await asyncio.sleep(d)
                cap2 = filters.apply(capture_clean(pane_id, n_lines))
                preview.text = ("=== PREVIEW (last %d lines from %s) ===\n" % (n_lines, pane_id)) + cap2
                write_file(cap_path, cap2)
                app.invalidate()
            set_status("Preview refreshed.")
        except Exception as e:
            set_status(f"Preview refresh failed; see {log_path}.")
            log_exc("(REFRESH ERROR)", e)
        finally:
            refreshing["busy"] = False
            app.invalidate()

    async def do_send():
        if sending["busy"]:
            return
        sending["busy"] = True
        notes_text = filters.apply(notes.text)
        write_file(notes_path, notes_text)
        set_status("Sending to model...")
        app.invalidate()
        try:
            loop = asyncio.get_running_loop()
            logger.info("Sending prompt to model={} captured_lines={} notes_length={}", model, n_lines, len(notes_text))
            h, b = await loop.run_in_executor(None, call_model_blocking, api_key, model, pane_capture, notes_text, hist_path, m_turns, paths, filters, config["instructions"])
            human.text = h
            bash.text = b
            notes.text = ""
            write_file(notes_path, "")
            set_status("Received. Press F6 to paste.")
            logger.info("Model response received bash_len={} human_len={}", len(b), len(h))
        except Exception as e:
            et = "(MODEL ERROR)\n" + repr(e) + "\n\n" + traceback.format_exc()
            et = filters.apply(et)
            human.text = et
            bash.text = ""
            write_file(human_path, et)
            write_file(bash_path, "")
            set_status(f"Model error. See HUMAN / {log_path}.")
            log_exc("(MODEL ERROR)", e)
        finally:
            sending["busy"] = False
            app.invalidate()

    @kb.add("f5")
    def _(event):
        asyncio.create_task(do_send())

    @kb.add("f6")
    def _(event):
        bt = bash.text
        if not bt.strip():
            set_status("BASH is empty; nothing to paste.")
            logger.warning("Paste requested with empty BASH output for pane {}", pane_id)
            return
        try:
            tmux_paste_and_enter(pane_id, bt)
            set_status(f"Pasted+Enter into {pane_id}. Refreshing preview...")
            logger.info("Pasted model output into pane {}", pane_id)
            asyncio.create_task(refresh_preview_multi([0.15, 0.6, 1.5, 2.5, 3.5]))
        except Exception as e:
            set_status(f"Paste failed; see {log_path}.")
            log_exc("(PASTE ERROR)", e)

    @kb.add("f10")
    @kb.add("escape")
    def _(event):
        event.app.exit(result=0)

    return await app.run_async()


def main():
    try:
        return asyncio.run(run_tmux_llm())
    except SystemExit:
        raise
    except Exception as e:
        try:
            paths = resolve_paths("tmux-llm")
            try:
                cfg = load_config()
            except Exception:
                cfg = None
            logger, log_path, log_exc = open_log(paths, (cfg or {}).get("logging"))
            log_exc("(FATAL)", e)
        except Exception:
            pass
        raise


__all__ = [
    "append_history",
    "build_ctx",
    "call_model_blocking",
    "capture_clean",
    "ensure_empty_file",
    "load_config",
    "load_history",
    "main",
    "open_log",
    "resolve_paths",
    "run_tmux_llm",
    "strip_ansi",
    "TextFilters",
    "tmux_paste_and_enter",
    "write_file",
]
