import asyncio
import difflib
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


def run_command(args, input_text=None, check=True):
    process = subprocess.run(args, input=input_text, text=True, capture_output=True)
    if check and process.returncode != 0:
        raise RuntimeError(
            f"cmd failed rc={process.returncode} cmd={' '.join(args)}\nstdout:\n{process.stdout}\nstderr:\n{process.stderr}"
        )
    return process.stdout


def strip_ansi(text):
    return re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", text)


def compute_cropped_preview(before, after):
    before_lines = (before or "").splitlines()
    after_lines = (after or "").splitlines()
    if not after_lines:
        return ""
    sequence_matcher = difflib.SequenceMatcher(None, before_lines, after_lines)
    new_lines = []
    for tag, i1, i2, j1, j2 in sequence_matcher.get_opcodes():
        if tag in ("insert", "replace"):
            new_lines.extend(after_lines[j1:j2])
    if not new_lines and before_lines != after_lines:
        new_lines = after_lines
    return "\n".join(new_lines).strip("\n")


def strip_prompt_prefix(line):
    if line is None:
        return "", False
    prompt_patterns = [
        re.compile(r"^\s*(?:\[[^\]]+\]\s*)?(?:[^\s@]+@)?[^\s:]+:[^\s]*[$#]\s*"),
        re.compile(r"^\s*[A-Za-z]:[^\n>]*>\s*"),
        re.compile(r"^\s*[#$%>]\s+"),
    ]
    for pattern in prompt_patterns:
        match = pattern.match(line)
        if match:
            return line[match.end():], True
    return line, False


def sanitize_preview(preview_text, bash_text):
    if not preview_text:
        return ""
    commands = [cmd.strip() for cmd in (bash_text or "").splitlines() if cmd.strip()]
    sanitized_lines = []
    for raw_line in preview_text.splitlines():
        line, had_prompt = strip_prompt_prefix(raw_line)
        normalized = line.strip()
        if commands and any(normalized == cmd or normalized.startswith(cmd + " ") for cmd in commands):
            continue
        if had_prompt and not normalized:
            continue
        if not normalized:
            continue
        sanitized_lines.append(line)
    return "\n".join(sanitized_lines).strip("\n")


def normalize_history_row(row, sanitize=False):
    if not isinstance(row, dict) or "bash" not in row:
        return None
    bash = row.get("bash") or ""
    preview = row.get("preview") or ""
    normalized = {
        "ts": row.get("ts"),
        "model": row.get("model") or "",
        "notes": row.get("notes") or "",
        "bash": str(bash) if bash is not None else "",
        "preview": str(preview) if preview is not None else "",
        "answer": str(row.get("answer") or ""),
    }
    if sanitize:
        normalized["preview"] = sanitize_preview(normalized["preview"], normalized["bash"])
    return normalized


def write_file(path, content):
    Path(path).write_text(content, encoding="utf-8")


def read_file(path):
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def ensure_empty_file(path):
    try:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        if path_obj.exists() and path_obj.stat().st_size > 0:
            path_obj.write_text("", encoding="utf-8")
        if not path_obj.exists():
            path_obj.write_text("", encoding="utf-8")
    except Exception:
        try:
            Path(path).write_text("", encoding="utf-8")
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
    config = {**DEFAULT_LOGGING_CONFIG, **(logging_cfg or {})}
    fmt = config.get("format", DEFAULT_LOG_FORMAT)
    log.remove()
    log.add(
        log_path,
        rotation=config.get("rotation", DEFAULT_LOGGING_CONFIG["rotation"]),
        retention=config.get("retention", DEFAULT_LOGGING_CONFIG["retention"]),
        enqueue=True,
        backtrace=True,
        diagnose=False,
        level=config.get("level", DEFAULT_LOGGING_CONFIG["level"]),
        encoding="utf-8",
        format=fmt,
    )
    log.add(
        err_path,
        rotation=config.get("error_rotation", DEFAULT_LOGGING_CONFIG["error_rotation"]),
        retention=config.get("error_retention", DEFAULT_LOGGING_CONFIG["error_retention"]),
        enqueue=True,
        backtrace=True,
        diagnose=False,
        level="ERROR",
        encoding="utf-8",
        format=fmt,
    )
    if config.get("stderr", True):
        log.add(
            sys.stderr,
            level=config.get("stderr_level", DEFAULT_LOGGING_CONFIG["stderr_level"]),
            backtrace=False,
            diagnose=False,
            format=fmt,
        )
    bound_logger = log.bind(component="tmux-llm")

    def log_exc(prefix, e):
        bound_logger.opt(exception=e).error("{} {}", prefix, e)

    return bound_logger, log_path, log_exc


def load_config():
    config_path = _config_path()
    config_raw = _read_config(config_path)
    if config_raw is None:
        config_raw = {"filters": DEFAULT_FILTER_PATTERNS, "instructions": DEFAULT_INSTRUCTIONS, "logging": DEFAULT_LOGGING_CONFIG}
        config_path.write_text(_render_config(config_raw), encoding="utf-8")
    filters = _normalize_filters(config_raw.get("filters"))
    instructions = _normalize_instructions(config_raw.get("instructions"))
    logging_cfg = _normalize_logging(config_raw.get("logging"))
    return {"filters": filters, "instructions": instructions, "logging": logging_cfg, "path": config_path}


def _config_path():
    base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "tmux-llm" / "config"
    config_dir = base
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "config.toml"


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
    config = {**DEFAULT_LOGGING_CONFIG}
    for key, value in logging_cfg.items():
        if value is None:
            continue
        config[key] = value
    return config


def _read_config(config_path):
    try:
        if config_path.exists():
            return tomllib.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {"filters": DEFAULT_FILTER_PATTERNS, "instructions": DEFAULT_INSTRUCTIONS, "logging": DEFAULT_LOGGING_CONFIG}
    return None


def _render_config(config):
    filters = _normalize_filters(config.get("filters"))
    instructions = _normalize_instructions(config.get("instructions")).strip("\n")
    logging_cfg = _normalize_logging(config.get("logging"))
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

    def add(self, filter_function):
        self._filters.append(filter_function)
        return self

    def apply(self, text):
        if not text:
            return text
        out = text
        for filter_function in self._filters:
            out = filter_function(out)
        return out

    def apply_obj(self, obj):
        if obj is None:
            return obj
        if isinstance(obj, str):
            return self.apply(obj)
        if isinstance(obj, list):
            return [self.apply_obj(x) for x in obj]
        if isinstance(obj, dict):
            return {k: self.apply_obj(v) for k, v in obj.items()}
        return obj

    def _redact_sensitive_values(self, text):
        out = text
        for pattern in self._api_key_patterns:
            def replace_match(match):
                return (match.group(1) if match.lastindex else "") + "<REDACTED>"
            out = pattern.sub(replace_match, out)
        return out


def load_history(history_path, max_entries, filters):
    rows = []
    history_file = Path(history_path)
    if history_file.exists():
        with history_file.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed_row = json.loads(line)
                    normalized = normalize_history_row(parsed_row, sanitize=True)
                    if normalized is None:
                        continue
                    rows.append(filters.apply_obj(normalized))
                except Exception:
                    continue
    return rows[-max_entries:]


def append_history(history_path, record, filters):
    history_file = Path(history_path)
    history_file.parent.mkdir(parents=True, exist_ok=True)
    normalized_record = normalize_history_row(record, sanitize=True)
    if normalized_record is None:
        return
    safe_record = filters.apply_obj(normalized_record)
    with history_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(safe_record, ensure_ascii=False) + "\n")


def build_context_messages(init_capture, history_rows, current_notes, filters, line_count=None):
    context_messages = []
    init_text = (init_capture or "").strip()
    if init_text:
        prefix = "INIT TERMINAL CAPTURE"
        if line_count:
            prefix += f" (last {line_count} lines)"
        context_messages.append({"role": "user", "content": f"{prefix}:\n{init_text}"})
    for row in history_rows:
        notes_text = str(row.get("notes") or "").strip()
        bash_text = str(row.get("bash") or "").strip()
        preview_text = str(row.get("preview") or "").strip()
        human_text = str(row.get("answer") or "").strip()
        parts = []
        if notes_text or bash_text or preview_text:
            parts.append("NOTES:\n" + (notes_text if notes_text else "(none)"))
        if bash_text:
            parts.append("BASH:\n" + bash_text)
            parts.append("RESULT:\n" + preview_text)
        elif preview_text:
            parts.append("RESULT:\n" + preview_text)
        block = "\n\n".join(parts).strip()
        if block:
            context_messages.append({"role": "user", "content": block})
        if human_text:
            context_messages.append({"role": "assistant", "content": "HUMAN:\n" + human_text})
    notes_block = (current_notes or "").strip()
    if notes_block:
        context_messages.append({"role": "user", "content": "NOTES:\n" + notes_block})
    return filters.apply_obj(context_messages)


def call_model_blocking(api_key, model, context, paths, filters, instructions):
    run_directory = paths["run"]
    human_path = str(run_directory / "human.txt")
    bash_path = str(run_directory / "bash.sh")
    client = OpenAI(api_key=api_key)
    response = client.responses.create(model=model, instructions=instructions, input=context)
    raw_output = (response.output_text or "").replace("\r\n", "\n").strip() + "\n"
    raw_output = filters.apply(raw_output)
    parsed_sections = re.search(r"(?s)^\s*HUMAN:\s*(.*?)\n\s*BASH:\s*(.*)\Z", raw_output)
    human = ""
    bash = ""
    if parsed_sections:
        human = parsed_sections.group(1).strip() + "\n" if parsed_sections.group(1).strip() else ""
        bash = parsed_sections.group(2)
    else:
        human = "(FORMAT ERROR) RAW OUTPUT:\n" + raw_output
        bash = ""
    write_file(human_path, human)
    write_file(bash_path, bash)
    return human, bash


def tmux_paste_and_enter(pane_id, bash_text):
    run_command(["tmux", "send-keys", "-t", pane_id, "-l", bash_text], check=True)
    if not bash_text.endswith("\n"):
        run_command(["tmux", "send-keys", "-t", pane_id, "Enter"], check=True)


def capture_clean(pane_id, line_count):
    capture_text = run_command(["tmux", "capture-pane", "-t", pane_id, "-pS", f"-{line_count}"], check=True)
    return strip_ansi(capture_text)


async def run_tmux_llm():
    config = load_config()
    paths = resolve_paths("tmux-llm")
    logger, log_path, log_exc = open_log(paths, config.get("logging"))
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is missing in environment")
    filters = TextFilters(api_key=api_key, patterns=config["filters"])
    model = os.environ.get("OPENAI_MODEL", "gpt-5-nano")
    pane_id = run_command(["tmux", "display-message", "-p", "#{pane_id}"]).strip()
    if not pane_id:
        raise SystemExit("EMPTY PANE_ID")
    run_directory = paths["run"]
    history_path = str(paths["hist"] / "history.jsonl")
    capture_path = str(run_directory / "capture.txt")
    notes_path = str(run_directory / "notes.txt")
    human_path = str(run_directory / "human.txt")
    bash_path = str(run_directory / "bash.sh")
    ensure_empty_file(notes_path)
    ensure_empty_file(human_path)
    ensure_empty_file(bash_path)
    capture_line_limit = int(os.environ.get("TMUX_LLM_LINES", "100"))
    max_history_turns = int(os.environ.get("TMUX_LLM_TURNS", "6"))
    logger.info(
        "tmux-llm session starting pane_id={} model={} lines={} turns={}",
        pane_id,
        model,
        capture_line_limit,
        max_history_turns,
    )
    init_capture = filters.apply(capture_clean(pane_id, capture_line_limit))
    write_file(capture_path, init_capture)
    session_state = {
        "init_capture": init_capture,
        "before_capture": init_capture,
        "pending": None,
        "context_baseline": None,
    }
    style = Style.from_dict({"frame.border": "ansiblue", "title": "ansicyan bold"})
    key_bindings = KeyBindings()
    header = FormattedTextControl(
        text=[
            (
                "class:title",
                "tmux-llm | Tab/Shift-Tab: focus | F5: send | F6: paste | F7: reset history | F8: clear term ctx | F10/Esc: quit",
            )
        ]
    )
    status_bar = FormattedTextControl(text=[("", "Ready.")])

    def set_status(message):
        status_bar.text = [("", message)]

    preview_area = TextArea(
        text=(" %d lines from %s\n" % (capture_line_limit, pane_id)) + init_capture, read_only=True, scrollbar=True, wrap_lines=False
    )
    notes_area = TextArea(text="", multiline=True, scrollbar=True, wrap_lines=True)
    bash_area = TextArea(text="", read_only=True, scrollbar=True, wrap_lines=False)
    human_area = TextArea(text="", read_only=True, scrollbar=True, wrap_lines=True)
    send_state = {"busy": False}
    refresh_state = {"busy": False}
    primary_width_weight = Dimension(weight=7)
    secondary_width_weight = Dimension(weight=3)
    top_row = VSplit([Frame(preview_area, title="Preview", width=primary_width_weight), Frame(notes_area, title="Prompt", width=secondary_width_weight)])
    bottom_row = VSplit([Frame(bash_area, title="BASH", width=primary_width_weight), Frame(human_area, title="HUMAN", width=secondary_width_weight)])
    layout_container = HSplit([Window(header, height=1), top_row, bottom_row, Window(status_bar, height=1)])
    application = Application(layout=Layout(layout_container, focused_element=notes_area), key_bindings=key_bindings, full_screen=True, style=style)

    @key_bindings.add("tab")
    def _(event):
        event.app.layout.focus_next()

    @key_bindings.add("s-tab")
    def _(event):
        event.app.layout.focus_previous()

    def compute_init_context(latest_capture):
        baseline = session_state.get("context_baseline")
        if baseline is not None:
            return compute_cropped_preview(baseline, latest_capture)
        return session_state["init_capture"]

    def capture_full_context():
        line_target = capture_line_limit
        try:
            history_size_raw = run_command(["tmux", "display-message", "-p", "#{history_size}"]).strip()
            pane_height_raw = run_command(["tmux", "display-message", "-p", "#{pane_height}"]).strip()
            history_size = int(history_size_raw or "0")
            pane_height = int(pane_height_raw or "0")
            if history_size >= 0 and pane_height >= 0:
                line_target = max(line_target, history_size + pane_height)
        except Exception:
            line_target = max(line_target * 2, line_target + 500)
        capture_text = filters.apply(capture_clean(pane_id, line_target))
        actual_lines = len(capture_text.splitlines())
        new_limit = max(line_target, actual_lines)
        return capture_text, new_limit

    async def clear_history_and_refresh_context():
        nonlocal capture_line_limit
        if send_state["busy"]:
            set_status("Wait for send to finish before clearing history.")
            return
        if refresh_state["busy"]:
            set_status("Preview refresh in progress; try again momentarily.")
            return
        set_status("Clearing history and capturing full terminal...")
        application.invalidate()
        try:
            ensure_empty_file(history_path)
            session_state["pending"] = None
            session_state["context_baseline"] = None
            full_capture, new_limit = await asyncio.to_thread(capture_full_context)
            capture_line_limit = new_limit
            session_state["init_capture"] = full_capture
            session_state["before_capture"] = full_capture
            preview_area.text = ("=== PREVIEW (last %d lines from %s) ===\n" % (capture_line_limit, pane_id)) + full_capture
            write_file(capture_path, full_capture)
            notes_area.text = ""
            bash_area.text = ""
            human_area.text = ""
            set_status(f"History cleared; using {capture_line_limit} lines from terminal.")
            logger.info("History cleared and recaptured terminal capture_len={} new_limit={}", len(full_capture), capture_line_limit)
        except Exception as e:
            set_status(f"History clear failed; see {log_path}.")
            log_exc("(HISTORY CLEAR ERROR)", e)
        finally:
            application.invalidate()

    async def clear_terminal_context_only():
        if send_state["busy"]:
            set_status("Wait for send to finish before clearing terminal context.")
            return
        if refresh_state["busy"]:
            set_status("Preview refresh in progress; try again momentarily.")
            return
        set_status("Clearing terminal context...")
        application.invalidate()
        try:
            baseline_capture = await asyncio.to_thread(lambda: filters.apply(capture_clean(pane_id, capture_line_limit)))
            session_state["context_baseline"] = baseline_capture
            session_state["init_capture"] = ""
            session_state["pending"] = None
            session_state["before_capture"] = baseline_capture
            ensure_empty_file(capture_path)
            preview_area.text = "=== TERMINAL CONTEXT CLEARED ===\nOld terminal output will be ignored for the next context."
            set_status("Terminal context cleared. New context starts now.")
            logger.info("Terminal context cleared baseline_len={} limit={}", len(baseline_capture), capture_line_limit)
        except Exception as e:
            set_status(f"Terminal clear failed; see {log_path}.")
            log_exc("(TERMINAL CLEAR ERROR)", e)
        finally:
            application.invalidate()

    async def refresh_preview_sequence(delays):
        if refresh_state["busy"]:
            return
        refresh_state["busy"] = True
        try:
            logger.info("Refreshing preview for pane {} with {} windows", pane_id, len(delays))
            latest_capture = None
            for delay_seconds in delays:
                await asyncio.sleep(delay_seconds)
                capture_snapshot = filters.apply(capture_clean(pane_id, capture_line_limit))
                latest_capture = capture_snapshot
                preview_area.text = ("=== PREVIEW (last %d lines from %s) ===\n" % (capture_line_limit, pane_id)) + capture_snapshot
                write_file(capture_path, capture_snapshot)
                application.invalidate()
            if latest_capture is not None:
                if session_state["pending"]:
                    cropped_preview = compute_cropped_preview(session_state["pending"].get("before_capture"), latest_capture)
                    record = {
                        "ts": session_state["pending"]["ts"],
                        "model": session_state["pending"]["model"],
                        "notes": session_state["pending"]["notes"],
                        "bash": session_state["pending"]["bash"],
                        "preview": cropped_preview,
                        "answer": session_state["pending"]["human"],
                    }
                    append_history(history_path, record, filters)
                    logger.info("History updated preview_len={} bash_len={}", len(cropped_preview), len(session_state["pending"]["bash"]))
                    session_state["pending"] = None
            set_status("Preview refreshed.")
        except Exception as e:
            set_status(f"Preview refresh failed; see {log_path}.")
            log_exc("(REFRESH ERROR)", e)
        finally:
            refresh_state["busy"] = False
            application.invalidate()

    async def send_prompt():
        if send_state["busy"]:
            return
        if session_state["pending"]:
            set_status("Previous command not finished; wait for preview refresh.")
            return
        send_state["busy"] = True
        notes_text = filters.apply(notes_area.text).strip()
        if not notes_text:
            notes_text = "(none)"
        write_file(notes_path, notes_text)
        set_status("Sending to model...")
        application.invalidate()
        try:
            event_loop = asyncio.get_running_loop()
            before_capture = filters.apply(capture_clean(pane_id, capture_line_limit))
            session_state["before_capture"] = before_capture
            init_capture_for_context = compute_init_context(before_capture)
            history_rows = load_history(history_path, max_history_turns, filters)
            context_messages = build_context_messages(
                init_capture_for_context, history_rows, notes_text, filters, line_count=capture_line_limit
            )
            logger.info("Sending prompt to model={} captured_lines={} notes_length={}", model, capture_line_limit, len(notes_text))
            human_output, bash_output = await event_loop.run_in_executor(
                None, call_model_blocking, api_key, model, context_messages, paths, filters, config["instructions"]
            )
            human_area.text = human_output
            bash_area.text = bash_output
            notes_area.text = ""
            write_file(notes_path, "")
            logger.info("Model response received bash_len={} human_len={}", len(bash_output), len(human_output))
            bash_trimmed = bash_output.strip()
            human_trimmed = human_output.strip()
            if bash_trimmed:
                session_state["pending"] = {
                    "notes": notes_text,
                    "bash": bash_trimmed,
                    "human": human_trimmed,
                    "model": model,
                    "ts": int(time.time()),
                    "before_capture": before_capture,
                }
                set_status("Received. Press F6 to paste.")
            else:
                session_state["pending"] = None
                set_status("Received. No BASH commands to paste.")
        except Exception as e:
            error_text = "(MODEL ERROR)\n" + repr(e) + "\n\n" + traceback.format_exc()
            error_text = filters.apply(error_text)
            human_area.text = error_text
            bash_area.text = ""
            write_file(human_path, error_text)
            write_file(bash_path, "")
            set_status(f"Model error. See HUMAN / {log_path}.")
            log_exc("(MODEL ERROR)", e)
        finally:
            send_state["busy"] = False
            application.invalidate()

    @key_bindings.add("f5")
    def _(event):
        asyncio.create_task(send_prompt())

    @key_bindings.add("f6")
    def _(event):
        bash_text_content = bash_area.text
        if not bash_text_content.strip():
            set_status("BASH is empty; nothing to paste.")
            logger.warning("Paste requested with empty BASH output for pane {}", pane_id)
            return
        try:
            tmux_paste_and_enter(pane_id, bash_text_content)
            set_status(f"Pasted+Enter into {pane_id}. Refreshing preview...")
            logger.info("Pasted model output into pane {}", pane_id)
            asyncio.create_task(refresh_preview_sequence([0.15, 0.6, 1.5, 2.5, 3.5]))
        except Exception as e:
            set_status(f"Paste failed; see {log_path}.")
            log_exc("(PASTE ERROR)", e)

    @key_bindings.add("f7")
    def _(event):
        asyncio.create_task(clear_history_and_refresh_context())

    @key_bindings.add("f8")
    def _(event):
        asyncio.create_task(clear_terminal_context_only())

    @key_bindings.add("f10")
    @key_bindings.add("escape")
    def _(event):
        event.app.exit(result=0)

    return await application.run_async()


def main():
    try:
        return asyncio.run(run_tmux_llm())
    except SystemExit:
        raise
    except Exception as e:
        try:
            paths = resolve_paths("tmux-llm")
            try:
                config = load_config()
            except Exception:
                config = None
            logger, log_path, log_exc = open_log(paths, (config or {}).get("logging"))
            log_exc("(FATAL)", e)
        except Exception:
            pass
        raise


__all__ = [
    "append_history",
    "build_context_messages",
    "call_model_blocking",
    "capture_clean",
    "compute_cropped_preview",
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
