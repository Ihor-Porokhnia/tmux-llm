import argparse
from pathlib import Path

MARKER_START = "# >>> tmux-llm config start"
MARKER_END = "# <<< tmux-llm config end"


def _render_block(cmd="tmux-llm"):
    bind_cmd = f"bind-key C-g display-popup -E -w 95% -h 90% '{cmd}'"
    return "\n".join(
        [
            MARKER_START,
            '# Managed by tmux-llm. Safe to re-run "tmux-llm-config install".',
            'if-shell \'test -n "$OPENAI_API_KEY"\' "set-environment -g OPENAI_API_KEY \'$OPENAI_API_KEY\'"',
            f'if-shell \'command -v {cmd} >/dev/null 2>&1\' "{bind_cmd}" '
            '"display-message \\"tmux-llm not installed (pip install -e .)\\""',
            MARKER_END,
            "",
        ]
    )


def install_tmux_config(tmux_conf=None, cmd="tmux-llm"):
    """
    Insert or update a managed tmux-llm block in ~/.tmux.conf (idempotent).
    """
    tmux_conf_path = Path(tmux_conf or Path.home() / ".tmux.conf")
    block = _render_block(cmd)
    if tmux_conf_path.exists():
        text = tmux_conf_path.read_text(encoding="utf-8")
    else:
        text = ""
    if MARKER_START in text and MARKER_END in text:
        before, _, rest = text.partition(MARKER_START)
        _, _, after = rest.partition(MARKER_END)
        new_text = before.rstrip() + "\n" + block + after.lstrip("\n")
    else:
        sep = "\n" if text and not text.endswith("\n") else ""
        new_text = text + sep + block
    tmux_conf_path.write_text(new_text, encoding="utf-8")
    return tmux_conf_path


def remove_tmux_config(tmux_conf=None):
    """
    Remove the managed tmux-llm block from ~/.tmux.conf if present.
    """
    tmux_conf_path = Path(tmux_conf or Path.home() / ".tmux.conf")
    if not tmux_conf_path.exists():
        return False
    text = tmux_conf_path.read_text(encoding="utf-8")
    if MARKER_START not in text or MARKER_END not in text:
        return False
    before, _, rest = text.partition(MARKER_START)
    _, _, after = rest.partition(MARKER_END)
    new_text = before.rstrip("\n")
    if after:
        if new_text:
            new_text += "\n"
        new_text += after.lstrip("\n")
    tmux_conf_path.write_text(new_text, encoding="utf-8")
    return True


def cli(argv=None):
    parser = argparse.ArgumentParser(description="Manage tmux-llm config injection into ~/.tmux.conf")
    parser.add_argument("--tmux-conf", default=None, help="Path to tmux.conf (default: ~/.tmux.conf)")
    parser.add_argument("--cmd", default="tmux-llm", help="Command bound in tmux (default: tmux-llm)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("install", help="Insert or update tmux-llm block")
    sub.add_parser("remove", help="Remove tmux-llm block")
    args = parser.parse_args(argv)
    if args.command == "install":
        p = install_tmux_config(args.tmux_conf, args.cmd)
        print(f"tmux-llm block installed in {p}")
        return 0
    if args.command == "remove":
        removed = remove_tmux_config(args.tmux_conf)
        if removed:
            print("tmux-llm block removed")
        else:
            print("tmux-llm block not found")
        return 0
    parser.error("Unknown command")


__all__ = ["cli", "install_tmux_config", "remove_tmux_config"]
