"""
Analysis package for the character-mortality project.

Importing this package forces UTF-8 on stdout/stderr. Many scripts here print
non-ASCII characters (progress ticks, arrows, box-drawing), and the Windows
console defaults to the cp1252 code page, which raises UnicodeEncodeError on
those prints and kills the run.

This covers everything under `src/`. `build_pipeline.py` sits at the top level
rather than in this package, so run it with UTF-8 mode enabled:

    PYTHONUTF8=1 python build_pipeline.py
"""

import datetime
import sys
from pathlib import Path

for _name in ("stdout", "stderr"):
    _stream = getattr(sys, _name, None)
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            # Stream isn't reconfigurable (already wrapped, or redirected oddly).
            pass


class _Tee:
    """Writes to every stream in `_targets`, flushing after each write."""

    def __init__(self, *targets):
        self._targets = targets

    def write(self, data):
        for target in self._targets:
            target.write(data)
        self.flush()

    def flush(self):
        for target in self._targets:
            target.flush()

    def isatty(self):
        return False


def setup_run_log(spec) -> None:
    """Mirror this run's console output to logs/<dotted.module.name>.log."""
    name = getattr(spec, "name", None)
    if not name:
        return
    if name.startswith("src."):
        name = name[len("src."):]

    log_dir = Path(__file__).resolve().parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    handle = open(log_dir / f"{name}.log", "a", encoding="utf-8")

    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    banner = f"\n{'=' * 70}\n{stamp}  python -m src.{name}\n{'=' * 70}\n"
    handle.write(banner)
    handle.flush()

    sys.stdout = _Tee(sys.stdout, handle)
    sys.stderr = _Tee(sys.stderr, handle)
