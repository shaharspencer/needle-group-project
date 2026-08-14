"""
Analysis package for the character-mortality project.

Importing this package forces UTF-8 on stdout/stderr. Many scripts here print
non-ASCII characters (progress ticks, arrows, box-drawing), and the Windows
console defaults to the cp1252 code page, which raises UnicodeEncodeError on
those prints and kills the run.

This covers everything under `src/`. `build_pipeline.py` sits at the top level
rather than in this package, so run it with UTF-8 mode enabled:

    PYTHONUTF8=1 python build_pipeline.py

Entry-point scripts call `setup_run_log(__spec__)` in their `if __name__ ==
"__main__":` block to mirror console output to `logs/X.Y.log`, appended run
over run with a timestamp header. A stage started with the shell's own
stdout redirect (`> file.log`) is invisible until it exits if Python buffers
the pipe -- this writes the file itself, one flush per line, so `tail`-ing a
stage from a second shell actually shows something. Nothing under `logs/` is
read by any pipeline stage; it exists to be looked at, not consumed. Not
tracked in git (see .gitignore).

This has to be opt-in per script rather than automatic from here: by the time
`python -m src.X.Y` gets around to importing this file, it has only imported
`src` and `src.X` as ordinary parent packages to resolve where `Y` lives --
the interpreter doesn't attach the real module spec to `sys.modules['__main__']`
until after that resolution finishes, so this file is always too early to see
it. Each script's own `__spec__` (distinct from `sys.modules['__main__']`) is
set correctly by the time its own `__main__` block runs, which is why the
call happens there instead.
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
    """Mirror this run's console output to logs/<dotted.module.name>.log.

    Call as `setup_run_log(__spec__)` from inside `if __name__ == "__main__":`
    -- that bare name, not sys.modules['__main__'].__spec__, is what correctly
    holds the module's real dotted path at that point (see this file's
    docstring for why the difference matters here).
    """
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
