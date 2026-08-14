"""
Analysis package for the character-mortality project.

Importing this package forces UTF-8 on stdout/stderr. Many scripts here print
non-ASCII characters (progress ticks, arrows, box-drawing), and the Windows
console defaults to the cp1252 code page, which raises UnicodeEncodeError on
those prints and kills the run.

This covers everything under `src/`. The older top-level scripts
(build_pipeline.py, feature_scan.py, tfidf.py, ...) are not in this package, so
run those with UTF-8 mode enabled:

    PYTHONUTF8=1 python build_pipeline.py
"""

import sys

for _name in ("stdout", "stderr"):
    _stream = getattr(sys, _name, None)
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            # Stream isn't reconfigurable (already wrapped, or redirected oddly).
            pass
