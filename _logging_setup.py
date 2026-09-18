"""
Shared logging setup for every script in this project.

Import and call `setup_logging(script_name)` at the very top of any
script's __main__ block. It:
  - tees all stdout/stderr to both the console AND a timestamped log
    file under logs/, so output survives even if the terminal window
    closes unexpectedly.
  - installs a global exception hook so uncaught exceptions (the kind
    that can flash a console window closed on Windows) get written to
    the log file before the process exits, not lost with the window.

Usage (top of any script):
    from _logging_setup import setup_logging
    log_path = setup_logging("download_datasets")
"""

import sys
import traceback
from datetime import datetime
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parent / "logs"


class _Tee:
    """Writes to multiple streams at once (console + log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass


_already_setup = False


def setup_logging(script_name: str) -> Path:
    global _already_setup
    if _already_setup:
        # A script imported another script that also calls setup_logging
        # (e.g. download_datasets.py calling prepare_splits.main() in-process).
        # Keep writing to the first log file rather than opening a second
        # one and re-teeing, which would just split the output across files.
        return None

    LOGS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"{script_name}_{timestamp}.log"

    log_file = open(log_path, "w", encoding="utf-8", buffering=1)

    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)

    def _excepthook(exc_type, exc_value, exc_tb):
        # Make sure uncaught exceptions are written to the log BEFORE
        # the process exits, so a window that closes on crash doesn't
        # take the traceback with it.
        traceback.print_exception(exc_type, exc_value, exc_tb, file=sys.stderr)
        sys.stderr.write(
            f"\n[FATAL] Script crashed. Full log saved to: {log_path}\n"
        )
        sys.stderr.flush()

    sys.excepthook = _excepthook
    _already_setup = True

    print(f"[log] Logging to {log_path}")
    return log_path
