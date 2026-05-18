"""
clustering/utils.py
-------------------
Shared utilities: logging setup, progress reporting, and JSON I/O.

All modules in this package obtain their logger via get_logger(__name__) so
every log line is prefixed with the originating module name.
"""

import json
import logging
import os
import sys
import time
from typing import Any, Optional

# ── Logging ──────────────────────────────────────────────────────────────────

_LOG_FORMAT = "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s"
_DATE_FORMAT = "%H:%M:%S"

_logging_configured = False


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    silent: bool = False,
) -> None:
    """Configure the root logger once.

    Call this at the top of run_clustering.py or any entry point.
    Subsequent calls are no-ops (idempotent).
    """
    global _logging_configured
    if _logging_configured:
        return

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handlers: list[logging.Handler] = []

    if not silent:
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(numeric_level)
        console.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
        handlers.append(console)

    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)          # always capture DEBUG in file
        fh.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
        handlers.append(fh)

    logging.basicConfig(level=logging.DEBUG, handlers=handlers)
    _logging_configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.  Call as: logger = get_logger(__name__)"""
    return logging.getLogger(name)


# ── Progress reporting ────────────────────────────────────────────────────────

class ProgressReporter:
    """
    Simple text-mode progress reporter for long-running loops.

    Usage::

        rpt = ProgressReporter(total=n_pairs, label="Pairwise TED")
        for i, j in pairs:
            rpt.update()
        rpt.done()
    """

    def __init__(self, total: int, label: str = "Progress", width: int = 40) -> None:
        self.total = max(total, 1)
        self.label = label
        self.width = width
        self._done = 0
        self._start = time.time()
        self._last_print = -1.0

    def update(self, n: int = 1) -> None:
        self._done += n
        now = time.time()
        # print at most every 0.5 s to avoid flooding the terminal
        if now - self._last_print < 0.5 and self._done < self.total:
            return
        self._last_print = now
        self._render()

    def _render(self) -> None:
        pct = self._done / self.total
        filled = int(self.width * pct)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.time() - self._start
        eta = (elapsed / self._done * (self.total - self._done)) if self._done else 0
        sys.stdout.write(
            f"\r  {self.label}: [{bar}] {self._done}/{self.total} "
            f"({pct*100:.1f}%)  ETA {eta:.0f}s"
        )
        sys.stdout.flush()

    def done(self) -> None:
        self._done = self.total
        self._render()
        elapsed = time.time() - self._start
        sys.stdout.write(f"  — done in {elapsed:.1f}s\n")
        sys.stdout.flush()


# ── JSON I/O ──────────────────────────────────────────────────────────────────

def save_json(data: Any, path: str, indent: int = 2) -> None:
    """Serialize *data* to JSON, creating parent directories as needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(path: str) -> Any:
    """Deserialize JSON from *path*."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Matrix helpers ────────────────────────────────────────────────────────────

def matrix_stats(matrix: list[list[float]]) -> dict:
    """Return basic statistics for a square symmetric distance matrix.

    Computes stats on the upper triangle only (off-diagonal values).
    """
    n = len(matrix)
    values = [
        matrix[i][j]
        for i in range(n)
        for j in range(i + 1, n)
    ]
    if not values:
        return {"n": n, "pairs": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}

    total = sum(values)
    mean = total / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = variance ** 0.5

    return {
        "n": n,
        "pairs": len(values),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "mean": round(mean, 6),
        "std": round(std, 6),
    }
