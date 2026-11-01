"""Helpers for tracking the process memory footprint.

The detector deliberately avoids hard `RLIMIT` caps: PyTorch and the maths
libraries reserve a large amount of *virtual* address space even when the real
resident set is small.  A hard address-space limit therefore causes confusing
crashes.  Instead we report the resident set size (RSS) so the camera loop can
warn or stop before the machine starts swapping.
"""

from __future__ import annotations

import os


def rss_mb() -> float | None:
    """Return the resident set size of this process in megabytes, if known."""
    try:
        with open("/proc/self/statm", encoding="ascii") as statm:
            pages = int(statm.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / 1_000_000
    except (OSError, ValueError, IndexError):
        pass

    try:  # Portable fallback, only used when /proc is unavailable.
        import psutil

        return psutil.Process().memory_info().rss / 1_000_000
    except Exception:
        return None


def format_rss() -> str:
    """Return a short human-readable memory string such as ``412 MB``."""
    value = rss_mb()
    return "unknown" if value is None else f"{value:.0f} MB"
