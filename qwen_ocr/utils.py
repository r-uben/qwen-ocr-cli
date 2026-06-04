"""Small, pure helpers: filename sanitizing + degenerate-output trimming.

Post-processing matters even with strong models (Dasanaike, 20M-doc post): high-res pages
sometimes end in a repetition loop. We trim a long trailing run of identical lines/blocks
before writing. Detection-only; socr's downstream audit still has the final say.
"""

from __future__ import annotations


def sanitize_filename(name: str) -> str:
    """Match socr's BaseEngine.sanitize_filename so output stems line up."""
    return "".join(c if c.isalnum() or c in "._- " else "_" for c in name).strip()


def trim_degenerate_tail(text: str, min_repeats: int = 6) -> str:
    """Drop a trailing run of an identical non-empty line repeated >= min_repeats times.

    Conservative: only the final collapsed run is removed, and only when it is clearly a
    loop (many identical consecutive lines). Leaves legitimate repeated short lines
    (e.g. table separators) below the threshold untouched.
    """
    lines = text.splitlines()
    if len(lines) < min_repeats:
        return text

    last = lines[-1].strip()
    if not last:
        return text

    run = 0
    for line in reversed(lines):
        if line.strip() == last:
            run += 1
        else:
            break

    if run >= min_repeats:
        kept = lines[: len(lines) - run + 1]  # keep one instance
        return "\n".join(kept).rstrip() + "\n"
    return text
