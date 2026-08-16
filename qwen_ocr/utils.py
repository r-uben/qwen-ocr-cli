"""Small, pure helpers: filename sanitizing + degenerate-output trimming.

Post-processing matters even with strong models (Dasanaike, 20M-doc post): high-res pages
sometimes end in a repetition loop. We trim a long trailing run of identical lines/blocks
before writing. Detection-only; socr's downstream audit still has the final say.
"""

from __future__ import annotations

import re

from qwen_ocr import config

#: A hybrid-thinking model's reasoning block, when the server inlines it into the
#: message content instead of a separate ``reasoning_content`` field.
_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Remove ``<think>...</think>`` blocks from a model response.

    Belt-and-braces for issue #4: the real fix is asking the server not to think
    (``InferenceParams.enable_thinking=False``), but a server that ignores the switch
    must not silently write reasoning into the document body. Only *matched* pairs are
    removed, so ordinary page text containing a stray angle bracket is untouched.
    """
    return _THINK_BLOCK.sub("", text).strip()


def smart_resize(
    width: int,
    height: int,
    min_pixels: int,
    max_pixels: int,
    factor: int = config.PATCH_FACTOR,
) -> tuple[int, int]:
    """Qwen's patch-aligned resize: the cookbook's rule, applied client-side.

    Returns the ``(width, height)`` to send so that both sides are multiples of
    ``factor`` and the area lands inside ``[min_pixels, max_pixels]``, preserving
    aspect ratio. This is the reference implementation from QwenLM/Qwen3-VL
    (``cookbooks/ocr.ipynb`` / ``qwen_vl_utils``), reproduced here so the pixel
    budget is enforced on EVERY backend rather than left to a server-side default
    we neither choose nor record (issue #5).
    """
    if min(width, height) <= 0:
        raise ValueError(f"degenerate image size: {width}x{height}")

    def _round(x: float) -> int:
        return max(factor, round(x / factor) * factor)

    w_bar, h_bar = _round(width), _round(height)
    if w_bar * h_bar > max_pixels:
        beta = ((width * height) / max_pixels) ** 0.5
        w_bar = max(factor, int(width / beta) // factor * factor)
        h_bar = max(factor, int(height / beta) // factor * factor)
    elif w_bar * h_bar < min_pixels:
        beta = (min_pixels / (width * height)) ** 0.5
        w_bar = -(-int(width * beta) // factor) * factor
        h_bar = -(-int(height * beta) // factor) * factor
    return w_bar, h_bar


def sanitize_filename(name: str) -> str:
    """Sanitize a stem for use as a *temp render PNG* name only.

    Output paths are owned by ``ocr-output-contract`` and are deliberately
    stem-preserving (no sanitization), so this is NOT applied to the output
    ``<stem>/<stem>.md`` path. It is used only to name the intermediate per-page
    PNGs ``_render_pdf`` writes into a TemporaryDirectory, where a filesystem-safe
    name avoids issues with odd source stems (the contract keys output off the
    raw input-relative path regardless).
    """
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
