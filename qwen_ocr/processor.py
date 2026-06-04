"""Render inputs to page images, OCR each via a backend, write per-page Markdown.

Accepts a PDF (rendered to PNGs at --dpi) or a directory of page images. Output layout is
`{out}/{stem}/{stem}.md` per input stem — exactly what socr's adapter reads back.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

from qwen_ocr.backends.base import Backend
from qwen_ocr.config import InferenceParams
from qwen_ocr.utils import sanitize_filename, trim_degenerate_tail

logger = logging.getLogger(__name__)

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}


@dataclass
class PageResult:
    stem: str
    output_path: Path | None
    ok: bool
    error: str = ""


def process(
    source: Path,
    out_dir: Path,
    backend: Backend,
    params: InferenceParams,
    dpi: int,
    reprocess: bool = False,
) -> list[PageResult]:
    """Process a PDF or image directory; return one PageResult per page image."""
    out_dir.mkdir(parents=True, exist_ok=True)

    if source.is_dir():
        images = _gather_images(source)
        if not images:
            raise ValueError(f"no page images found in {source}")
        return [_ocr_one(img, out_dir, backend, params, reprocess) for img in images]

    if source.suffix.lower() == ".pdf":
        with tempfile.TemporaryDirectory() as tmp:
            images = _render_pdf(source, Path(tmp), dpi)
            return [_ocr_one(img, out_dir, backend, params, reprocess) for img in images]

    raise ValueError(f"unsupported input: {source} (expected a .pdf or a directory)")


def _gather_images(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES
    )


def _render_pdf(pdf_path: Path, tmp_dir: Path, dpi: int) -> list[Path]:
    """Render each PDF page to a PNG named after the doc stem + page index."""
    import fitz

    stem = sanitize_filename(pdf_path.stem)
    out: list[Path] = []
    with fitz.open(pdf_path) as doc:
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        for i in range(len(doc)):
            pix = doc[i].get_pixmap(matrix=mat)
            name = stem if len(doc) == 1 else f"{stem}_p{i + 1:04d}"
            png = tmp_dir / f"{name}.png"
            pix.save(png)
            out.append(png)
    return out


def _ocr_one(
    image: Path,
    out_dir: Path,
    backend: Backend,
    params: InferenceParams,
    reprocess: bool,
) -> PageResult:
    stem = sanitize_filename(image.stem)
    md_path = out_dir / stem / f"{stem}.md"

    if md_path.exists() and not reprocess:
        logger.info("skip %s (exists; use --reprocess)", stem)
        return PageResult(stem, md_path, ok=True)

    try:
        text = backend.ocr_image(image, params)
        text = trim_degenerate_tail(text)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(text, encoding="utf-8")
        return PageResult(stem, md_path, ok=True)
    except Exception as exc:  # one bad page must not abort the batch
        logger.error("OCR failed for %s: %s", stem, exc)
        return PageResult(stem, None, ok=False, error=str(exc))
