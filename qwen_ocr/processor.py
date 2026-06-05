"""Render inputs to page images, OCR each via a backend, write canonical Markdown.

qwen is a *local, page-based* engine: it renders a PDF (or reads a directory of
page images) to per-page PNGs and OCRs each page through one of its backends
(ollama / vllm / dashscope / openrouter). This module owns *how OCR happens*;
the shared ``ocr-output-contract`` package owns *where the bytes go* and what
shape the metadata takes.

The per-page text list this module produces is fed straight into the contract's
:func:`assemble_pages`, so every page of a document lands in ONE
``<root>/<rel/dir>/<stem>/<stem>.md`` under ``## Page N`` headers (fixes the
CRITICAL multi-page data loss, QWEN-01: qwen used to write one folder per page,
and socr's read-back kept only the first). An empty/whitespace page response is
a per-page FAILURE recorded in the sidecar (QWEN-02), never a silent 0-byte
success reported as ok.
"""

from __future__ import annotations

import hashlib
import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from ocr_output_contract import (
    DocMetadata,
    RootIndex,
    RunOutcome,
    Status,
    assemble_pages,
    doc_dir_for,
    markdown_path_for,
    relative_key,
    resolve_output_root,
    sha256_checksum,
    utc_timestamp,
    write_doc_metadata,
)

from qwen_ocr.backends.base import Backend
from qwen_ocr.config import InferenceParams
from qwen_ocr.utils import sanitize_filename, trim_degenerate_tail

logger = logging.getLogger(__name__)

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
#: Inputs qwen treats as a single source document to render page-by-page.
_DOC_SUFFIXES = {".pdf"}


@dataclass
class DocResult:
    """Result of OCR'ing one source document (a PDF or an image directory).

    ``pages`` holds the per-page markdown in order; ``page_errors`` maps a
    1-indexed page number to its error string for any page that failed (or whose
    model response was empty). ``status`` maps to the contract enum.
    """

    source: Path
    pages: list[str]
    processing_time: float = 0.0
    page_errors: dict[int, str] = field(default_factory=dict)
    error: str | None = None

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def status(self) -> Status:
        """``completed`` = every page ok; ``partial`` = some ok; ``failed`` = none.

        A failed page slot still carries a placeholder marker so page numbering
        stays aligned, so genuine successes are counted as ``page_count -
        len(page_errors)`` rather than by scanning the (placeholder-filled) text.
        """
        if self.pages and not self.page_errors and not self.error:
            return Status.COMPLETED
        succeeded = self.page_count - len(self.page_errors)
        if succeeded > 0:
            return Status.PARTIAL
        return Status.FAILED


# ---------------------------------------------------------------------------
# Page OCR (bytes -> page-text; qwen-owned). Output shape is the contract's job.
# ---------------------------------------------------------------------------


def _ocr_pages(
    source: Path,
    images: list[Path],
    backend: Backend,
    params: InferenceParams,
    start: float,
) -> DocResult:
    """OCR an ordered list of page-image paths into a DocResult.

    Each page is an independent backend call so page boundaries and per-page
    failures are real. A page that errors OR returns empty/whitespace text is
    recorded in ``page_errors`` (QWEN-02: empty != success) and gets an explicit
    failure marker in its slot, keeping the page count and ``## Page N``
    numbering aligned with the source.
    """
    pages: list[str] = []
    page_errors: dict[int, str] = {}
    for idx, image in enumerate(images, start=1):
        try:
            text = backend.ocr_image(image, params)
            text = trim_degenerate_tail(text)
            if not text.strip():
                raise ValueError("empty OCR response (no text returned)")
            pages.append(text)
        except Exception as exc:
            logger.error("OCR failed for page %d of %s: %s", idx, source.name, exc)
            page_errors[idx] = str(exc)
            pages.append(f"*[OCR failed for page {idx}]*")
    return DocResult(
        source=source,
        pages=pages,
        processing_time=time.time() - start,
        page_errors=page_errors,
    )


def _ocr_one_document(
    doc: Path,
    backend: Backend,
    params: InferenceParams,
    dpi: int,
) -> DocResult:
    """OCR one document, rendering a PDF to PNGs inside a live temp directory."""
    start = time.time()
    try:
        if doc.is_dir():
            images = _gather_images(doc)
            return _ocr_pages(doc, images, backend, params, start)
        if doc.suffix.lower() in _DOC_SUFFIXES:
            with tempfile.TemporaryDirectory() as tmp:
                images = _render_pdf(doc, Path(tmp), dpi)
                return _ocr_pages(doc, images, backend, params, start)
        raise ValueError(f"unsupported input: {doc} (expected a .pdf or a directory)")
    except Exception as exc:
        logger.error("could not process %s: %s", doc, exc)
        return DocResult(source=doc, pages=[], processing_time=time.time() - start, error=str(exc))


def _gather_images(directory: Path) -> list[Path]:
    images = sorted(p for p in directory.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f"no page images found in {directory}")
    return images


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


# ---------------------------------------------------------------------------
# Output writing (all routed through the ocr-output-contract package)
# ---------------------------------------------------------------------------


def _backend_model(backend: Backend) -> str:
    """Best-effort model id for provenance, via the backend's endpoint tuple."""
    try:
        _base_url, _key, model = backend._endpoint()
        return model
    except Exception:
        return getattr(backend, "model", "") or ""


def _doc_checksum(source: Path) -> str:
    """Content checksum for idempotency. Hash a PDF's bytes, or an image dir's.

    For an image-directory document there is no single file, so derive a stable
    checksum from the sorted page filenames and their bytes.
    """
    if source.is_file():
        return sha256_checksum(source)
    h = hashlib.sha256()
    for img in _gather_images(source):
        h.update(img.name.encode("utf-8"))
        with open(img, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def _build_doc_metadata(
    result: DocResult,
    markdown_path: Path,
    output_root: Path,
    backend: Backend,
) -> DocMetadata:
    """Assemble the per-document metadata record from a DocResult."""
    status = result.status
    error = None
    if status is not Status.COMPLETED:
        if result.page_errors:
            error = "; ".join(f"page {n}: {msg}" for n, msg in sorted(result.page_errors.items()))
        elif result.error:
            error = result.error
    return DocMetadata(
        status=status,
        checksum=_doc_checksum(result.source),
        model=_backend_model(backend),
        backend=backend.name,
        processing_time=result.processing_time,
        timestamp=utc_timestamp(),
        output_path=str(markdown_path.relative_to(output_root)),
        pages=result.page_count,
        error=error,
    )


def _write_document(
    result: DocResult,
    output_root: Path,
    rel_key: str,
    backend: Backend,
    index: RootIndex,
) -> tuple[DocMetadata, Path]:
    """Write the aggregated markdown + BOTH metadata levels for one document.

    Output is always written (even on failure) so failures are recorded with
    ``status=failed`` per the canon. The single ``<stem>/<stem>.md`` aggregates
    every page under ``## Page N`` headers (fixes QWEN-01).
    """
    doc_dir = doc_dir_for(output_root, rel_key)
    doc_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = markdown_path_for(doc_dir, rel_key)

    body = assemble_pages(result.pages) if result.pages else "*[OCR Failed]*\n"
    markdown_path.write_text(body, encoding="utf-8")

    meta = _build_doc_metadata(result, markdown_path, output_root, backend)
    write_doc_metadata(doc_dir, rel_key, meta)
    index.record(rel_key, meta)
    return meta, markdown_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def process(
    source: Path,
    backend: Backend,
    params: InferenceParams,
    dpi: int,
    output_dir: Path | None = None,
    reprocess: bool = False,
) -> RunOutcome:
    """Process an input (file or directory of documents) through the contract.

    ``source`` is either a single document (a ``.pdf``, or a directory of page
    images treated as ONE document) or a batch directory tree containing such
    documents. Output goes to ``resolve_output_root(source, output_dir)`` —
    default ``<input-parent>/ocr/``; ``-o`` overrides; never required.

    Returns a :class:`RunOutcome` whose ``exit_code`` is nonzero if any
    document/page failed (uniform across single-file and batch).
    """
    documents = _discover_documents(source)
    if not documents:
        raise ValueError(f"no documents found at {source}")

    output_root = resolve_output_root(source, output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    # A single image-dir document keys on its own folder name (an image dir has
    # no file stem); a batch tree keys each .pdf input-relative to the tree root.
    single_image_dir = len(documents) == 1 and documents[0] == source and source.is_dir()
    scan_root = source.parent if (source.is_file() or single_image_dir) else source
    index = RootIndex(output_root)

    outcome = RunOutcome()
    for doc in documents:
        rel_key = relative_key(doc, scan_root)
        if not reprocess and index.is_completed(rel_key, _doc_checksum(doc)):
            logger.info("skip %s (already completed; use --reprocess)", rel_key)
            outcome.add(Status.COMPLETED)
            continue

        result = _ocr_one_document(doc, backend, params, dpi)
        meta, markdown_path = _write_document(result, output_root, rel_key, backend, index)
        outcome.add(
            meta.status,
            detail=None if meta.status is Status.COMPLETED else rel_key,
            output_path=str(markdown_path),
        )

    return outcome


def _discover_documents(source: Path) -> list[Path]:
    """Return the list of source documents under ``source``.

    A bare ``.pdf`` is one document. A directory is treated as a SINGLE image-dir
    document when it directly contains page images (socr renders a PDF to PNGs
    and passes the dir); otherwise it is a batch tree and every ``.pdf`` under it
    (recursively) is one document.
    """
    if source.is_file():
        return [source]
    if source.is_dir():
        has_direct_images = any(
            p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES for p in source.iterdir()
        )
        if has_direct_images:
            return [source]
        return sorted(p for p in source.rglob("*") if p.suffix.lower() in _DOC_SUFFIXES)
    return []
