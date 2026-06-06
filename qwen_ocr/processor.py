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
    is_within_output_root,
    iter_input_files,
    markdown_path_for,
    relative_key,
    resolve_output_root,
    run_fingerprint,
    safe_checksum,
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


def _run_fingerprint(backend: Backend, dpi: int, params: InferenceParams) -> str:
    """Run-config fingerprint so a re-run under different output-affecting flags reprocesses.

    qwen has no task selector, but its page text genuinely depends on more than
    model + backend: the render ``--dpi`` (resolution at which the PDF is
    rasterized) and the :class:`InferenceParams` (prompt, token budget,
    temperature, repetition penalty, max image side) all change what OCR a given
    input produces. v0.1.2's ``run_fingerprint(extra=...)`` folds these RESOLVED
    flags into the fingerprint, which :meth:`RootIndex.is_completed` consults, so
    re-running the same input at a different ``--dpi`` (or with changed inference
    params) reprocesses instead of silently reusing stale lower-res OCR.
    """
    extra = {
        "dpi": dpi,
        "max_output_tokens": params.max_output_tokens,
        "temperature": params.temperature,
        "repetition_penalty": params.repetition_penalty,
        "max_image_side": params.max_image_side,
        "prompt": params.prompt,
    }
    return run_fingerprint(_backend_model(backend), backend.name, extra=extra)


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


def _safe_doc_checksum(source: Path) -> str | None:
    """Tolerant :func:`_doc_checksum`: ``None`` instead of raising on bad input.

    A document discovered during the batch scan can be unreadable by the time it
    is processed (permission denied, deleted/replaced mid-run, a broken symlink
    that passed discovery). Returning ``None`` lets the caller record that one
    doc as a per-file FAILURE and CONTINUE the batch, rather than letting an
    ``OSError`` propagate and abort the whole run (the SYS-02 "one bad file
    aborts the batch" failure mode). Mirrors the contract's
    :func:`safe_checksum` for the single-file case and extends the tolerance to
    the image-directory case.
    """
    try:
        if source.is_file():
            return safe_checksum(source)
        return _doc_checksum(source)
    except OSError:
        return None


def _build_doc_metadata(
    result: DocResult,
    markdown_path: Path,
    output_root: Path,
    backend: Backend,
    fingerprint: str,
) -> DocMetadata:
    """Assemble the per-document metadata record from a DocResult."""
    status = result.status
    error = None
    if status is not Status.COMPLETED:
        if result.page_errors:
            error = "; ".join(f"page {n}: {msg}" for n, msg in sorted(result.page_errors.items()))
        elif result.error:
            error = result.error
    # Tolerant checksum so persisting a status=failed record never itself throws
    # when the input became unreadable mid-run. An empty sentinel never matches a
    # real sha256:... checksum, so a failed entry is never wrongly skipped later.
    return DocMetadata(
        status=status,
        checksum=_safe_doc_checksum(result.source) or "",
        model=_backend_model(backend),
        backend=backend.name,
        processing_time=result.processing_time,
        timestamp=utc_timestamp(),
        output_path=str(markdown_path.relative_to(output_root)),
        pages=result.page_count,
        error=error,
        fingerprint=fingerprint,
    )


def _write_document(
    result: DocResult,
    output_root: Path,
    rel_key: str,
    backend: Backend,
    index: RootIndex,
    fingerprint: str,
) -> tuple[DocMetadata, Path]:
    """Write the aggregated markdown + BOTH metadata levels for one document.

    Output is always written (even on failure) so failures are recorded with
    ``status=failed`` per the canon. The single ``<stem>/<stem>.md`` aggregates
    every page under ``## Page N`` headers (fixes QWEN-01). ``fingerprint`` is the
    run-config fingerprint recorded so a re-run under different output-affecting
    flags reprocesses (must match the one used for the skip pre-check).
    """
    doc_dir = doc_dir_for(output_root, rel_key)
    doc_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = markdown_path_for(doc_dir, rel_key)

    body = assemble_pages(result.pages) if result.pages else "*[OCR Failed]*\n"
    markdown_path.write_text(body, encoding="utf-8")

    meta = _build_doc_metadata(result, markdown_path, output_root, backend, fingerprint)
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
    # Resolve the output root BEFORE discovery so the recursive batch scan can
    # prune the engine's own ``ocr/`` output subtree (no self-ingestion on re-run).
    output_root = resolve_output_root(source, output_dir)
    documents, single_image_dir = _discover_documents(source, output_root)
    if not documents:
        raise ValueError(f"no documents found at {source}")

    output_root.mkdir(parents=True, exist_ok=True)
    # A single image-dir document keys on its own folder name (an image dir has
    # no file stem); a batch tree keys each .pdf input-relative to the tree root.
    scan_root = source.parent if (source.is_file() or single_image_dir) else source
    index = RootIndex(output_root)
    # Fingerprint covers all output-affecting flags (model, backend, dpi, inference
    # params): a re-run at a different --dpi or with changed params reprocesses.
    fingerprint = _run_fingerprint(backend, dpi, params)

    outcome = RunOutcome()
    for doc in documents:
        rel_key = relative_key(doc, scan_root)
        # Tolerant checksum: an input unreadable at pre-check time (permission
        # denied, deleted/replaced mid-run, broken symlink) is recorded as a
        # per-file FAILURE and the batch CONTINUES, instead of an OSError
        # aborting the whole run (SYS-02). v0.1.2's safe_checksum primitive.
        checksum = _safe_doc_checksum(doc)
        if checksum is None:
            logger.error("input unreadable, recording failed: %s", rel_key)
            result = DocResult(source=doc, pages=[], error="input unreadable (checksum failed)")
            meta, markdown_path = _write_document(
                result, output_root, rel_key, backend, index, fingerprint
            )
            outcome.add(meta.status, detail=rel_key, output_path=str(markdown_path))
            continue

        if not reprocess and index.is_completed(rel_key, checksum, fingerprint=fingerprint):
            logger.info("skip %s (already completed; use --reprocess)", rel_key)
            # Quiet-mode scripting contract: emit the existing .md path for a
            # skipped-but-present doc (v0.1.1 is_completed already verified it
            # exists on disk, but recompute + re-check defensively).
            skip_md = markdown_path_for(doc_dir_for(output_root, rel_key), rel_key)
            outcome.add(
                Status.COMPLETED,
                output_path=str(skip_md) if skip_md.exists() else None,
            )
            continue

        result = _ocr_one_document(doc, backend, params, dpi)
        meta, markdown_path = _write_document(
            result, output_root, rel_key, backend, index, fingerprint
        )
        outcome.add(
            meta.status,
            detail=None if meta.status is Status.COMPLETED else rel_key,
            output_path=str(markdown_path),
        )

    return outcome


def _is_page_image_dir(source: Path, output_root: Path) -> bool:
    """True if ``source`` directly contains ONLY page images (no PDFs, no junk).

    This is the ONLY shape qwen treats as a single image-directory document: the
    pure page-image dir socr renders a PDF into (``page_0001.png`` ...). Dotfiles
    and OS junk (``.DS_Store``, ``Thumbs.db``) are ignored so they don't demote a
    clean scan dir. A directory that also holds a PDF (or any non-image,
    non-junk file) is NOT a single image-doc — it is a batch tree (see
    :func:`_discover_documents`), so a co-located PDF is never silently dropped.

    The engine's OWN resolved output root (e.g. ``<source>/ocr/`` when ``-o`` is
    omitted) is ignored for classification: otherwise the ``ocr/`` subdir created
    by the first run would demote the same image dir to a "batch tree" on the
    second run, where ``_DOC_SUFFIXES={.pdf}`` finds nothing and process() raises
    "no documents found" — a documented invocation breaking on its second call.
    """
    saw_image = False
    for p in source.iterdir():
        if p.name.startswith(".") or p.name in {"Thumbs.db", "thumbs.db"}:
            continue  # OS/editor junk: ignore for classification
        if is_within_output_root(p, output_root):
            continue  # the engine's own output subtree: never demotes the scan
        if not p.is_file():
            return False  # a (non-output) subdirectory means a batch tree
        if p.suffix.lower() in _IMAGE_SUFFIXES:
            saw_image = True
            continue
        return False  # any non-image file (e.g. a PDF) -> treat the dir as a batch
    return saw_image


def _discover_documents(source: Path, output_root: Path) -> tuple[list[Path], bool]:
    """Return ``(documents, single_image_dir)`` for an input path.

    * A bare file is one document.
    * A directory containing ONLY page images is a SINGLE image-dir document
      (the pure ``page_0001.png`` ... dir socr passes). Returns
      ``single_image_dir=True`` so the caller keys it on the folder name.
    * Any other directory is a BATCH tree: every ``.pdf`` under it (recursively)
      is one document, discovered via the contract's :func:`iter_input_files`,
      which excludes the resolved ``output_root`` so the engine never re-ingests
      its own ``ocr/`` outputs on a re-run. A directory holding a PDF alongside
      loose images takes this path, so the PDF is OCR'd (the loose images, which
      have no document identity, are not folded into a bogus single doc).
    """
    if source.is_file():
        return [source], False
    if source.is_dir():
        # The output root itself is never a source document.
        if is_within_output_root(source, output_root):
            return [], False
        if _is_page_image_dir(source, output_root):
            return [source], True
        docs = list(iter_input_files(source, output_root, _DOC_SUFFIXES))
        return docs, False
    return [], False
