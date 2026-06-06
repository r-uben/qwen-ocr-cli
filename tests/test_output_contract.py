"""Engine-level conformance tests: qwen's REAL output vs the shared contract.

The contract *primitives* (path/key computation, page assembly, metadata writers,
the exit-code policy) are unit-tested inside the ``ocr-output-contract`` package
itself, so they are NOT re-tested here.

What stays here is the engine-side proof: run qwen's actual processor (with a
mocked backend) over real inputs and assert the produced output tree conforms to
the family-wide contract via the package's reusable
:func:`ocr_output_contract.conformance.assert_conforms` harness — including the
two cases that motivated this work:

* QWEN-01 (CRITICAL): a multi-page PDF must aggregate into ONE ``<stem>/<stem>.md``
  with ``## Page 1`` and ``## Page 2`` both present (no per-page folders, no
  silent data loss when read back).
* QWEN-02 (HIGH): an empty/whitespace model response is a FAILURE recorded with
  ``status=failed`` and drives a nonzero exit (not a silent 0-byte success).
"""

from __future__ import annotations

import json

import fitz
from ocr_output_contract.conformance import ExpectedDoc, assert_conforms

from qwen_ocr.backends.base import Availability, Backend
from qwen_ocr.config import InferenceParams
from qwen_ocr.processor import process


class FakeBackend(Backend):
    """A mocked backend: deterministic per-page text, no network/model."""

    name = "ollama"

    def __init__(self, text="OCR page text", model="qwen3-vl:8b"):
        self.text = text
        self._model = model
        self.calls = 0

    def availability(self):
        return Availability(True)

    def _endpoint(self):
        return ("http://localhost:11434/v1", "ollama", self._model)

    def ocr_image(self, image_path, params):
        self.calls += 1
        return self.text


def _make_pdf(path, pages=2):
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page(width=300, height=400)
        page.insert_text((40, 60), f"Source page {i + 1}")
    doc.save(path)
    doc.close()


def test_multipage_pdf_conforms_no_data_loss(tmp_path):
    """QWEN-01 fixed: a 2-page PDF -> one <stem>/<stem>.md with both pages."""
    pdf = tmp_path / "sample.pdf"
    _make_pdf(pdf, pages=2)
    out = tmp_path / "out"

    # Distinct text per page so we can prove BOTH survived (not just the first).
    counter = {"n": 0}

    class PerPage(FakeBackend):
        def ocr_image(self, image_path, params):
            counter["n"] += 1
            return f"PAGE-{counter['n']}-CONTENT"

    outcome = process(pdf, PerPage(), InferenceParams(), dpi=120, output_dir=out)
    assert outcome.exit_code == 0

    # The package harness asserts every canonical invariant against the REAL tree.
    assert_conforms(
        out,
        [ExpectedDoc(rel_key="sample.pdf", pages=2, status="completed")],
        require_failures_nonzero_exit=outcome.exit_code != 0,
    )

    md = out / "sample" / "sample.md"
    body = md.read_text()
    assert "## Page 1" in body and "## Page 2" in body
    # Both pages' genuine content present — neither silently dropped.
    assert "PAGE-1-CONTENT" in body and "PAGE-2-CONTENT" in body
    # No per-page folders (the old lossy layout).
    assert not (out / "sample_p0001").exists()
    assert not (out / "sample_p0002").exists()


def test_empty_response_fails_and_exits_nonzero(tmp_path):
    """QWEN-02 fixed: an empty model response -> status=failed, nonzero exit."""
    pdf = tmp_path / "blank.pdf"
    _make_pdf(pdf, pages=1)
    out = tmp_path / "out"

    outcome = process(pdf, FakeBackend(text="   "), InferenceParams(), dpi=120, output_dir=out)
    assert outcome.exit_code != 0

    assert_conforms(
        out,
        [ExpectedDoc(rel_key="blank.pdf", status="failed")],
        require_failures_nonzero_exit=True,
    )
    doc_meta = json.loads((out / "blank" / "metadata.json").read_text())
    assert doc_meta["status"] == "failed"


def test_partial_pdf_conforms_with_partial_status(tmp_path):
    """One page empty, one ok -> partial, recorded, nonzero exit."""
    pdf = tmp_path / "mixed.pdf"
    _make_pdf(pdf, pages=2)
    out = tmp_path / "out"

    class Flaky(FakeBackend):
        def ocr_image(self, image_path, params):
            self.calls += 1
            return "good content" if self.calls == 1 else ""

    outcome = process(pdf, Flaky(), InferenceParams(), dpi=120, output_dir=out)
    assert outcome.exit_code != 0

    assert_conforms(
        out,
        [ExpectedDoc(rel_key="mixed.pdf", pages=2, status="partial")],
        require_failures_nonzero_exit=True,
    )
    doc_meta = json.loads((out / "mixed" / "metadata.json").read_text())
    assert "error" in doc_meta


def test_nested_batch_conforms_no_basename_collision(tmp_path):
    """Two same-basename PDFs in different subdirs both survive (input-relative key)."""
    root = tmp_path / "in"
    (root / "a").mkdir(parents=True)
    (root / "b").mkdir(parents=True)
    _make_pdf(root / "a" / "intro.pdf", pages=1)
    _make_pdf(root / "b" / "intro.pdf", pages=1)
    out = tmp_path / "out"

    outcome = process(root, FakeBackend(), InferenceParams(), dpi=120, output_dir=out)
    assert outcome.exit_code == 0

    assert_conforms(
        out,
        [
            ExpectedDoc(rel_key="a/intro.pdf", pages=1, status="completed"),
            ExpectedDoc(rel_key="b/intro.pdf", pages=1, status="completed"),
        ],
    )


def test_image_dir_document_conforms(tmp_path):
    """A directory of page images is one conforming document keyed by folder name."""
    from PIL import Image

    src = tmp_path / "scan"
    src.mkdir()
    for n in (1, 2):
        Image.new("RGB", (60, 60), "white").save(src / f"page_{n:04d}.png")
    out = tmp_path / "out"

    outcome = process(src, FakeBackend(), InferenceParams(), dpi=120, output_dir=out)
    assert outcome.exit_code == 0

    assert_conforms(
        out,
        [ExpectedDoc(rel_key="scan", pages=2, status="completed")],
    )


def test_mixed_dir_pdf_conforms_pdf_not_dropped(tmp_path):
    """A dir with a PDF + a stray image is a batch: the PDF is OCR'd and conforms.

    Regression guard for the MEDIUM mixed-dir bug: the directory must NOT be
    treated as one bogus image-dir document (which silently dropped the PDF).
    """
    from PIL import Image

    src = tmp_path / "mixed"
    src.mkdir()
    _make_pdf(src / "real.pdf", pages=2)
    Image.new("RGB", (60, 60), "white").save(src / "cover.png")
    out = tmp_path / "out"

    outcome = process(src, FakeBackend(), InferenceParams(), dpi=120, output_dir=out)
    assert outcome.exit_code == 0

    # The real PDF is a conforming document under its input-relative key.
    assert_conforms(
        out,
        [ExpectedDoc(rel_key="real.pdf", pages=2, status="completed")],
    )
