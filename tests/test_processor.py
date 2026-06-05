"""Processor IO: aggregate pages into one <stem>/<stem>.md, skip + trim + status."""

from __future__ import annotations

import json

import fitz
from ocr_output_contract import Status
from PIL import Image

from qwen_ocr.backends.base import Availability, Backend
from qwen_ocr.config import InferenceParams
from qwen_ocr.processor import process
from qwen_ocr.utils import trim_degenerate_tail


class FakeBackend(Backend):
    name = "fake"

    def __init__(self, text="hello world"):
        self.text = text
        self.calls = 0

    def availability(self):
        return Availability(True)

    def _endpoint(self):
        return ("http://x/v1", None, "fake-model")

    def ocr_image(self, image_path, params):
        self.calls += 1
        return self.text


def _make_pdf(path, pages=2):
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"page {i + 1}")
    doc.save(path)
    doc.close()


def test_multipage_pdf_aggregates_to_one_md(tmp_path):
    # QWEN-01: a 2-page PDF must produce ONE <stem>/<stem>.md, not a folder/page.
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, pages=2)
    out = tmp_path / "out"

    backend = FakeBackend("text")
    outcome = process(pdf, backend, InferenceParams(), dpi=150, output_dir=out)

    assert backend.calls == 2
    assert outcome.exit_code == 0
    md = out / "doc" / "doc.md"
    assert md.exists()
    body = md.read_text()
    assert "## Page 1" in body and "## Page 2" in body
    # No per-page folders.
    assert not (out / "doc_p0001").exists()
    assert not (out / "doc_p0002").exists()


def test_image_dir_is_one_document(tmp_path):
    src = tmp_path / "imgs"
    src.mkdir()
    for n in (1, 2, 3):
        Image.new("RGB", (50, 50), "white").save(src / f"page_{n:04d}.png")
    out = tmp_path / "out"

    backend = FakeBackend("# ok")
    outcome = process(src, backend, InferenceParams(), dpi=200, output_dir=out)

    assert backend.calls == 3
    assert outcome.exit_code == 0
    md = out / "imgs" / "imgs.md"
    assert md.exists()
    body = md.read_text()
    assert body.count("## Page ") == 3


def test_default_output_root_is_input_parent_ocr(tmp_path):
    # No -o: default is <input-parent>/ocr/ (never required).
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, pages=1)
    process(pdf, FakeBackend(), InferenceParams(), dpi=120)
    assert (tmp_path / "ocr" / "doc" / "doc.md").exists()
    assert (tmp_path / "ocr" / "metadata.json").exists()


def test_metadata_both_levels_written(tmp_path):
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, pages=2)
    out = tmp_path / "out"
    process(pdf, FakeBackend(), InferenceParams(), dpi=120, output_dir=out)

    doc_meta = json.loads((out / "doc" / "metadata.json").read_text())
    assert doc_meta["status"] == "completed"
    assert doc_meta["backend"] == "fake"
    assert doc_meta["model"] == "fake-model"
    assert doc_meta["pages"] == 2
    assert doc_meta["output_path"] == "doc/doc.md"
    assert doc_meta["checksum"].startswith("sha256:")

    root = json.loads((out / "metadata.json").read_text())
    assert root["files"]["doc.pdf"]["status"] == "completed"


def test_skip_existing_unless_reprocess(tmp_path):
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, pages=1)
    out = tmp_path / "out"

    b1 = FakeBackend("first")
    process(pdf, b1, InferenceParams(), dpi=120, output_dir=out)
    assert b1.calls == 1

    b2 = FakeBackend("second")
    process(pdf, b2, InferenceParams(), dpi=120, output_dir=out)  # already completed → skip
    assert b2.calls == 0

    b3 = FakeBackend("third")
    process(pdf, b3, InferenceParams(), dpi=120, output_dir=out, reprocess=True)
    assert b3.calls == 1


def test_empty_response_is_failure_not_silent_success(tmp_path):
    # QWEN-02: an empty model response is a FAILURE, not a 0-byte "ok".
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, pages=1)
    out = tmp_path / "out"

    outcome = process(pdf, FakeBackend(""), InferenceParams(), dpi=120, output_dir=out)
    assert outcome.exit_code != 0
    assert outcome.failed == 1

    doc_meta = json.loads((out / "doc" / "metadata.json").read_text())
    assert doc_meta["status"] == Status.FAILED.value
    assert "empty OCR response" in doc_meta["error"]


def test_partial_failure_exits_nonzero(tmp_path):
    # One page empty, one page ok → partial → nonzero exit (was: silent exit 0).
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, pages=2)
    out = tmp_path / "out"

    class FlakyBackend(FakeBackend):
        def ocr_image(self, image_path, params):
            self.calls += 1
            return "good page" if self.calls == 1 else ""

    outcome = process(pdf, FlakyBackend(), InferenceParams(), dpi=120, output_dir=out)
    assert outcome.exit_code != 0
    assert outcome.partial == 1

    doc_meta = json.loads((out / "doc" / "metadata.json").read_text())
    assert doc_meta["status"] == Status.PARTIAL.value
    assert doc_meta["pages"] == 2


def test_backend_error_is_isolated_per_page(tmp_path):
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, pages=1)

    class Boom(FakeBackend):
        def ocr_image(self, image_path, params):
            raise RuntimeError("model exploded")

    out = tmp_path / "out"
    outcome = process(pdf, Boom(), InferenceParams(), dpi=120, output_dir=out)
    assert outcome.exit_code != 0
    doc_meta = json.loads((out / "doc" / "metadata.json").read_text())
    assert "model exploded" in doc_meta["error"]


def test_trim_degenerate_tail():
    good = "line a\nline b\nline c"
    assert trim_degenerate_tail(good) == good

    loop = "real text\n" + "\n".join(["spam"] * 10)
    out = trim_degenerate_tail(loop)
    assert out.count("spam") == 1
    assert "real text" in out
