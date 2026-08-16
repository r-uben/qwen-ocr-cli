"""Processor IO: aggregate pages into one <stem>/<stem>.md, skip + trim + status."""

from __future__ import annotations

import json

import fitz
from ocr_output_contract import UNREADABLE_CHECKSUM, Status
from PIL import Image

from qwen_ocr.backends.base import Availability, Backend
from qwen_ocr.config import InferenceParams
from qwen_ocr.processor import process
from qwen_ocr.utils import strip_thinking, trim_degenerate_tail


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


def test_mixed_dir_pdf_and_stray_image_processes_pdf(tmp_path):
    # A dir holding a PDF + a loose top-level image is a BATCH tree: the PDF must
    # be OCR'd, not silently folded into one bogus image-dir document (regression
    # guard for the MEDIUM mixed-dir data-loss bug).
    src = tmp_path / "mixed"
    src.mkdir()
    _make_pdf(src / "real.pdf", pages=1)
    Image.new("RGB", (50, 50), "white").save(src / "cover.png")
    out = tmp_path / "out"

    backend = FakeBackend("body")
    outcome = process(src, backend, InferenceParams(), dpi=120, output_dir=out)

    assert outcome.exit_code == 0
    # The PDF was processed to its own <stem>/<stem>.md ...
    assert (out / "real" / "real.md").exists()
    # ... and the dir was NOT treated as a single image-dir document.
    assert not (out / "mixed" / "mixed.md").exists()
    root = json.loads((out / "metadata.json").read_text())
    assert "real.pdf" in root["files"]


def test_pure_image_dir_still_one_document(tmp_path):
    # A directory of ONLY page images is still one image-dir document (socr path).
    src = tmp_path / "scan"
    src.mkdir()
    for n in (1, 2):
        Image.new("RGB", (50, 50), "white").save(src / f"page_{n:04d}.png")
    out = tmp_path / "out"

    outcome = process(src, FakeBackend("# ok"), InferenceParams(), dpi=120, output_dir=out)
    assert outcome.exit_code == 0
    assert (out / "scan" / "scan.md").exists()


def test_page_image_dir_ignores_os_junk(tmp_path):
    # A .DS_Store next to page images must not demote the dir out of image-doc mode.
    src = tmp_path / "scan"
    src.mkdir()
    for n in (1, 2):
        Image.new("RGB", (50, 50), "white").save(src / f"page_{n:04d}.png")
    (src / ".DS_Store").write_bytes(b"\x00\x01")
    out = tmp_path / "out"

    outcome = process(src, FakeBackend("# ok"), InferenceParams(), dpi=120, output_dir=out)
    assert outcome.exit_code == 0
    assert (out / "scan" / "scan.md").exists()


def test_default_output_root_not_self_ingested_on_rerun(tmp_path):
    # The default <input>/ocr/ root sits inside the scanned tree; a re-run must not
    # re-discover its own outputs as fresh inputs (iter_input_files prunes it).
    src = tmp_path / "papers"
    src.mkdir()
    _make_pdf(src / "a.pdf", pages=1)

    b1 = FakeBackend("x")
    out1 = process(src, b1, InferenceParams(), dpi=120)  # default -> papers/ocr/
    assert out1.completed == 1
    assert (src / "ocr" / "a" / "a.md").exists()

    # Second run: only the one PDF is rediscovered (skipped), no stray .md inputs.
    b2 = FakeBackend("x")
    out2 = process(src, b2, InferenceParams(), dpi=120)
    assert b2.calls == 0
    assert out2.completed == 1  # exactly the one real input, not its outputs


def test_quiet_skip_emits_existing_md_path(tmp_path):
    # Cached-skip must emit the existing .md path (quiet-mode scripting contract).
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, pages=1)
    out = tmp_path / "out"

    first = process(pdf, FakeBackend("hi"), InferenceParams(), dpi=120, output_dir=out)
    assert first.outputs == [str(out / "doc" / "doc.md")]

    # Re-run: skipped, but the existing .md path is still emitted (not empty).
    b2 = FakeBackend("hi")
    second = process(pdf, b2, InferenceParams(), dpi=120, output_dir=out)
    assert b2.calls == 0
    assert second.completed == 1
    assert second.outputs == [str(out / "doc" / "doc.md")]


def test_deleted_md_forces_reprocess(tmp_path):
    # v0.1.1 is_completed verifies the .md on disk: a deleted output is re-emitted.
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, pages=1)
    out = tmp_path / "out"

    process(pdf, FakeBackend("hi"), InferenceParams(), dpi=120, output_dir=out)
    (out / "doc" / "doc.md").unlink()  # index survives, .md gone

    b2 = FakeBackend("again")
    process(pdf, b2, InferenceParams(), dpi=120, output_dir=out)
    assert b2.calls == 1  # NOT skipped despite the surviving index entry
    assert (out / "doc" / "doc.md").exists()


def test_model_change_invalidates_cache(tmp_path):
    # Run fingerprint: a re-run under a different model reprocesses (no stale reuse).
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, pages=1)
    out = tmp_path / "out"

    process(pdf, FakeBackend("v1"), InferenceParams(), dpi=120, output_dir=out)

    class OtherModel(FakeBackend):
        def _endpoint(self):
            return ("http://x/v1", None, "different-model")

    b2 = OtherModel("v2")
    outcome = process(pdf, b2, InferenceParams(), dpi=120, output_dir=out)
    assert b2.calls == 1  # different model -> reprocessed, not skipped
    assert outcome.completed == 1


def test_dpi_change_invalidates_cache(tmp_path):
    # v0.1.2: --dpi is an output-affecting flag folded into the run fingerprint,
    # so re-rendering the same PDF at a different DPI reprocesses (no stale reuse).
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, pages=1)
    out = tmp_path / "out"

    process(pdf, FakeBackend("v1"), InferenceParams(), dpi=120, output_dir=out)

    b2 = FakeBackend("v2")
    process(pdf, b2, InferenceParams(), dpi=120, output_dir=out)
    assert b2.calls == 0  # same dpi -> skipped

    b3 = FakeBackend("v3")
    process(pdf, b3, InferenceParams(), dpi=300, output_dir=out)
    assert b3.calls == 1  # different dpi -> reprocessed, not skipped


def test_inference_params_change_invalidates_cache(tmp_path):
    # Output-affecting inference params (e.g. prompt) are in the fingerprint too.
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, pages=1)
    out = tmp_path / "out"

    process(pdf, FakeBackend("v1"), InferenceParams(), dpi=120, output_dir=out)

    b2 = FakeBackend("v2")
    other = InferenceParams(prompt="a different OCR prompt")
    process(pdf, b2, other, dpi=120, output_dir=out)
    assert b2.calls == 1  # changed prompt -> reprocessed


def test_unreadable_input_recorded_failed_batch_continues(tmp_path):
    # SYS-02 via v0.1.2 safe_checksum: an input unreadable at the idempotency
    # pre-check is recorded status=failed and the batch CONTINUES (no abort).
    import os
    import stat

    root = tmp_path / "in"
    root.mkdir()
    _make_pdf(root / "a.pdf", pages=1)
    bad = root / "b.pdf"
    _make_pdf(bad, pages=1)
    _make_pdf(root / "c.pdf", pages=1)
    out = tmp_path / "out"

    os.chmod(bad, 0)  # unreadable at checksum time
    try:
        # Capture whether chmod(0) actually denied reads in THIS run (it does not
        # when running as root, e.g. some CI), before permissions are restored.
        bad_was_unreadable = not os.access(bad, os.R_OK)
        outcome = process(root, FakeBackend("x"), InferenceParams(), dpi=120, output_dir=out)
    finally:
        os.chmod(bad, stat.S_IRUSR | stat.S_IWUSR)  # restore for cleanup

    # The two good files were processed; the bad one is recorded failed, no abort.
    assert outcome.completed == 2
    assert outcome.failed == 1
    assert outcome.exit_code != 0
    assert (out / "a" / "a.md").exists()
    assert (out / "c" / "c.md").exists()
    bad_meta = json.loads((out / "b" / "metadata.json").read_text())
    assert bad_meta["status"] == "failed"
    assert "unreadable" in (bad_meta["error"] or "")
    # v0.1.3: a failure record must carry a VALID ``sha256:`` checksum (the
    # conformance harness rejects None/""), so the unreadable-input fallback is
    # the ``sha256:`` UNREADABLE_CHECKSUM sentinel.
    assert bad_meta["checksum"].startswith("sha256:")
    if bad_was_unreadable:
        assert bad_meta["checksum"] == UNREADABLE_CHECKSUM


def test_image_dir_default_output_idempotent_on_rerun(tmp_path):
    # The default <input>/ocr/ root lands INSIDE the scanned image dir; a re-run
    # must still classify it as one image-dir document (not demote it to a batch
    # tree that finds no .pdf and raises "no documents found").
    src = tmp_path / "scan"
    src.mkdir()
    for n in (1, 2):
        Image.new("RGB", (40, 40), "white").save(src / f"page_{n:04d}.png")

    b1 = FakeBackend("ok")
    out1 = process(src, b1, InferenceParams(), dpi=120)  # default -> scan/ocr/
    assert out1.completed == 1
    assert (src / "ocr" / "scan" / "scan.md").exists()

    # Second run: the freshly-created ocr/ subdir must NOT demote the scan to a
    # batch tree; the same image-dir document is rediscovered and skipped.
    b2 = FakeBackend("ok")
    out2 = process(src, b2, InferenceParams(), dpi=120)
    assert b2.calls == 0  # skipped (already completed), not re-OCR'd
    assert out2.completed == 1  # not "no documents found"


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


def test_strip_thinking_removes_reasoning_block():
    # Issue #4: a server that ignores the switch must not write reasoning into the body.
    assert strip_thinking("<think>I should transcribe...</think>\n# Title") == "# Title"
    # Unmatched/absent tags leave real page text alone.
    body = "The value of <think> in the model is discussed"
    assert strip_thinking(body) == body


def test_page_text_is_stripped_of_thinking(tmp_path):
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, pages=1)
    out = tmp_path / "out"
    process(
        pdf,
        FakeBackend("<think>let me look</think>real page text"),
        InferenceParams(),
        dpi=72,
        output_dir=out,
    )
    body = (out / "doc" / "doc.md").read_text()
    assert "let me look" not in body
    assert "real page text" in body
