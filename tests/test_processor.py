"""Processor IO: render/read pages, write {out}/{stem}/{stem}.md, skip + trim."""

from __future__ import annotations

import fitz
from PIL import Image

from qwen_ocr.backends.base import Backend
from qwen_ocr.config import InferenceParams
from qwen_ocr.processor import process
from qwen_ocr.utils import trim_degenerate_tail


class FakeBackend(Backend):
    name = "fake"

    def __init__(self, text="hello world"):
        self.text = text
        self.calls = 0

    def availability(self):
        from qwen_ocr.backends.base import Availability
        return Availability(True)

    def _endpoint(self):
        return ("http://x/v1", None, "m")

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


def test_image_dir_writes_per_page_layout(tmp_path):
    src = tmp_path / "imgs"
    src.mkdir()
    for n in (1, 2, 3):
        Image.new("RGB", (50, 50), "white").save(src / f"page_{n:04d}.png")
    out = tmp_path / "out"

    backend = FakeBackend("# ok")
    results = process(src, out, backend, InferenceParams(), dpi=200)

    assert backend.calls == 3
    assert all(r.ok for r in results)
    md = out / "page_0001" / "page_0001.md"
    assert md.exists() and md.read_text() == "# ok"


def test_pdf_is_rendered_and_ocrd(tmp_path):
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, pages=2)
    out = tmp_path / "out"

    backend = FakeBackend("text")
    process(pdf, out, backend, InferenceParams(), dpi=150)

    assert backend.calls == 2
    assert (out / "doc_p0001" / "doc_p0001.md").exists()
    assert (out / "doc_p0002" / "doc_p0002.md").exists()


def test_single_page_pdf_uses_bare_stem(tmp_path):
    pdf = tmp_path / "one.pdf"
    _make_pdf(pdf, pages=1)
    out = tmp_path / "out"
    process(pdf, out, FakeBackend(), InferenceParams(), dpi=150)
    assert (out / "one" / "one.md").exists()


def test_skip_existing_unless_reprocess(tmp_path):
    src = tmp_path / "imgs"
    src.mkdir()
    Image.new("RGB", (50, 50), "white").save(src / "p.png")
    out = tmp_path / "out"

    b1 = FakeBackend("first")
    process(src, out, b1, InferenceParams(), dpi=200)
    assert (out / "p" / "p.md").read_text() == "first"

    b2 = FakeBackend("second")
    process(src, out, b2, InferenceParams(), dpi=200)  # exists → skipped
    assert b2.calls == 0
    assert (out / "p" / "p.md").read_text() == "first"

    b3 = FakeBackend("third")
    process(src, out, b3, InferenceParams(), dpi=200, reprocess=True)
    assert b3.calls == 1
    assert (out / "p" / "p.md").read_text() == "third"


def test_backend_error_is_isolated_per_page(tmp_path):
    src = tmp_path / "imgs"
    src.mkdir()
    Image.new("RGB", (50, 50), "white").save(src / "a.png")

    class Boom(FakeBackend):
        def ocr_image(self, image_path, params):
            raise RuntimeError("model exploded")

    results = process(src, tmp_path / "out", Boom(), InferenceParams(), dpi=200)
    assert len(results) == 1 and not results[0].ok
    assert "model exploded" in results[0].error


def test_trim_degenerate_tail():
    good = "line a\nline b\nline c"
    assert trim_degenerate_tail(good) == good

    loop = "real text\n" + "\n".join(["spam"] * 10)
    out = trim_degenerate_tail(loop)
    assert out.count("spam") == 1
    assert "real text" in out
