"""CLI surface — version, backends listing, process wiring (backend mocked)."""

from __future__ import annotations

import fitz
from click.testing import CliRunner

from qwen_ocr import cli
from qwen_ocr.backends.base import Availability, Backend


class FakeBackend(Backend):
    name = "fake"

    def __init__(self, text="ocr text"):
        self.text = text

    def availability(self):
        return Availability(True)

    def _endpoint(self):
        return ("http://x/v1", None, "fake-model")

    def ocr_image(self, image_path, params):
        return self.text


def _make_pdf(path, pages=1):
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"page {i + 1}")
    doc.save(path)
    doc.close()


def test_version():
    res = CliRunner().invoke(cli.main, ["--version"])
    assert res.exit_code == 0
    assert "qwen-ocr" in res.output


def test_process_subcommand_listed():
    res = CliRunner().invoke(cli.main, ["--help"])
    assert "process" in res.output
    assert "backends" in res.output


def test_backends_listing(monkeypatch):
    monkeypatch.setattr(
        cli,
        "probe_all",
        lambda: {"ollama": Availability(True), "api": Availability(False, "no key")},
    )
    res = CliRunner().invoke(cli.main, ["backends"])
    assert res.exit_code == 0
    assert "[+] ollama" in res.output
    assert "[x] api" in res.output and "no key" in res.output


def test_process_writes_canonical_output(monkeypatch, tmp_path):
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, pages=2)
    out = tmp_path / "out"

    monkeypatch.setattr(cli, "make_backend", lambda name, model=None: FakeBackend())

    res = CliRunner().invoke(cli.main, ["process", str(pdf), "-o", str(out), "--backend", "ollama"])
    assert res.exit_code == 0, res.output
    body = (out / "doc" / "doc.md").read_text()
    assert "## Page 1" in body and "## Page 2" in body


def test_output_not_required_defaults_to_input_parent_ocr(monkeypatch, tmp_path):
    # -o is NOT required (fixes qwen's "-o required" divergence).
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, pages=1)

    monkeypatch.setattr(cli, "make_backend", lambda name, model=None: FakeBackend())

    res = CliRunner().invoke(cli.main, ["process", str(pdf), "--backend", "ollama"])
    assert res.exit_code == 0, res.output
    assert (tmp_path / "ocr" / "doc" / "doc.md").exists()


def test_quiet_emits_output_paths(monkeypatch, tmp_path):
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, pages=1)
    out = tmp_path / "out"

    monkeypatch.setattr(cli, "make_backend", lambda name, model=None: FakeBackend())

    res = CliRunner().invoke(
        cli.main, ["process", str(pdf), "-o", str(out), "--backend", "ollama", "-q"]
    )
    assert res.exit_code == 0, res.output
    assert str(out / "doc" / "doc.md") in res.output


def test_quiet_failure_diagnostics_go_to_stderr(monkeypatch, tmp_path):
    # Under -q, failure diagnostics must still reach stderr (socr reads stderr on
    # a nonzero exit); stdout stays clean (paths only).
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, pages=1)
    out = tmp_path / "out"

    monkeypatch.setattr(cli, "make_backend", lambda name, model=None: FakeBackend(text=""))

    res = CliRunner().invoke(
        cli.main, ["process", str(pdf), "-o", str(out), "--backend", "ollama", "-q"]
    )
    assert res.exit_code != 0
    assert "failed" in res.stderr  # the failure summary reached stderr under -q
    assert "failed" not in res.stdout  # stdout stays clean for scripting


def test_empty_response_exits_nonzero(monkeypatch, tmp_path):
    # QWEN-02 at the CLI boundary: an empty page → nonzero exit.
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, pages=1)
    out = tmp_path / "out"

    monkeypatch.setattr(cli, "make_backend", lambda name, model=None: FakeBackend(text=""))

    res = CliRunner().invoke(cli.main, ["process", str(pdf), "-o", str(out), "--backend", "ollama"])
    assert res.exit_code != 0


def test_explicit_backend_unavailable_errors(monkeypatch, tmp_path):
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, pages=1)

    class Dead(FakeBackend):
        def availability(self):
            return Availability(False, "server down")

    monkeypatch.setattr(cli, "make_backend", lambda name, model=None: Dead())
    res = CliRunner().invoke(
        cli.main, ["process", str(pdf), "-o", str(tmp_path / "o"), "--backend", "vllm"]
    )
    assert res.exit_code != 0
    assert "server down" in res.output
