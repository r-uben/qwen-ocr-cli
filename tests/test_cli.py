"""CLI surface — version, backends listing, process wiring (backend mocked)."""

from __future__ import annotations

from click.testing import CliRunner
from PIL import Image

from qwen_ocr import cli
from qwen_ocr.backends.base import Availability, Backend


class FakeBackend(Backend):
    name = "fake"

    def availability(self):
        return Availability(True)

    def _endpoint(self):
        return ("http://x/v1", None, "m")

    def ocr_image(self, image_path, params):
        return "ocr text"


def test_version():
    res = CliRunner().invoke(cli.main, ["--version"])
    assert res.exit_code == 0
    assert "qwen-ocr" in res.output


def test_process_requires_process_subcommand(tmp_path):
    # socr always calls the explicit `process` subcommand.
    res = CliRunner().invoke(cli.main, ["--help"])
    assert "process" in res.output
    assert "backends" in res.output


def test_backends_listing(monkeypatch):
    monkeypatch.setattr(
        cli, "probe_all",
        lambda: {"ollama": Availability(True), "api": Availability(False, "no key")},
    )
    res = CliRunner().invoke(cli.main, ["backends"])
    assert res.exit_code == 0
    assert "[+] ollama" in res.output
    assert "[x] api" in res.output and "no key" in res.output


def test_process_writes_output(monkeypatch, tmp_path):
    src = tmp_path / "imgs"
    src.mkdir()
    Image.new("RGB", (40, 40), "white").save(src / "page.png")
    out = tmp_path / "out"

    monkeypatch.setattr(cli, "make_backend", lambda name: FakeBackend())

    res = CliRunner().invoke(
        cli.main, ["process", str(src), "-o", str(out), "--backend", "ollama"]
    )
    assert res.exit_code == 0, res.output
    assert (out / "page" / "page.md").read_text() == "ocr text"


def test_explicit_backend_unavailable_errors(monkeypatch, tmp_path):
    src = tmp_path / "imgs"
    src.mkdir()
    Image.new("RGB", (40, 40), "white").save(src / "p.png")

    class Dead(FakeBackend):
        def availability(self):
            return Availability(False, "server down")

    monkeypatch.setattr(cli, "make_backend", lambda name: Dead())
    res = CliRunner().invoke(
        cli.main, ["process", str(src), "-o", str(tmp_path / "o"), "--backend", "vllm"]
    )
    assert res.exit_code != 0
    assert "server down" in res.output
