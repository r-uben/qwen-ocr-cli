"""Backend selection logic — fully mocked, no live servers."""

from __future__ import annotations

import pytest

from qwen_ocr.backends import select
from qwen_ocr.backends.base import Availability


def _force(monkeypatch, live: set[str]):
    """Make each backend report available iff its name is in `live`."""
    for name, factory in select._FACTORIES.items():
        ok = name in live
        monkeypatch.setattr(
            factory,
            "availability",
            lambda self, ok=ok, name=name: Availability(ok, "" if ok else f"{name} down"),
        )


def test_cost_prefers_free_backends_first(monkeypatch):
    _force(monkeypatch, {"ollama", "vllm", "api"})
    assert select.resolve_backend("cost").name == "vllm"  # free, first in cost order


def test_quality_prefers_api_first(monkeypatch):
    _force(monkeypatch, {"ollama", "vllm", "api"})
    assert select.resolve_backend("quality").name == "api"


def test_speed_prefers_ollama_first(monkeypatch):
    _force(monkeypatch, {"ollama", "vllm", "api"})
    assert select.resolve_backend("speed").name == "ollama"


def test_falls_through_to_only_live_backend(monkeypatch):
    _force(monkeypatch, {"api"})
    assert select.resolve_backend("cost").name == "api"


def test_none_available_raises_with_reasons(monkeypatch):
    _force(monkeypatch, set())
    with pytest.raises(RuntimeError) as exc:
        select.resolve_backend("cost")
    msg = str(exc.value)
    assert "ollama down" in msg and "vllm down" in msg and "api down" in msg


def test_unknown_preference_rejected(monkeypatch):
    with pytest.raises(ValueError):
        select.resolve_backend("cheapest")


def test_make_backend_unknown_name():
    with pytest.raises(ValueError):
        select.make_backend("tesseract")
