"""Backend availability probes + the shared OCR call, with HTTP mocked."""

from __future__ import annotations

import base64

import httpx
import pytest
from PIL import Image

from qwen_ocr.backends.api import ApiBackend
from qwen_ocr.backends.base import _encode_image, _extract_text
from qwen_ocr.backends.ollama import OllamaBackend
from qwen_ocr.backends.vllm import VLLMBackend
from qwen_ocr.config import InferenceParams

# --- availability probes ---


def test_vllm_unavailable_without_url(monkeypatch):
    b = VLLMBackend(base_url="")
    assert not b.is_available()
    assert "VLLM_BASE_URL" in b.availability().reason


def test_api_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    b = ApiBackend()
    assert not b.is_available()


def test_api_prefers_dashscope(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "k2")
    b = ApiBackend()
    assert b.provider == "dashscope"
    base_url, key, _ = b._endpoint()
    assert key == "k1" and "dashscope" in base_url


def test_ollama_reports_missing_model(monkeypatch):
    def fake_get(url, timeout):
        return httpx.Response(
            200,
            json={"models": [{"name": "llama3:latest"}]},
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(httpx, "get", fake_get)
    b = OllamaBackend(model="qwen3-vl")
    avail = b.availability()
    assert not avail.ok and "not pulled" in avail.reason


def test_ollama_matches_model_with_tag(monkeypatch):
    def fake_get(url, timeout):
        return httpx.Response(
            200,
            json={"models": [{"name": "qwen3-vl:latest"}]},
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(httpx, "get", fake_get)
    assert OllamaBackend(model="qwen3-vl").is_available()


def test_ollama_endpoint_sends_resolved_tag(monkeypatch):
    # Regression: bare "qwen3-vl" must be sent as the concrete installed tag
    # ("qwen3-vl:8b"), else Ollama 404s on an unresolvable model.
    def fake_get(url, timeout):
        return httpx.Response(
            200,
            json={"models": [{"name": "qwen3-vl:8b"}]},
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(httpx, "get", fake_get)
    b = OllamaBackend(model="qwen3-vl")
    _, _, model = b._endpoint()  # lazily probes + resolves
    assert model == "qwen3-vl:8b"


# --- image encoding + response parsing ---


def test_encode_image_downscales(tmp_path):
    from io import BytesIO

    img = tmp_path / "p.png"
    Image.new("RGB", (8000, 100), "white").save(img)
    url = _encode_image(img, max_side=4000)
    assert url.startswith("data:image/png;base64,")
    raw = base64.b64decode(url.split(",", 1)[1])
    # decoded image's longest side must be capped at max_side
    with Image.open(BytesIO(raw)) as got:
        assert max(got.size) == 4000


def test_extract_text_happy():
    data = {"choices": [{"message": {"content": "# Title\n\nbody"}}]}
    assert _extract_text(data) == "# Title\n\nbody"


def test_extract_text_malformed_raises():
    with pytest.raises(ValueError):
        _extract_text({"nope": 1})


def test_ocr_image_posts_and_parses(monkeypatch, tmp_path):
    img = tmp_path / "p.png"
    Image.new("RGB", (100, 100), "white").save(img)

    captured = {}

    def fake_post(url, json, headers, timeout):
        captured["url"] = url
        captured["model"] = json["model"]
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok text"}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    out = VLLMBackend(base_url="http://x/v1", model="m").ocr_image(img, InferenceParams())
    assert out == "ok text"
    assert captured["url"].endswith("/chat/completions")
    assert captured["model"] == "m"
