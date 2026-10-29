"""Backend availability probes + the shared OCR call, with HTTP mocked."""

from __future__ import annotations

import base64

import httpx
import pytest
from PIL import Image

from qwen_ocr import config
from qwen_ocr.backends.api import ApiBackend
from qwen_ocr.backends.base import (
    OcrImageResult,
    _encode_image,
    _extract_finish_reason,
    _extract_text,
    _extract_text_and_finish_reason,
    _extract_usage,
)
from qwen_ocr.backends.ollama import OllamaBackend
from qwen_ocr.backends.vllm import VLLMBackend
from qwen_ocr.config import InferenceParams
from qwen_ocr.utils import smart_resize

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


def _decoded_size(url):
    from io import BytesIO

    raw = base64.b64decode(url.split(",", 1)[1])
    with Image.open(BytesIO(raw)) as got:
        return got.size


def test_encode_image_downscales_into_budget(tmp_path):
    # Issue #5: a 300-DPI A4 page (~8.7 MP) must arrive inside the cookbook budget.
    img = tmp_path / "p.png"
    Image.new("RGB", (2480, 3508), "white").save(img)
    url, metrics = _encode_image(img, config.MIN_PIXELS, config.MAX_PIXELS)
    assert url.startswith("data:image/png;base64,")
    assert metrics.sent_width * metrics.sent_height <= config.MAX_PIXELS
    w, h = _decoded_size(url)
    assert w * h <= config.MAX_PIXELS
    assert w % config.PATCH_FACTOR == 0 and h % config.PATCH_FACTOR == 0
    assert abs((w / h) - (2480 / 3508)) < 0.02  # aspect ratio preserved


def test_encode_image_upscales_tiny_page(tmp_path):
    # Below min_pixels the processor would pad an under-resolved page; scale it up.
    img = tmp_path / "p.png"
    Image.new("RGB", (200, 300), "white").save(img)
    url, _ = _encode_image(img, config.MIN_PIXELS, config.MAX_PIXELS)
    w, h = _decoded_size(url)
    assert w * h >= config.MIN_PIXELS


def test_smart_resize_leaves_in_budget_image_alone():
    size = (1024, 1024)  # 1.05 MP, patch-aligned, inside [min, max]
    assert smart_resize(*size, config.MIN_PIXELS, config.MAX_PIXELS) == size


def test_smart_resize_honours_raised_ceiling():
    # A deliberately raised ceiling must actually buy resolution (dense tables).
    big = config.MAX_PIXELS * 4
    w, h = smart_resize(2480, 3508, config.MIN_PIXELS, big)
    assert config.MAX_PIXELS < w * h <= big


def test_extract_text_happy():
    data = {"choices": [{"message": {"content": "# Title\n\nbody"}, "finish_reason": "stop"}]}
    assert _extract_text(data) == "# Title\n\nbody"
    text, reason = _extract_text_and_finish_reason(data)
    assert text == "# Title\n\nbody" and reason == "stop"


def test_extract_text_empty_with_stop_is_empty():
    data = {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]}
    text, reason = _extract_text_and_finish_reason(data)
    assert text == "" and reason == "stop"


def test_extract_finish_reason_openai_length():
    data = {"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]}
    assert _extract_finish_reason(data) == "length"


def test_extract_finish_reason_ollama_done_reason():
    data = {
        "choices": [{"message": {"content": "partial"}}],
        "done_reason": "length",
    }
    assert _extract_finish_reason(data) == "length"


def test_extract_usage_openai_and_ollama():
    oai = {"usage": {"prompt_tokens": 100, "completion_tokens": 50}, "choices": [{}]}
    u = _extract_usage(oai)
    assert u.prompt_tokens == 100 and u.completion_tokens == 50
    ollama = {"prompt_eval_count": 80, "eval_count": 40, "choices": [{}]}
    u2 = _extract_usage(ollama)
    assert u2.prompt_tokens == 80 and u2.completion_tokens == 40


def test_extract_text_malformed_raises():
    with pytest.raises(ValueError):
        _extract_text({"nope": 1})


def test_encode_image_sent_size_grows_with_max_pixels(tmp_path):
    img = tmp_path / "p.png"
    Image.new("RGB", (2480, 3508), "white").save(img)
    low_url, low_m = _encode_image(img, config.MIN_PIXELS, config.MAX_PIXELS)
    high_ceiling = config.MAX_PIXELS * 4
    high_url, high_m = _encode_image(img, config.MIN_PIXELS, high_ceiling)
    assert low_m.sent_megapixels < high_m.sent_megapixels
    low_area = _decoded_size(low_url)[0] * _decoded_size(low_url)[1]
    high_area = _decoded_size(high_url)[0] * _decoded_size(high_url)[1]
    assert high_area > low_area


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
    assert isinstance(out, OcrImageResult)
    assert out.text == "ok text"
    assert out.finish_reason is None
    assert captured["url"].endswith("/chat/completions")
    assert captured["model"] == "m"


# --- thinking mode (issue #4) ---


def _capture_payload(monkeypatch, backend, tmp_path, params):
    """POST once through `backend` and return the request body."""
    img = tmp_path / "p.png"
    Image.new("RGB", (100, 100), "white").save(img)
    captured = {}

    def fake_post(url, json, headers, timeout):
        captured.update(json)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    backend.ocr_image(img, params)
    return captured


def test_thinking_off_by_default():
    # OCR is transcription, not reasoning: the default must disable thinking.
    assert InferenceParams().enable_thinking is False


def test_vllm_sends_chat_template_kwargs(monkeypatch, tmp_path):
    body = _capture_payload(
        monkeypatch, VLLMBackend(base_url="http://x/v1", model="m"), tmp_path, InferenceParams()
    )
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_thinking_flag_leaves_server_default_alone(monkeypatch, tmp_path):
    body = _capture_payload(
        monkeypatch,
        VLLMBackend(base_url="http://x/v1", model="m"),
        tmp_path,
        InferenceParams(enable_thinking=True),
    )
    assert "chat_template_kwargs" not in body and "think" not in body


def test_truncated_finish_reason_surfaces_in_result(monkeypatch, tmp_path):
    img = tmp_path / "p.png"
    Image.new("RGB", (100, 100), "white").save(img)

    def fake_post(url, json, headers, timeout, **kw):
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "cut off"}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2},
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    out = VLLMBackend(base_url="http://x/v1", model="m").ocr_image(img, InferenceParams())
    assert out.truncated and out.finish_reason == "length"


def test_message_order_image_first(monkeypatch, tmp_path):
    img = tmp_path / "p.png"
    Image.new("RGB", (100, 100), "white").save(img)
    captured = {}

    def fake_post(url, json, headers, timeout, **kw):
        captured["content"] = json["messages"][0]["content"]
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    VLLMBackend(base_url="http://x/v1", model="m").ocr_image(
        img, InferenceParams(message_order="image-first")
    )
    parts = captured["content"]
    assert parts[0]["type"] == "image_url" and parts[1]["type"] == "text"


def test_max_output_tokens_in_request_payload(monkeypatch, tmp_path):
    img = tmp_path / "p.png"
    Image.new("RGB", (100, 100), "white").save(img)
    captured = {}

    def fake_post(url, json, headers, timeout, **kw):
        captured["max_tokens"] = json["max_tokens"]
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    VLLMBackend(base_url="http://x/v1", model="m").ocr_image(
        img, InferenceParams(max_output_tokens=4096)
    )
    assert captured["max_tokens"] == 4096


def test_env_max_pixels_changes_sent_image_size(monkeypatch, tmp_path):
    import importlib

    import qwen_ocr.config as cfg

    img = tmp_path / "p.png"
    Image.new("RGB", (2480, 3508), "white").save(img)
    baseline_max = cfg.MAX_PIXELS
    _, low_metrics = _encode_image(img, cfg.MIN_PIXELS, baseline_max)
    monkeypatch.setenv("QWEN_OCR_MAX_PIXELS", str(baseline_max * 4))
    importlib.reload(cfg)
    _, high_metrics = _encode_image(img, cfg.MIN_PIXELS, cfg.MAX_PIXELS)
    assert high_metrics.sent_megapixels > low_metrics.sent_megapixels
    monkeypatch.delenv("QWEN_OCR_MAX_PIXELS", raising=False)
    importlib.reload(cfg)


def test_ollama_path_uses_same_encode_budget(monkeypatch, tmp_path):
    # Ollama inherits base.ocr_image; pixel budget must apply (issue #5 / audit item 5).
    img = tmp_path / "p.png"
    Image.new("RGB", (2480, 3508), "white").save(img)
    captured = {}

    def fake_post(url, json, headers, timeout, **kw):
        captured["url"] = json["messages"][0]["content"][1]["image_url"]["url"]
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    b = OllamaBackend(base_url="http://x/v1", model="m")
    b._resolved_model = "m"
    custom_max = config.MAX_PIXELS * 2
    b.ocr_image(img, InferenceParams(max_pixels=custom_max))
    w, h = _decoded_size(captured["url"])
    assert w * h <= custom_max


def test_ollama_sends_think_false(monkeypatch, tmp_path):
    # Ollama ignores chat_template_kwargs; it gates reasoning on `think`.
    b = OllamaBackend(base_url="http://x/v1", model="m")
    b._resolved_model = "m:8b"  # skip the probe
    body = _capture_payload(monkeypatch, b, tmp_path, InferenceParams())
    assert body["think"] is False
    assert "chat_template_kwargs" not in body


def test_dashscope_sends_both_switches(monkeypatch, tmp_path):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    body = _capture_payload(monkeypatch, ApiBackend(), tmp_path, InferenceParams())
    assert body["enable_thinking"] is False
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


# --- 5xx body + single retry (socr#1020) ---


class _RetryingBackend(VLLMBackend):
    """Stand-in for a free local backend (the policy flag Ollama sets)."""

    retry_on_5xx = True


def _backend(url="http://x/v1", retry=True):
    cls = _RetryingBackend if retry else VLLMBackend
    return cls(base_url=url, model="m")


def _mock_post(monkeypatch, handler):
    """Route httpx.post through a MockTransport; return the list of seen requests."""
    seen = []

    def wrapped(request):
        seen.append(request)
        return handler(request)

    def fake_post(url, json, headers, timeout, **kw):
        with httpx.Client(transport=httpx.MockTransport(wrapped)) as c:
            return c.post(url, json=json, headers=headers, timeout=timeout, **kw)

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr("qwen_ocr.backends.base.SERVER_ERROR_RETRY_DELAY_S", 0.0)
    return seen


def _page(tmp_path):
    img = tmp_path / "p.png"
    Image.new("RGB", (100, 100), "white").save(img)
    return img


def test_500_then_200_retries_once(monkeypatch, tmp_path):
    replies = iter(
        [
            httpx.Response(500, text="boom"),
            httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}),
        ]
    )
    seen = _mock_post(monkeypatch, lambda r: next(replies))
    out = _backend().ocr_image(_page(tmp_path), InferenceParams())
    assert out.text == "ok"
    assert len(seen) == 2
    assert seen[0].content == seen[1].content  # same request, same model


def test_500_then_500_raises_with_body(monkeypatch, tmp_path):
    seen = _mock_post(monkeypatch, lambda r: httpx.Response(500, text="model runner crashed " * 40))
    with pytest.raises(httpx.HTTPStatusError) as ei:
        _backend("http://user:pw@x/v1").ocr_image(_page(tmp_path), InferenceParams())
    assert "model runner crashed" in str(ei.value)
    assert "pw" not in str(ei.value)
    assert len(seen) == 2


def test_userinfo_sent_as_auth_not_in_url(monkeypatch, tmp_path):
    ok = httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
    seen = _mock_post(monkeypatch, lambda r: ok)
    _backend("http://user:pw@x/v1").ocr_image(_page(tmp_path), InferenceParams())
    assert "user" not in str(seen[0].url) and "pw" not in str(seen[0].url)
    assert seen[0].headers["authorization"].startswith("Basic ")


def test_404_does_not_retry_and_is_redacted(monkeypatch, tmp_path):
    seen = _mock_post(monkeypatch, lambda r: httpx.Response(404, text="bad password=hunter2"))
    with pytest.raises(httpx.HTTPStatusError) as ei:
        _backend().ocr_image(_page(tmp_path), InferenceParams())
    assert len(seen) == 1
    assert "hunter2" not in str(ei.value)


def test_timeout_does_not_retry(monkeypatch, tmp_path):
    def boom(request):
        raise httpx.ReadTimeout("slow", request=request)

    seen = _mock_post(monkeypatch, boom)
    with pytest.raises(httpx.ReadTimeout):
        _backend().ocr_image(_page(tmp_path), InferenceParams())
    assert len(seen) == 1


def test_cloud_backend_does_not_retry(monkeypatch, tmp_path):
    seen = _mock_post(monkeypatch, lambda r: httpx.Response(500, text="boom"))
    with pytest.raises(httpx.HTTPStatusError):
        _backend(retry=False).ocr_image(_page(tmp_path), InferenceParams())
    assert len(seen) == 1
    assert ApiBackend.retry_on_5xx is False and OllamaBackend.retry_on_5xx is True


def test_retry_skipped_when_deadline_exhausted(monkeypatch, tmp_path):
    clock = {"t": 0.0}
    monkeypatch.setattr("qwen_ocr.backends.base.time.monotonic", lambda: clock["t"])

    def slow_500(request):
        clock["t"] += 100.0  # the first attempt eats the whole budget
        return httpx.Response(500, text="boom")

    seen = _mock_post(monkeypatch, slow_500)
    with pytest.raises(httpx.HTTPStatusError):
        _backend().ocr_image(_page(tmp_path), InferenceParams(timeout=100.0))
    assert len(seen) == 1


def test_retry_gets_remaining_budget_after_sleep(monkeypatch, tmp_path):
    clock = {"t": 0.0}
    monkeypatch.setattr("qwen_ocr.backends.base.time.monotonic", lambda: clock["t"])
    monkeypatch.setattr(
        "qwen_ocr.backends.base.time.sleep", lambda s: clock.update(t=clock["t"] + s)
    )
    timeouts = []
    replies = iter([500, 200])

    def fake_post(url, json, headers, timeout, **kw):
        timeouts.append(timeout)
        clock["t"] += 10.0
        code = next(replies)
        body = {"choices": [{"message": {"content": "ok"}}]} if code == 200 else None
        return httpx.Response(
            code, json=body, text="x" if body is None else None, request=httpx.Request("POST", url)
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr("qwen_ocr.backends.base.SERVER_ERROR_RETRY_DELAY_S", 5.0)
    _backend().ocr_image(_page(tmp_path), InferenceParams(timeout=100.0))
    assert timeouts == [100.0, 100.0 - 10.0 - 5.0]


def test_redaction_precedes_truncation():
    from qwen_ocr.backends.base import ERROR_BODY_CHARS, redact_credentials

    key = "sk-SECRETSECRETSECRET"
    body = "x" * (ERROR_BODY_CHARS - 5) + key  # key straddles the cut
    out = redact_credentials(body, (key,))[:ERROR_BODY_CHARS]
    assert "SECRET" not in out and "sk-" not in out


def test_redaction_patterns():
    from qwen_ocr.backends.base import redact_credentials

    s = 'password=ab token: "x1" Authorization: Bearer abc.def http://u:p@h/ echo sk-bare'
    out = redact_credentials(s, ("sk-bare",))
    for leaked in ("ab ", "x1", "abc.def", "u:p", "sk-bare"):
        assert leaked not in out


def test_logs_leak_no_credentials(monkeypatch, tmp_path, caplog):
    import logging

    caplog.set_level(logging.DEBUG)
    _mock_post(monkeypatch, lambda r: httpx.Response(500, text="boom"))
    with pytest.raises(httpx.HTTPStatusError) as ei:
        _backend("http://user:pw@x/v1").ocr_image(_page(tmp_path), InferenceParams())
    assert "pw" not in caplog.text and "user:" not in caplog.text
    assert "pw" not in str(ei.value)


def test_302_raises_and_is_not_parsed(monkeypatch, tmp_path):
    seen = _mock_post(
        monkeypatch,
        lambda r: httpx.Response(302, json={"choices": [{"message": {"content": "NOT OCR"}}]}),
    )
    with pytest.raises(httpx.HTTPStatusError):
        _backend().ocr_image(_page(tmp_path), InferenceParams())
    assert len(seen) == 1


def test_redaction_quoted_value_with_spaces_and_header_lines():
    from qwen_ocr.backends.base import redact_credentials

    s = (
        "password=\"my secret pass\" token='a b c'\n"
        "Authorization: Basic dXNlcjpwdw==\nAuthorization: Bearer abc.def ghi\n"
        "proxy-authorization: Digest realm=x"
    )
    out = redact_credentials(s)
    for leaked in ("secret pass", "b c", "dXNlcjpwdw", "ghi", "Digest", "abc.def"):
        assert leaked not in out


def test_exact_secret_does_not_leave_token_remainder():
    from qwen_ocr.backends.base import redact_credentials

    key = "sk-AAA"
    out = redact_credentials(f"api_key={key}BBBCCC and bare {key}", (key,))
    assert "BBBCCC" not in out and key not in out


def test_redaction_escaped_quote_in_quoted_value():
    from qwen_ocr.backends.base import redact_credentials

    dq = r'{"password":"prefix\"suffix"}'
    sq = r"{'token':'prefix\'suffix'}"
    assert "suffix" not in redact_credentials(dq)
    assert "suffix" not in redact_credentials(sq)
