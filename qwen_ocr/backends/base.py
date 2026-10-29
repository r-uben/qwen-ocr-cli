"""Backend protocol + a shared OpenAI-compatible vision-chat implementation.

All three backends (Ollama, vLLM, cloud API) speak the OpenAI chat-completions API with
an image passed as a base64 data URL. They differ only in base URL, auth, model id, and how
availability is probed. So the actual OCR call lives once here; subclasses are thin.

`availability()` must NEVER raise — a dead backend returns Availability(ok=False, reason=...),
so the selector can probe every candidate safely.
"""

from __future__ import annotations

import abc
import base64
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import httpx

from qwen_ocr.config import InferenceParams
from qwen_ocr.utils import smart_resize

logger = logging.getLogger(__name__)

# One retry on a 5xx, after this pause. Ollama 500s are often transient (model load, GPU hiccup).
SERVER_ERROR_RETRY_DELAY_S = 2.0
# Characters of the 5xx response body kept in the error message.
ERROR_BODY_CHARS = 300

_USERINFO_RE = re.compile(r"(?P<scheme>[A-Za-z][A-Za-z0-9+.-]*://)[^/\s@]+@")
# Whole Authorization header lines: consume the scheme and credential to end of line.
_AUTH_LINE_RE = re.compile(r"(?i)(?P<k>\b(?:proxy-)?authorization\b[\"']?\s*[=:]\s*)[^\r\n]*")
_BEARER_RE = re.compile(r"(?i)\b(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]+")
# key=value / "key": "value" for credential-looking keys; a quoted value is matched whole,
# so spaces inside it do not leave a suffix behind.
_KV_RE = re.compile(
    r"(?i)(?P<k>\b(?:password|passwd|pwd|secret|token|api[_-]?key|apikey)\b"
    r"[\"']?\s*[=:]\s*)(?:\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|[^\s\"',&;}]+)"
)
REDACTED = "***"

# finish_reason / done_reason values that mean the model hit an output limit.
# Truncated OCR is treated as a per-page FAILURE (never silent partial text).
TRUNCATION_REASONS = frozenset({"length", "max_tokens"})


@dataclass(frozen=True)
class ImageSendMetrics:
    """Original vs resized dimensions for one page image."""

    original_width: int
    original_height: int
    sent_width: int
    sent_height: int
    min_pixels: int
    max_pixels: int

    @property
    def original_megapixels(self) -> float:
        return self.original_width * self.original_height / 1e6

    @property
    def sent_megapixels(self) -> float:
        return self.sent_width * self.sent_height / 1e6


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass(frozen=True)
class OcrImageResult:
    """One page OCR call: raw assistant text plus provenance for sidecars."""

    text: str
    finish_reason: str | None
    usage: TokenUsage
    image: ImageSendMetrics
    model: str

    @property
    def truncated(self) -> bool:
        return (self.finish_reason or "") in TRUNCATION_REASONS


def redact_credentials(text: str, secrets: tuple[str, ...] = ()) -> str:
    """Strip credentials from text bound for an error message or log.

    Pattern passes run first, then ``secrets`` (exact values the caller knows: API key,
    URL password) are replaced wherever they still appear, catching bare echoed keys no
    pattern could. Exact replacement goes last so ``***`` never splits a token mid-match.
    """
    text = _USERINFO_RE.sub(r"\g<scheme>" + REDACTED + "@", text)
    text = _AUTH_LINE_RE.sub(lambda m: m.group("k") + REDACTED, text)
    text = _BEARER_RE.sub(lambda m: m.group(0).split()[0] + " " + REDACTED, text)
    text = _KV_RE.sub(lambda m: m.group("k") + REDACTED, text)
    for secret in sorted((x for x in secrets if x), key=len, reverse=True):
        text = text.replace(secret, REDACTED)
    return text


class BackendServerError(httpx.HTTPStatusError):
    """HTTP error from the backend; the message carries the redacted response body."""


@dataclass(frozen=True)
class Availability:
    ok: bool
    reason: str = ""


class Backend(abc.ABC):
    """One OCR backend. Subclasses set name/base_url/model/auth + a probe."""

    name: str
    # Retry policy, explicit per backend: only free local backends may re-send a request after
    # a 5xx. A billed cloud backend may have charged for the failed call, so it never retries.
    retry_on_5xx: bool = False

    @abc.abstractmethod
    def availability(self) -> Availability:
        """Cheap liveness probe. Must not raise."""

    def is_available(self) -> bool:
        return self.availability().ok

    @abc.abstractmethod
    def _endpoint(self) -> tuple[str, str | None, str]:
        """Return (base_url, api_key_or_None, model)."""

    def _thinking_payload(self, enable_thinking: bool) -> dict:
        """Extra request keys that turn a hybrid-thinking model's reasoning off.

        OpenAI-compatible servers spell this differently. vLLM (and OpenRouter's
        Qwen routes) forward ``chat_template_kwargs`` into the chat template, where
        ``enable_thinking=False`` selects the non-thinking branch — verified live on
        Qwen/Qwen3.5-35B-A3B-FP8 under vLLM 0.17 (issue #4). Ollama overrides this
        with its own ``think`` key. Returning ``{}`` when thinking is *enabled*
        leaves the server default untouched.
        """
        if enable_thinking:
            return {}
        return {"chat_template_kwargs": {"enable_thinking": False}}

    def ocr_image(self, image_path: Path, params: InferenceParams) -> OcrImageResult:
        """OCR a single page image to Markdown via OpenAI-compatible chat."""
        base_url, api_key, model = self._endpoint()
        data_url, image_metrics = _encode_image(image_path, params.min_pixels, params.max_pixels)

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        text_part = {"type": "text", "text": params.prompt}
        image_part = {"type": "image_url", "image_url": {"url": data_url}}
        if params.message_order == "image-first":
            content = [image_part, text_part]
        else:
            content = [text_part, image_part]

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": params.max_output_tokens,
            "temperature": params.temperature,
            # OpenAI ignores unknown keys; vLLM/Ollama/DashScope honour these.
            "repetition_penalty": params.repetition_penalty,
        }
        payload.update(self._thinking_payload(params.enable_thinking))

        # httpx logs the request URL; keep userinfo out of it and send it as Basic auth.
        parsed = httpx.URL(base_url.rstrip("/") + "/chat/completions")
        extra: dict = {}
        secrets = [api_key or ""]
        if parsed.userinfo:
            user, _, password = parsed.userinfo.decode().partition(":")
            extra["auth"] = httpx.BasicAuth(unquote(user), unquote(password))
            secrets += [parsed.userinfo.decode(), password, unquote(password)]
            parsed = parsed.copy_with(userinfo=None)
        resp = _post_with_5xx_retry(
            str(parsed),
            payload,
            headers,
            params.timeout,
            retry=self.retry_on_5xx,
            secrets=tuple(secrets),
            **extra,
        )
        data = resp.json()
        text, finish_reason = _extract_text_and_finish_reason(data)
        return OcrImageResult(
            text=text,
            finish_reason=finish_reason,
            usage=_extract_usage(data),
            image=image_metrics,
            model=model,
        )


def _post_with_5xx_retry(
    url: str,
    payload: dict,
    headers: dict,
    timeout: float,
    *,
    retry: bool,
    secrets: tuple[str, ...] = (),
    **extra,
) -> httpx.Response:
    """POST; if ``retry``, retry a 5xx once (same request) after a short delay.

    One monotonic deadline spans both attempts: each attempt gets the budget left at that
    moment, and the retry is skipped when the delay plus a nonzero budget no longer fits.
    (httpx applies a timeout per operation, so a single attempt is bounded by the remaining
    budget per read/connect, not as a hard wall-clock cap.) Timeouts, 3xx and 4xx never
    retry. Any non-2xx raises with the redacted response body.
    """
    deadline = time.monotonic() + timeout
    resp = httpx.post(url, json=payload, headers=headers, timeout=timeout, **extra)
    if retry and resp.status_code >= 500:
        if deadline - time.monotonic() > SERVER_ERROR_RETRY_DELAY_S:
            logger.warning("HTTP %d from backend; retrying once", resp.status_code)
            time.sleep(SERVER_ERROR_RETRY_DELAY_S)
            remaining = deadline - time.monotonic()
            if remaining > 0:
                resp = httpx.post(url, json=payload, headers=headers, timeout=remaining, **extra)
    if not resp.is_success:  # like raise_for_status: 3xx/4xx/5xx all raise
        # Redact the whole body first, then truncate, so a token cut at the boundary can't leak.
        body = redact_credentials(resp.text, secrets)[:ERROR_BODY_CHARS]
        raise BackendServerError(
            f"HTTP {resp.status_code} from {redact_credentials(url, secrets)}: {body}",
            request=resp.request,
            response=resp,
        )
    return resp


def _encode_image(
    image_path: Path, min_pixels: int, max_pixels: int
) -> tuple[str, ImageSendMetrics]:
    """Base64 data URL, resized to the patch-aligned pixel budget (issue #5).

    The resolution the model actually sees is decided here, not by the server's
    default processor config: :func:`smart_resize` maps the rendered page into
    ``[min_pixels, max_pixels]`` on multiples of Qwen3-VL's patch factor. LANCZOS
    keeps small glyphs legible through the downsample. The resolved budget is logged
    per page so a quality regression is traceable to resolution.
    """
    from io import BytesIO

    from PIL import Image

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        target = smart_resize(w, h, min_pixels, max_pixels)
        if target != (w, h):
            img = img.resize(target, Image.LANCZOS)
        logger.info(
            "%s: %dx%d (%.2f MP) -> %dx%d (%.2f MP); budget %.2f-%.2f MP",
            image_path.name,
            w,
            h,
            w * h / 1e6,
            target[0],
            target[1],
            target[0] * target[1] / 1e6,
            min_pixels / 1e6,
            max_pixels / 1e6,
        )
        metrics = ImageSendMetrics(
            original_width=w,
            original_height=h,
            sent_width=target[0],
            sent_height=target[1],
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        buf = BytesIO()
        img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}", metrics


def _extract_finish_reason(data: dict[str, Any]) -> str | None:
    """OpenAI ``choices[0].finish_reason`` or Ollama native ``done_reason``."""
    try:
        choice = data["choices"][0]
        reason = choice.get("finish_reason")
        if reason is not None:
            return str(reason)
    except (KeyError, IndexError, TypeError):
        pass
    done = data.get("done_reason")
    return str(done) if done is not None else None


def _extract_usage(data: dict[str, Any]) -> TokenUsage:
    usage = data.get("usage") or {}
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if prompt is None and completion is None:
        prompt = data.get("prompt_eval_count")
        completion = data.get("eval_count")
    return TokenUsage(
        prompt_tokens=int(prompt) if prompt is not None else None,
        completion_tokens=int(completion) if completion is not None else None,
    )


def _extract_text_and_finish_reason(data: dict[str, Any]) -> tuple[str, str | None]:
    """Pull assistant text and finish/done reason from an OpenAI-compatible response."""
    try:
        message = data["choices"][0]["message"]
        text = (message.get("content") or "").strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"unexpected OCR response shape: {data!r}") from exc
    return text, _extract_finish_reason(data)


def _extract_text(data: dict) -> str:
    """Pull assistant text only (tests / legacy callers)."""
    text, _ = _extract_text_and_finish_reason(data)
    return text
