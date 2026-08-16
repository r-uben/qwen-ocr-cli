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
from dataclasses import dataclass
from pathlib import Path

import httpx

from qwen_ocr.config import InferenceParams


@dataclass(frozen=True)
class Availability:
    ok: bool
    reason: str = ""


class Backend(abc.ABC):
    """One OCR backend. Subclasses set name/base_url/model/auth + a probe."""

    name: str

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

    def ocr_image(self, image_path: Path, params: InferenceParams) -> str:
        """OCR a single page image to Markdown via OpenAI-compatible chat."""
        base_url, api_key, model = self._endpoint()
        data_url = _encode_image(image_path, params.max_image_side)

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": params.prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "max_tokens": params.max_output_tokens,
            "temperature": params.temperature,
            # OpenAI ignores unknown keys; vLLM/Ollama/DashScope honour these.
            "repetition_penalty": params.repetition_penalty,
        }
        payload.update(self._thinking_payload(params.enable_thinking))

        url = base_url.rstrip("/") + "/chat/completions"
        resp = httpx.post(url, json=payload, headers=headers, timeout=params.timeout)
        resp.raise_for_status()
        data = resp.json()
        return _extract_text(data)


def _encode_image(image_path: Path, max_side: int) -> str:
    """Base64 data URL, downscaling the longest side to max_side to bound tokens."""
    from io import BytesIO

    from PIL import Image

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        longest = max(w, h)
        if max_side and longest > max_side:
            scale = max_side / longest
            img = img.resize((round(w * scale), round(h * scale)))
        buf = BytesIO()
        img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _extract_text(data: dict) -> str:
    """Pull assistant text from an OpenAI-compatible response."""
    try:
        return (data["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError) as exc:  # malformed/empty response
        raise ValueError(f"unexpected OCR response shape: {data!r}") from exc
