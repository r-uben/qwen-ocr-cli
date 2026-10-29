"""Shared fakes for OCR result shapes in unit tests."""

from __future__ import annotations

from qwen_ocr import config
from qwen_ocr.backends.base import ImageSendMetrics, OcrImageResult, TokenUsage


def fake_ocr_result(
    text: str,
    *,
    finish_reason: str | None = "stop",
    model: str = "fake-model",
    min_pixels: int = config.MIN_PIXELS,
    max_pixels: int = config.MAX_PIXELS,
) -> OcrImageResult:
    return OcrImageResult(
        text=text,
        finish_reason=finish_reason,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=20),
        image=ImageSendMetrics(
            original_width=100,
            original_height=100,
            sent_width=100,
            sent_height=100,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        ),
        model=model,
    )
