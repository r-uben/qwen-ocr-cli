"""Cloud API backend — DashScope or OpenRouter. Best quality. Probe: an API key is set.

Qwen3.5-VL via API is the socOCRbench winner among open models (~0.56), rivaling Gemini
Flash at a fraction of the cost. We prefer DashScope (first-party) when both keys exist.
"""

from __future__ import annotations

import os

from qwen_ocr import config
from qwen_ocr.backends.base import Availability, Backend


class ApiBackend(Backend):
    name = "api"

    def __init__(self, model: str | None = None) -> None:
        self._dashscope_key = os.environ.get("DASHSCOPE_API_KEY", "")
        self._openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
        self._model_override = model

    @property
    def provider(self) -> str:
        if self._dashscope_key:
            return "dashscope"
        if self._openrouter_key:
            return "openrouter"
        return ""

    def _endpoint(self) -> tuple[str, str | None, str]:
        if self.provider == "dashscope":
            return (
                config.DASHSCOPE_URL,
                self._dashscope_key,
                self._model_override or config.DASHSCOPE_MODEL,
            )
        if self.provider == "openrouter":
            return (
                config.OPENROUTER_URL,
                self._openrouter_key,
                self._model_override or config.OPENROUTER_MODEL,
            )
        raise RuntimeError("no cloud API key set")  # guarded by availability()

    def availability(self) -> Availability:
        if self.provider:
            return Availability(True)
        return Availability(
            False, "no API key (set DASHSCOPE_API_KEY or OPENROUTER_API_KEY)"
        )
