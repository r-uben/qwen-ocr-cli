"""Self-hosted vLLM backend (e.g. Bocconi HPC) — free, powerful. Probe: server reachable."""

from __future__ import annotations

import httpx

from qwen_ocr import config
from qwen_ocr.backends.base import Availability, Backend


class VLLMBackend(Backend):
    name = "vllm"

    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        self.base_url = base_url if base_url is not None else config.VLLM_URL
        self.model = model or config.VLLM_MODEL

    def _endpoint(self) -> tuple[str, str | None, str]:
        return self.base_url, None, self.model

    def availability(self) -> Availability:
        if not self.base_url:
            return Availability(False, "VLLM_BASE_URL not set")
        try:
            resp = httpx.get(self.base_url.rstrip("/") + "/models", timeout=3.0)
            resp.raise_for_status()
        except Exception as exc:
            return Availability(False, f"vLLM not reachable at {self.base_url} ({exc})")
        return Availability(True)
