"""Local Ollama backend — free, private, offline. Probe: daemon up + model pulled."""

from __future__ import annotations

import httpx

from qwen_ocr import config
from qwen_ocr.backends.base import Availability, Backend


class OllamaBackend(Backend):
    name = "ollama"

    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        self.base_url = base_url or config.OLLAMA_URL
        self.model = model or config.OLLAMA_MODEL
        self._resolved_model: str | None = None  # concrete installed tag, set by probe

    def _endpoint(self) -> tuple[str, str | None, str]:
        # Send the concrete installed tag (e.g. "qwen3-vl:8b"), not the bare name:
        # Ollama 404s on an unresolvable model. Resolve lazily if not probed yet.
        if self._resolved_model is None:
            self.availability()
        model = self._resolved_model or self.model
        # Ollama's OpenAI-compatible endpoint ignores the key but one is required.
        return self.base_url, "ollama", model

    def availability(self) -> Availability:
        # Ollama exposes /api/tags at the host root (not under /v1).
        root = self.base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[: -len("/v1")]
        try:
            resp = httpx.get(f"{root}/api/tags", timeout=3.0)
            resp.raise_for_status()
        except Exception as exc:  # daemon down / unreachable
            return Availability(False, f"Ollama not reachable at {root} ({exc})")

        names = [m.get("name", "") for m in resp.json().get("models", [])]
        # Prefer an exact match; else the first tagged variant (qwen3-vl → qwen3-vl:8b).
        match = next(
            (n for n in names if n == self.model),
            next((n for n in names if n.split(":")[0] == self.model), None),
        )
        if match:
            self._resolved_model = match
            return Availability(True)
        return Availability(
            False, f"model '{self.model}' not pulled (run: ollama pull {self.model})"
        )
