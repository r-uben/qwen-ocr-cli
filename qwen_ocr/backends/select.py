"""Smart backend selection — pick the best *available* backend for a preference.

`auto` probes candidates in a preference-ordered list and returns the first live one.
If none is live, raise with a precise per-backend reason so the user knows exactly what
to fix (pull a model / start a server / set a key) rather than a vague failure.
"""

from __future__ import annotations

from qwen_ocr.backends.api import ApiBackend
from qwen_ocr.backends.base import Availability, Backend
from qwen_ocr.backends.ollama import OllamaBackend
from qwen_ocr.backends.vllm import VLLMBackend

# Preference → probe order. free-first by default (cost); quality flips to cloud-first.
_ORDER: dict[str, list[str]] = {
    "cost": ["vllm", "ollama", "api"],
    "quality": ["api", "vllm", "ollama"],
    "speed": ["ollama", "vllm", "api"],
}

_FACTORIES = {
    "ollama": OllamaBackend,
    "vllm": VLLMBackend,
    "api": ApiBackend,
}


def make_backend(name: str, model: str | None = None) -> Backend:
    """Construct a single named backend, optionally overriding its model."""
    try:
        return _FACTORIES[name](model=model)
    except KeyError:
        raise ValueError(
            f"unknown backend {name!r}; choose from {sorted(_FACTORIES)} or 'auto'"
        ) from None


def probe_all() -> dict[str, Availability]:
    """Availability of every backend, for `qwen-ocr backends`."""
    return {name: factory().availability() for name, factory in _FACTORIES.items()}


def resolve_backend(prefer: str = "cost", model: str | None = None) -> Backend:
    """Return the first available backend in the preference order."""
    order = _ORDER.get(prefer)
    if order is None:
        raise ValueError(f"unknown preference {prefer!r}; choose from {sorted(_ORDER)}")

    reasons: list[str] = []
    for name in order:
        backend = _FACTORIES[name](model=model)
        avail = backend.availability()
        if avail.ok:
            return backend
        reasons.append(f"  {name}: {avail.reason}")

    raise RuntimeError(
        "no Qwen-VL backend available. Tried (prefer=" + prefer + "):\n"
        + "\n".join(reasons)
    )
