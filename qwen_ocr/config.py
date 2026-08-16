"""Defaults and tunables — all overridable by flag or env, none inlined elsewhere.

Model IDs, endpoints, and inference parameters live here so routing/IO code never
hardcodes them. The OCR prompt is deliberately open (no magic thresholds): we ask the
model to transcribe and preserve structure, and let it reason from the page.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# --- Backend endpoints / models (env override → default) ---
OLLAMA_URL = os.environ.get("QWEN_OCR_OLLAMA_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.environ.get("QWEN_OCR_OLLAMA_MODEL", "qwen3-vl")

VLLM_URL = os.environ.get("VLLM_BASE_URL", "")  # empty ⇒ vllm backend unavailable
VLLM_MODEL = os.environ.get("QWEN_OCR_VLLM_MODEL", "Qwen/Qwen3-VL-7B-Instruct")

# Cloud API: DashScope (OpenAI-compatible) or OpenRouter, whichever key is present.
DASHSCOPE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_MODEL = os.environ.get("QWEN_OCR_API_MODEL", "qwen3.5-vl-plus")
OPENROUTER_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = os.environ.get("QWEN_OCR_API_MODEL", "qwen/qwen3.5-vl-plus")

# --- Inference parameters (tunable; resolution matters more than people think) ---
DEFAULT_DPI = 300  # higher than socr's 200 — validated lever for local OCR quality
MAX_IMAGE_SIDE = 4000  # downscale longest side before sending (token/cost guard)
MAX_OUTPUT_TOKENS = 8192
TEMPERATURE = 0.0
# Neutral (1.0) by design: a repetition penalty CORRUPTS OCR of tables/forms, whose
# cells, separators, zeros and aligned spacing are legitimately repetitive. Degenerate
# loops are handled downstream by utils.trim_degenerate_tail instead (Gemini research,
# corroborating Dasanaike's post-processing approach). Raise only with evidence.
REPETITION_PENALTY = 1.0
REQUEST_TIMEOUT = 300.0
# Hybrid-thinking models (Qwen3.5 / Qwen3.6) answer an OCR prompt with their *reasoning*
# ("The user wants me to transcribe... **1. Analyze the Image:**") wrapped around the
# transcription; downstream (socr) ingests that commentary as body text of the document.
# Measured on Bocconi HPC (H100, vLLM 0.17, Qwen/Qwen3.5-35B-A3B-FP8): the same request
# plus thinking-off returns clean output. OCR is transcription, not reasoning — off by
# default, with `--thinking` as the escape hatch. See issue #4.
ENABLE_THINKING = False

OCR_PROMPT = (
    "Transcribe all text in this image into clean Markdown. "
    "Preserve document structure: headings, lists, and paragraphs. "
    "Render tables as Markdown tables and mathematics as LaTeX. "
    "Output only the transcription, with no commentary or code fences."
)


@dataclass(frozen=True)
class InferenceParams:
    """Per-call generation knobs, threaded from CLI flags."""

    max_output_tokens: int = MAX_OUTPUT_TOKENS
    temperature: float = TEMPERATURE
    repetition_penalty: float = REPETITION_PENALTY
    max_image_side: int = MAX_IMAGE_SIDE
    timeout: float = REQUEST_TIMEOUT
    prompt: str = OCR_PROMPT
    enable_thinking: bool = ENABLE_THINKING
