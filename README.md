# qwen-ocr

OCR via the Qwen-VL vision-language models, with a smart backend that runs **locally
(Ollama), on a self-hosted GPU (vLLM), or in the cloud (DashScope / OpenRouter)** —
auto-selecting whichever is available.

Qwen-VL is the strongest *open* OCR family on social-science / historical documents:
on [socOCRbench](https://noahdasanaike.github.io/posts/sococrbench.html), Qwen3.5-VL
(27B–397B) scores ~0.54–0.58, rivaling Gemini Flash and far ahead of GLM-OCR (0.37)
and DeepSeek-OCR (0.09). This CLI exists to give the [`socr`](../../socr) pipeline a
local/cheap tier that is actually competitive — not a near-garbage fallback.

> One caveat worth stating up front: those scores are for *messy* documents (handwriting,
> degraded scans, dense tables). On clean born-digital PDFs every model does well and the
> gaps shrink. Use this where OCR actually matters: scans, figures, tables, equations.

## Install

```bash
uv tool install --editable .       # exposes the `qwen-ocr` command
```

## Usage

```bash
# Auto backend: probe what's live, pick the best available
qwen-ocr process paper.pdf -o ./out

# Force a backend
qwen-ocr process paper.pdf -o ./out --backend ollama   # local, free, private
qwen-ocr process scans/      -o ./out --backend api     # cloud, best quality
qwen-ocr process scans/      -o ./out --backend vllm    # self-hosted GPU

# Steer auto-selection by what you care about
qwen-ocr process paper.pdf -o ./out --prefer cost       # free backends first (default)
qwen-ocr process paper.pdf -o ./out --prefer quality    # cloud 397B first

qwen-ocr backends                                        # show which backends are live
qwen-ocr --version
```

Input may be a **PDF** or a **directory of page images** (`socr` renders pages to PNGs and
passes the directory). Output is written as `{out}/{stem}/{stem}.md`, one markdown file per
input — the layout `socr`'s adapter reads back.

## Backends

| `--backend` | Runs on | Cost | Model (default) | When `auto` picks it |
|---|---|---|---|---|
| `ollama` | your machine | free | `qwen3-vl` (pull first) | Ollama up + model pulled |
| `vllm` | self-hosted GPU (HPC) | free | served `Qwen3-VL` | `VLLM_BASE_URL` reachable |
| `api` | DashScope / OpenRouter | ~$0.40–0.60 / M tok | `qwen3.5-vl-plus` | API key set |

`auto` with `--prefer cost` (default) tries `vllm → ollama → api`; `--prefer quality` tries
`api → vllm → ollama`. It never silently produces nothing: if no backend is live it errors
with exactly what's missing (model not pulled, server unreachable, key unset).

## Environment

```bash
# Cloud API (one of)
export DASHSCOPE_API_KEY=...        # Alibaba DashScope
export OPENROUTER_API_KEY=...       # OpenRouter (qwen/qwen3.5-vl-*)
# Self-hosted vLLM
export VLLM_BASE_URL=http://localhost:8000/v1
```

See `.env.example`. No keys are required for the local Ollama backend.

## Why this is a separate repo

`socr` orchestrates engines but does not embed them: every engine is a standalone sibling
CLI at `../ocr/{name}-ocr-cli` with its own deps, installable and runnable on its own. This
keeps heavy/optional model stacks out of `socr` and lets `qwen-ocr` be used independently.
