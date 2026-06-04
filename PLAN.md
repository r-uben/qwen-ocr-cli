# PLAN.md — qwen-ocr-cli

Goal: a drop-in `qwen-ocr` engine for `socr` that runs Qwen-VL across three backends
(Ollama / vLLM / cloud API) with a smart auto-selector, giving the pipeline a local/cheap
OCR tier that is *actually competitive* (Qwen3.5-VL ≈ 0.54–0.58 on socOCRbench) instead of
the current near-garbage GLM (0.37) / DeepSeek (0.09) tier.

## Why (one paragraph)

socOCRbench grades socr's engines: Gemini wins (0.60–0.64, so cloud-first is correct on hard
pages), Mistral is mid-pack (0.45) yet sits *above* Gemini on socr's cost ladder — a bug — and
the free local tier is built on the two worst models tested. Qwen3.5-VL is the missing rung:
open, cheap, near-Gemini. We add it as an external engine, then fix socr's routing around it.

## Milestones

### M1 — CLI skeleton + Ollama backend (MVP, ship first)
- [ ] `pyproject.toml`: `[project.scripts] qwen-ocr = "qwen_ocr.cli:main"`, deps (click, httpx,
      pymupdf, pillow), ruff config. `.env.example`, `.gitignore` (`.venv*`, `.git` artifacts).
- [ ] `cli.py`: Click group with `process`, `backends`, `--version`. Flags exactly per CLAUDE.md.
- [ ] `processor.py`: PDF→PNG render (pymupdf at `--dpi`) OR read image dir → per-page loop →
      write `{out}/{stem}/{stem}.md`. Mirrors glm-ocr's IO so socr reads it unchanged.
- [ ] `backends/ollama.py`: `qwen3-vl` via Ollama HTTP. `ocr_image(png) -> markdown`.
- [ ] `backends/base.py`: `Backend` protocol + `is_available()`.
- [ ] Tests: CLI parsing, IO layout, render, ollama backend (HTTP mocked). Pass with no models.
- [ ] iCloud git/venv setup per rules, `git init`, first commit on `main`.

### M2 — vLLM + API backends + smart selector
- [ ] `backends/vllm.py` (OpenAI-compatible; `VLLM_BASE_URL`) and `backends/api.py`
      (DashScope + OpenRouter; key from env).
- [ ] `backends/select.py`: `resolve_backend(prefer)` —
      `prefer=cost` (default): try `vllm → ollama → api`;
      `prefer=quality`: try `api → vllm → ollama`;
      `prefer=speed`: `ollama → vllm → api`.
      Each candidate must `is_available()`; first hit wins. None live ⇒ raise with a precise
      per-backend reason (model not pulled / server unreachable / key unset). Probing never crashes.
- [ ] `--backend auto` (default) routes through the selector; explicit backend skips it.
- [ ] `config.py`: default model IDs per backend + tunable inference params (max tokens,
      repetition penalty, image max-side). No inline magic numbers.

### M3 — Quality knobs (informed by Dasanaike's 20M-doc lessons)
- [ ] `--dpi` already plumbed; default higher than socr's 200 (test 300/400 — "resolution
      matters more than people think"). Document the validated default.
- [ ] Light post-processing: detect + trim trailing repetition / degenerate loops before write
      (truncate+dedup). Keep it cheap; socr's scorer still audits downstream.
- [ ] `qwen-ocr backends` shows live backends + chosen model + probe detail.

## socr-side integration (separate `feat/qwen-engine` branch in ../../socr)

> Not in this repo — tracked here so the contract is one document.
- [ ] `EngineType.QWEN = "qwen"` in `core/config.py`; add to `ENGINE_PRIORITY` /
      `_LOCAL_ENGINE_ORDER` (above GLM/DeepSeek — Qwen is the better local model).
- [ ] `engines/qwen.py`: ~40-line `BaseEngine` adapter (mirror `glm.py`); register in
      `engines/registry.py`. Backend chosen via a new `config.qwen_backend` (default `auto`).
- [ ] `core/providers.py`: add Qwen profile; **fix the ladder** — Mistral must sit *below*
      Gemini (it's worse and pricier), Qwen local = free/0.0, Qwen API ≈ 0.0005.
- [ ] Parametrize/raise `render_dpi`; benchmark the effect.
- [ ] Run `socr`'s `benchmark/` on real papers: Qwen(local) vs GLM vs Gemini — measure before
      trusting. "Test on your own images; leaderboards don't discriminate top models."

## Non-goals (resist scope creep)

- No fine-tuning. No new model architectures. Zero-shot inference only.
- Don't chase new OCR releases; pick a backend default that works and stick with it.
- Don't reimplement socr's audit/consensus here — this CLI just produces good per-page markdown.

## Status

- [x] Spec docs (this file, README, CLAUDE.md)
- [ ] M1 — next
