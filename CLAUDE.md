# CLAUDE.md — qwen-ocr-cli

OCR CLI wrapping the Qwen-VL vision models. A standalone sibling of the other
`../ocr/{name}-ocr-cli` engines, consumed by [`socr`](../../socr) via a thin adapter.

## The contract socr depends on (DO NOT break)

`socr`'s `BaseEngine` shells out to this exact interface and reads markdown back. Any change
here must stay compatible with `socr/src/socr/engines/qwen.py`:

```
qwen-ocr process <path-or-image-dir> -o <out> --backend <b> --dpi <N> [-w N] [-q] [--verbose] [--reprocess]
qwen-ocr --version          # exit 0 when installed  → socr availability probe
qwen-ocr backends           # human-readable: which backends are live
```

- `process` is a **required subcommand** (Click group), matching `glm-ocr` / `deepseek-ocr`.
- Input is a **PDF** OR a **directory of page images** (PNG). `socr` renders pages to PNGs and
  passes the dir for per-page routing — both must work.
- Output: `{out}/{stem}/{stem}.md`, one file per input stem. This is the only layout `socr`
  reads (`_read_page_output` / `_read_output`). Do not flatten or rename.
- `-q` silences stdout; non-zero exit ⇒ `socr` marks the page CLI_ERROR and escalates.

When in doubt about a flag, read `../glm-ocr-cli/glm_ocr/cli.py` — mirror it.

## Backends (`qwen_ocr/backends/`)

One module per backend, all behind a common `Backend` protocol (`ocr_image(png) -> str`):
- `ollama.py` — local `qwen3-vl` via Ollama HTTP. Free. Probe: model pulled + daemon up.
- `vllm.py` — OpenAI-compatible vLLM server (HPC). Free. Probe: `VLLM_BASE_URL` reachable.
- `api.py` — DashScope / OpenRouter. Paid. Probe: API key env set.
- `select.py` — `resolve_backend(prefer)` does the smart auto-pick (see PLAN.md). Probing must
  never raise: a dead backend returns `unavailable`, never a crash.

## Build / test

```bash
uv sync                         # install deps into the project venv
uv run qwen-ocr --version       # smoke test
uv run pytest -q                # unit tests (mock the backends; no live models in CI)
uv run ruff check . && uv run ruff format .
```

- **YOU MUST** use `uv run`; never `python file.py`. Entry point is `[project.scripts]` in
  `pyproject.toml` → `qwen-ocr = "qwen_ocr.cli:main"`.
- Tests mock backend HTTP/subprocess calls — they must pass with **no** GPU, no key, no Ollama.
  Live-model behaviour goes behind a `@pytest.mark.integration` opt-in, never in the default run.

## iCloud (this repo lives under iCloud Drive)

Before `git init` or creating a venv, follow `~/.claude/rules/icloud.md`:
- Separate the gitdir: real `.git` under `~/projects/qwen-ocr-cli.git`, a `gitdir:` pointer
  file in the repo. Set `xattr com.apple.fileprovider.ignore#P 1 .git`.
- Use an external venv: `UV_PROJECT_ENVIRONMENT=~/venvs/qwen-ocr-cli`. `.gitignore` `.venv*`.
- Never let `.venv` or `.git` internals become tracked/synced artifacts.

## Conventions

- No hardcoded model magic numbers in prompts; let the model read the page. Resolution and
  inference params (max tokens, repetition penalty) are tunable flags, calibrated from data.
- New backend → `feat/` branch; bug fix → `fix/`. Keep `main` clean. Decision log under
  `docs/log/YYYY-MM-DD_*.md` for non-trivial work.
- Default backend model IDs live in `config.py`, overridable by flag/env — never inline.
