# CLAUDE.md — qwen-ocr-cli

OCR CLI wrapping the Qwen-VL vision models. A standalone sibling of the other
`../ocr/{name}-ocr-cli` engines, consumed by [`socr`](../../socr) via a thin adapter.

## The canonical output contract (DO NOT break)

Output writing is **owned by the shared `ocr-output-contract` package** (pinned via a
`git+https` dependency in `pyproject.toml`). qwen keeps its bytes->page-text + backend logic;
the contract owns *where bytes go and what metadata looks like*, so qwen's output is
byte-structure-identical to every sibling engine. Import the same names the gemini reference
implementation imports (see `../gemini-ocr-cli/gemini_ocr/processor.py`).

```
qwen-ocr process <path-or-dir> [-o <out>] --backend <b> --dpi <N> [-w N] [-q] [--verbose] [--reprocess]
qwen-ocr --version          # exit 0 when installed  → socr availability probe
qwen-ocr backends           # human-readable: which backends are live
```

- `process` is a **required subcommand** (Click group), matching `glm-ocr` / `deepseek-ocr`.
- Input is a **PDF**, a **directory of page images** (treated as ONE document), or a **tree of
  PDFs** (batch). `socr` renders pages to PNGs and passes the dir — both must work.
- Output root defaults to `<input-parent>/ocr/`; `-o` overrides verbatim but is **never
  required** (this used to diverge — qwen forced `-o`).
- Per-document layout: ONE `<root>/<rel/dir>/<stem>/<stem>.md` aggregating all pages under
  `## Page N` headers (this fixed the CRITICAL QWEN-01 multi-page data loss — qwen used to
  write one folder per page and socr's read-back kept only the first). Use the contract's
  `resolve_output_root` / `relative_key` / `doc_dir_for` / `markdown_path_for` /
  `assemble_pages`; never reconstruct paths by hand.
- Metadata at both levels via `DocMetadata` + `write_doc_metadata` + `RootIndex` (per-doc
  sidecar AND a root index keyed by input-relative path). Failures recorded `status="failed"`.
- An **empty/whitespace model response is a per-page FAILURE** (QWEN-02), never a 0-byte "ok".
- Exit code via `RunOutcome.exit_code`: nonzero if any document/page failed.
- `-q` emits output `.md` paths only; non-zero exit ⇒ `socr` marks the page CLI_ERROR.

`tests/test_output_contract.py` proves conformance via the package's
`ocr_output_contract.conformance.assert_conforms` harness (incl. the QWEN-01 multi-page and
QWEN-02 empty-response cases). Run it on any change to output writing.

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
