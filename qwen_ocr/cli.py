"""qwen-ocr CLI — Click group matching the socr engine contract.

    qwen-ocr process <path-or-dir> [-o <out>] [--backend auto|ollama|vllm|api]
                     [--prefer cost|quality|speed] [--model M] [--dpi N]
                     [-w N] [-q] [--verbose] [--reprocess]
    qwen-ocr backends
    qwen-ocr --version

Output goes to ``<input-parent>/ocr/`` by default (``-o`` overrides; never
required), and is written through the shared ``ocr-output-contract`` package so
qwen's output is byte-structure-identical to the rest of the engine family.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from qwen_ocr import __version__, config
from qwen_ocr.backends.select import make_backend, probe_all, resolve_backend
from qwen_ocr.config import InferenceParams
from qwen_ocr.processor import process


@click.group()
@click.version_option(__version__, prog_name="qwen-ocr")
def main() -> None:
    """OCR via Qwen-VL with smart backend selection."""


@main.command()
@click.argument("source", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o", "--output", "output", type=click.Path(path_type=Path), default=None,
    help="Output root (default: <input-parent>/ocr/). Writes <stem>/<stem>.md per document.",
)
@click.option("--backend", type=click.Choice(["auto", "ollama", "vllm", "api"]),
              default="auto", show_default=True, help="OCR backend, or auto-select.")
@click.option("--prefer", type=click.Choice(["cost", "quality", "speed"]),
              default="cost", show_default=True, help="Auto-select bias (backend=auto only).")
@click.option("--model", default=None,
              help="Override the backend model (e.g. qwen3.5:27b, qwen3.5:cloud).")
@click.option("--dpi", type=int, default=config.DEFAULT_DPI, show_default=True,
              help="Render DPI for PDF pages.")
@click.option("-w", "--workers", type=int, default=1, help="Reserved for parallel pages.")
@click.option("--reprocess", is_flag=True, help="Re-OCR documents already recorded completed.")
@click.option("-q", "--quiet", is_flag=True, help="Suppress progress; emit output paths only.")
@click.option("--verbose", is_flag=True, help="Verbose logging.")
def process_cmd(source, output, backend, prefer, model, dpi, workers, reprocess, quiet, verbose):
    """Process a PDF (or a directory of page images / a tree of PDFs)."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        engine = (
            resolve_backend(prefer, model=model) if backend == "auto"
            else make_backend(backend, model=model)
        )
    except (RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    if backend != "auto":
        avail = engine.availability()
        if not avail.ok:
            raise click.ClickException(f"backend '{backend}' unavailable: {avail.reason}")

    params = InferenceParams()
    try:
        outcome = process(
            Path(source), engine, params, dpi,
            output_dir=Path(output) if output else None,
            reprocess=reprocess,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    total = outcome.completed + outcome.failed + outcome.partial
    if quiet:
        # Scripting contract: one output .md path per line on stdout.
        for path in outcome.outputs:
            click.echo(path)
    else:
        click.echo(f"qwen-ocr: backend={engine.name} dpi={dpi}")
        click.echo(f"  {outcome.completed}/{total} document(s) completed")
        if outcome.has_failures:
            click.echo(
                f"  {outcome.failed} failed, {outcome.partial} partial: "
                + ", ".join(outcome.failures),
                err=True,
            )

    # Uniform exit policy (canon SYS-02): nonzero if any document/page failed.
    if outcome.exit_code != 0:
        sys.exit(outcome.exit_code)


# socr's BaseEngine calls the subcommand `process`; expose it under that name.
main.add_command(process_cmd, name="process")


@main.command()
def backends() -> None:
    """Show which backends are live."""
    for name, avail in probe_all().items():
        mark = "[+]" if avail.ok else "[x]"
        detail = "" if avail.ok else f"  — {avail.reason}"
        click.echo(f"  {mark} {name}{detail}")


if __name__ == "__main__":
    main()
