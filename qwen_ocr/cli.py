"""qwen-ocr CLI — Click group matching the socr engine contract.

    qwen-ocr process <path-or-dir> -o <out> [--backend auto|ollama|vllm|api]
                     [--prefer cost|quality|speed] [--dpi N] [-w N] [-q] [--verbose] [--reprocess]
    qwen-ocr backends
    qwen-ocr --version
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
@click.option("-o", "--output", "output", type=click.Path(path_type=Path), required=True,
              help="Output directory; writes {out}/{stem}/{stem}.md per page.")
@click.option("--backend", type=click.Choice(["auto", "ollama", "vllm", "api"]),
              default="auto", show_default=True, help="OCR backend, or auto-select.")
@click.option("--prefer", type=click.Choice(["cost", "quality", "speed"]),
              default="cost", show_default=True, help="Auto-select bias (backend=auto only).")
@click.option("--dpi", type=int, default=config.DEFAULT_DPI, show_default=True,
              help="Render DPI for PDF pages.")
@click.option("-w", "--workers", type=int, default=1, help="Reserved for parallel pages.")
@click.option("--reprocess", is_flag=True, help="Re-OCR pages whose .md already exists.")
@click.option("-q", "--quiet", is_flag=True, help="Suppress progress output.")
@click.option("--verbose", is_flag=True, help="Verbose logging.")
def process_cmd(source, output, backend, prefer, dpi, workers, reprocess, quiet, verbose):
    """Process a PDF or image directory."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        engine = resolve_backend(prefer) if backend == "auto" else make_backend(backend)
    except (RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    if backend != "auto":
        avail = engine.availability()
        if not avail.ok:
            raise click.ClickException(f"backend '{backend}' unavailable: {avail.reason}")

    if not quiet:
        click.echo(f"qwen-ocr: backend={engine.name} dpi={dpi} → {output}")

    params = InferenceParams()
    results = process(Path(source), Path(output), engine, params, dpi, reprocess)

    ok = sum(1 for r in results if r.ok)
    if not quiet:
        click.echo(f"  {ok}/{len(results)} pages succeeded")
    for r in results:
        if not r.ok:
            click.echo(f"  [fail] {r.stem}: {r.error}", err=True)

    if ok == 0:
        sys.exit(1)


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
