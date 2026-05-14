"""CLI entry point: `ssa-bench`."""

from __future__ import annotations

import datetime as dt
import logging
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

from .orchestrator import run_niah_single
from .runners.anthropic import AnthropicRunner
from .tasks.corpus import DEFAULT_CORPUS_URL


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )


def _default_output_dir(task: str, model: str) -> Path:
    today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    safe_model = model.replace("/", "-").replace(":", "-")
    return Path("runs") / f"{today}-{task}-{safe_model}"


@click.group()
def cli() -> None:
    """SSA benchmark replication CLI."""
    load_dotenv()


@cli.command("niah-single")
@click.option(
    "--runner",
    "runner_name",
    type=click.Choice(["anthropic"], case_sensitive=False),
    default="anthropic",
    show_default=True,
    help="Which provider runner to use.",
)
@click.option(
    "--model",
    default="claude-opus-4-6",
    show_default=True,
    help="Model alias (e.g., 'claude-opus-4-6', 'claude-sonnet-4-6', or a full version id).",
)
@click.option("--n", "n_samples", type=int, default=50, show_default=True, help="Sample count.")
@click.option(
    "--context-tokens",
    type=int,
    default=128_000,
    show_default=True,
    help="Target input tokens per sample (cl100k approximation; actual recorded per sample).",
)
@click.option("--seed", type=int, default=42, show_default=True, help="Master seed.")
@click.option(
    "--mode",
    type=click.Choice(["single", "batch"], case_sensitive=False),
    default="batch",
    show_default=True,
    help="'single' for synchronous calls (smoke test); 'batch' for 50%-off batch API.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory for run artifacts (default: runs/YYYY-MM-DD-niah-single-<model>).",
)
@click.option(
    "--corpus-url",
    default=DEFAULT_CORPUS_URL,
    show_default=True,
    help="URL for the haystack corpus (Project Gutenberg Moby Dick by default).",
)
@click.option(
    "--max-output-tokens",
    type=int,
    default=64,
    show_default=True,
    help="Generation cap; NIAH answers are short.",
)
@click.option(
    "--temperature",
    type=float,
    default=0.0,
    show_default=True,
)
@click.option(
    "--log-level",
    default="INFO",
    show_default=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
)
def niah_single(
    runner_name: str,
    model: str,
    n_samples: int,
    context_tokens: int,
    seed: int,
    mode: str,
    output_dir: Path | None,
    corpus_url: str,
    max_output_tokens: int,
    temperature: float,
    log_level: str,
) -> None:
    """Run RULER NIAH-single end to end."""
    _setup_logging(log_level)

    if output_dir is None:
        output_dir = _default_output_dir("niah-single", model)

    if runner_name == "anthropic":
        runner = AnthropicRunner(model=model)
    else:
        raise click.ClickException(f"unsupported runner: {runner_name}")

    summary = run_niah_single(
        runner=runner,
        n_samples=n_samples,
        target_input_tokens=context_tokens,
        seed=seed,
        mode=mode,
        output_dir=output_dir,
        corpus_url=corpus_url,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )

    click.echo("")
    click.echo(f"Accuracy: {summary['accuracy']:.1%}")
    click.echo(
        f"95% CI:   {summary['ci_95_low']:.1%}–{summary['ci_95_high']:.1%}"
    )
    click.echo(f"n:        {summary['n']} (errored: {summary['n_errored']})")
    click.echo(f"Output:   {output_dir}/results.md")


if __name__ == "__main__":
    cli()
