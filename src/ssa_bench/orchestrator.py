"""End-to-end orchestrator: generate → run → score → save."""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from .runners.base import Runner, RunnerRequest, RunnerResponse
from .scoring.exact_match import (
    Verdict,
    score_all,
    summary,
    write_verdicts_jsonl,
)
from .tasks.corpus import ensure_corpus
from .tasks.niah_single import (
    NIAHSample,
    generate_samples,
    write_samples_jsonl,
)

log = logging.getLogger(__name__)


@dataclass
class RunConfig:
    task: str
    runner_name: str
    model_alias: str
    n_samples: int
    target_input_tokens: int
    seed: int
    mode: str  # "single" or "batch"
    output_dir: Path
    corpus_url: str
    max_output_tokens: int = 64
    temperature: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["output_dir"] = str(self.output_dir)
        return d


def _git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
            )
            .strip()
            or None
        )
    except Exception:
        return None


def _write_responses_jsonl(responses: list[RunnerResponse], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in responses:
            fh.write(json.dumps(asdict(r)) + "\n")


def _write_results_md(
    *,
    cfg: RunConfig,
    samples: list[NIAHSample],
    verdicts: list[Verdict],
    summ: dict,
    model_versions: set[str],
    path: Path,
) -> None:
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    commit = _git_commit()
    lines: list[str] = []
    lines.append(f"# {cfg.task} — {cfg.model_alias} — {started.split('T')[0]}\n")
    lines.append("## Headline\n")
    lines.append(
        f"Accuracy: **{summ['accuracy']:.1%}** "
        f"(95% CI {summ['ci_95_low']:.1%}–{summ['ci_95_high']:.1%}) "
        f"on {summ['n']} samples (n_correct={summ['n_correct']}, "
        f"n_errored={summ['n_errored']}).\n"
    )
    lines.append("## Run configuration\n")
    lines.append("| Setting | Value |")
    lines.append("|---|---|")
    lines.append(f"| Task | `{cfg.task}` |")
    lines.append(f"| Runner | `{cfg.runner_name}` |")
    lines.append(f"| Model alias | `{cfg.model_alias}` |")
    pinned = ", ".join(sorted(model_versions)) if model_versions else "(none recorded)"
    lines.append(f"| Pinned model version | `{pinned}` |")
    lines.append(f"| n_samples | {cfg.n_samples} |")
    lines.append(f"| target_input_tokens | {cfg.target_input_tokens:,} (cl100k approx) |")
    if samples:
        actual_min = min(s.actual_input_tokens for s in samples)
        actual_max = max(s.actual_input_tokens for s in samples)
        actual_avg = sum(s.actual_input_tokens for s in samples) // len(samples)
        lines.append(
            f"| actual_input_tokens (cl100k) | min={actual_min:,}, "
            f"avg={actual_avg:,}, max={actual_max:,} |"
        )
    lines.append(f"| seed | {cfg.seed} |")
    lines.append(f"| mode | `{cfg.mode}` |")
    lines.append(f"| temperature | {cfg.temperature} |")
    lines.append(f"| max_output_tokens | {cfg.max_output_tokens} |")
    lines.append(f"| corpus URL | {cfg.corpus_url} |")
    if samples:
        lines.append(f"| corpus sha256 | `{samples[0].haystack_sha256}` |")
    lines.append(f"| git commit | `{commit or 'unknown'}` |")
    lines.append(f"| python | {sys.version.split()[0]} |")
    lines.append(f"| platform | {platform.platform()} |")
    lines.append(f"| run started (UTC) | {started} |")
    lines.append("")
    lines.append("## Artifacts in this directory\n")
    lines.append("- `config.json` — exact runtime configuration.")
    lines.append("- `samples.jsonl` — generated samples (deterministic given seed).")
    lines.append("- `responses.jsonl` — raw model responses including token usage.")
    lines.append("- `verdicts.jsonl` — per-sample correctness verdicts.")
    lines.append("- `summary.json` — aggregate accuracy and CI.")
    lines.append("")
    lines.append("## Methodology\n")
    lines.append(
        "See [METHODOLOGY.md](../../METHODOLOGY.md) for the full pre-registered "
        "methodology. The task spec follows RULER (Hsieh et al., NVIDIA, 2024) "
        "NIAH-single. Scoring is exact substring match of the 7-digit value "
        "against the model response. Wilson 95% CI is reported alongside the "
        "point estimate."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_niah_single(
    *,
    runner: Runner,
    n_samples: int,
    target_input_tokens: int,
    seed: int,
    mode: str,
    output_dir: Path,
    corpus_url: str,
    cache_dir: Path | None = None,
    max_output_tokens: int = 64,
    temperature: float = 0.0,
) -> dict:
    """Run NIAH-single end to end and write all artifacts.

    Returns the summary dict (also written to ``summary.json``).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = RunConfig(
        task="ruler.niah_single",
        runner_name=runner.name,
        model_alias=getattr(runner, "model", "(unknown)"),
        n_samples=n_samples,
        target_input_tokens=target_input_tokens,
        seed=seed,
        mode=mode,
        output_dir=output_dir,
        corpus_url=corpus_url,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )
    (output_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2) + "\n")

    log.info("ensuring corpus is cached…")
    corpus_path = ensure_corpus(url=corpus_url, cache_dir=cache_dir)
    log.info("corpus at %s", corpus_path)

    log.info("generating %d samples at target=%d tokens, seed=%d", n_samples, target_input_tokens, seed)
    samples = generate_samples(
        n_samples=n_samples,
        target_input_tokens=target_input_tokens,
        seed=seed,
        corpus_path=corpus_path,
    )
    write_samples_jsonl(samples, output_dir / "samples.jsonl")
    log.info("samples written")

    requests = [
        RunnerRequest(
            sample_id=s.sample_id,
            prompt=s.prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
        for s in samples
    ]

    log.info("dispatching %d requests in mode=%s", len(requests), mode)
    if mode == "single":
        responses: list[RunnerResponse] = []
        for req in requests:
            resp = runner.run_one(req)
            responses.append(resp)
            log.info(
                "sample %d done (%d input, %d output, err=%s)",
                resp.sample_id,
                resp.input_tokens,
                resp.output_tokens,
                resp.error,
            )
    elif mode == "batch":
        responses = runner.run_batch(requests)
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    _write_responses_jsonl(responses, output_dir / "responses.jsonl")
    log.info("responses written")

    expected_by_sample = {s.sample_id: s.expected_answer for s in samples}
    response_by_sample = {r.sample_id: r.text for r in responses}
    error_by_sample = {r.sample_id: r.error for r in responses}
    verdicts = score_all(
        expected_by_sample=expected_by_sample,
        response_by_sample=response_by_sample,
        error_by_sample=error_by_sample,
    )
    write_verdicts_jsonl(verdicts, output_dir / "verdicts.jsonl")

    summ = summary(verdicts)
    (output_dir / "summary.json").write_text(json.dumps(summ, indent=2) + "\n")

    model_versions = {r.model_version for r in responses if r.model_version}
    _write_results_md(
        cfg=cfg,
        samples=samples,
        verdicts=verdicts,
        summ=summ,
        model_versions=model_versions,
        path=output_dir / "results.md",
    )

    log.info(
        "DONE: accuracy=%.1f%% (n=%d, errored=%d) → %s",
        100 * summ["accuracy"],
        summ["n"],
        summ["n_errored"],
        output_dir,
    )
    return summ


def _bool_check_environment(env_var: str) -> bool:
    return bool(os.environ.get(env_var, "").strip())
