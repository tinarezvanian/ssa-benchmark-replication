"""Exact-match scoring for NIAH-single.

A sample is correct iff the expected answer (the 7-digit value) appears
as a contiguous substring of the model's response. Whitespace and
punctuation are not stripped from the response before matching, but the
expected answer itself is digit-only and unique within the haystack, so
substring match is sufficient.

Stricter alternatives we deliberately do NOT use:
- "Response starts with expected answer": punishes verbose models that
  preface answers with text. RULER's published methodology does not
  require this.
- "Response equals expected answer (after strip)": same concern.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Verdict:
    sample_id: int
    expected: str
    response: str
    correct: bool
    error: str | None = None


def score_one(*, expected: str, response: str) -> bool:
    """Return True iff the expected answer is a substring of the response."""
    return expected in response


def score_all(
    *,
    expected_by_sample: dict[int, str],
    response_by_sample: dict[int, str],
    error_by_sample: dict[int, str | None] | None = None,
) -> list[Verdict]:
    """Score a batch of responses.

    Missing samples (in expected but not in response) are recorded as
    errored verdicts with correct=False.
    """
    error_by_sample = error_by_sample or {}
    verdicts: list[Verdict] = []
    for sample_id, expected in sorted(expected_by_sample.items()):
        response = response_by_sample.get(sample_id, "")
        err = error_by_sample.get(sample_id)
        correct = False if err else score_one(expected=expected, response=response)
        verdicts.append(
            Verdict(
                sample_id=sample_id,
                expected=expected,
                response=response,
                correct=correct,
                error=err,
            )
        )
    return verdicts


def write_verdicts_jsonl(verdicts: list[Verdict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for v in verdicts:
            fh.write(json.dumps(asdict(v)) + "\n")


def summary(verdicts: list[Verdict]) -> dict:
    """Compute an aggregate summary including a 95% Wilson CI for accuracy."""
    n = len(verdicts)
    n_correct = sum(1 for v in verdicts if v.correct)
    n_errored = sum(1 for v in verdicts if v.error)
    if n == 0:
        accuracy = 0.0
        ci_low = ci_high = 0.0
    else:
        accuracy = n_correct / n
        # Wilson 95% CI
        from math import sqrt

        z = 1.96
        p = accuracy
        denom = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denom
        half = (z * sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
        ci_low = max(0.0, center - half)
        ci_high = min(1.0, center + half)
    return {
        "n": n,
        "n_correct": n_correct,
        "n_errored": n_errored,
        "accuracy": accuracy,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
    }
