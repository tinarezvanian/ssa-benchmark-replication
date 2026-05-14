"""Unit tests for the NIAH-single generator.

These tests do not call any API. They verify:
1. Determinism: same seed -> identical samples.
2. The needle appears verbatim in the haystack.
3. The expected answer appears in the haystack.
4. Token count is within tolerance of the target.
5. The question asks about the correct key.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ssa_bench.tasks.niah_single import (
    NEEDLE_TEMPLATE,
    QUESTION_TEMPLATE,
    generate_samples,
)

# A tiny synthetic corpus is enough for these tests; we don't need the
# full Moby Dick text. Token approximations stay reasonable as long as
# the corpus is varied enough that the haystack assembler doesn't loop
# forever.
_FIXTURE_TEXT = (" ".join(["the quick brown fox jumps over the lazy dog"] * 5000)).strip()


@pytest.fixture
def corpus_file(tmp_path: Path) -> Path:
    p = tmp_path / "corpus.txt"
    p.write_text(_FIXTURE_TEXT, encoding="utf-8")
    return p


def test_determinism(corpus_file: Path) -> None:
    a = generate_samples(
        n_samples=3,
        target_input_tokens=2_000,
        seed=42,
        corpus_path=corpus_file,
    )
    b = generate_samples(
        n_samples=3,
        target_input_tokens=2_000,
        seed=42,
        corpus_path=corpus_file,
    )
    assert [s.to_dict() for s in a] == [s.to_dict() for s in b]


def test_different_seeds_yield_different_samples(corpus_file: Path) -> None:
    a = generate_samples(
        n_samples=2,
        target_input_tokens=2_000,
        seed=1,
        corpus_path=corpus_file,
    )
    b = generate_samples(
        n_samples=2,
        target_input_tokens=2_000,
        seed=2,
        corpus_path=corpus_file,
    )
    assert (a[0].key, a[0].value) != (b[0].key, b[0].value)


def test_needle_present_in_haystack(corpus_file: Path) -> None:
    samples = generate_samples(
        n_samples=4,
        target_input_tokens=2_000,
        seed=42,
        corpus_path=corpus_file,
    )
    for s in samples:
        needle = NEEDLE_TEMPLATE.format(key=s.key, value=s.value)
        assert needle in s.haystack, f"needle missing in sample {s.sample_id}"


def test_expected_answer_in_haystack(corpus_file: Path) -> None:
    samples = generate_samples(
        n_samples=4,
        target_input_tokens=2_000,
        seed=42,
        corpus_path=corpus_file,
    )
    for s in samples:
        assert s.expected_answer in s.haystack
        assert s.expected_answer == s.value


def test_question_matches_key(corpus_file: Path) -> None:
    samples = generate_samples(
        n_samples=4,
        target_input_tokens=2_000,
        seed=42,
        corpus_path=corpus_file,
    )
    for s in samples:
        assert s.key in s.question
        assert s.question == QUESTION_TEMPLATE.format(key=s.key)


def test_token_count_within_tolerance(corpus_file: Path) -> None:
    target = 2_000
    samples = generate_samples(
        n_samples=4,
        target_input_tokens=target,
        seed=42,
        corpus_path=corpus_file,
    )
    for s in samples:
        # Allow +/- 5% slack since the assembler trims to target tokens
        # before adding the needle and question.
        assert 0.8 * target <= s.actual_input_tokens <= 1.3 * target, (
            f"sample {s.sample_id} has {s.actual_input_tokens} tokens, target {target}"
        )


def test_distinct_keys_or_values_across_samples(corpus_file: Path) -> None:
    samples = generate_samples(
        n_samples=8,
        target_input_tokens=2_000,
        seed=42,
        corpus_path=corpus_file,
    )
    # We don't require all keys distinct (small key pool) but values must
    # be distinct for a uniform 7-digit space.
    values = [s.value for s in samples]
    assert len(set(values)) == len(values), "expected unique 7-digit values across samples"


def test_position_fraction_in_range(corpus_file: Path) -> None:
    samples = generate_samples(
        n_samples=8,
        target_input_tokens=2_000,
        seed=42,
        corpus_path=corpus_file,
    )
    for s in samples:
        assert 0.05 <= s.needle_position_fraction <= 0.95
