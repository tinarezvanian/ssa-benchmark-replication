"""NIAH-single (Needle In A Haystack, single needle) task generator.

Implements the task spec from RULER (Hsieh et al., NVIDIA, 2024):
    https://arxiv.org/abs/2404.06654

Each sample consists of:
1. A long haystack of naturalistic text (default corpus: Moby Dick from
   Project Gutenberg, sha256-verified).
2. A single needle injected at a deterministic position within the haystack.
   The needle has the form:
       "One of the special magic numbers for {key} is: {value}."
3. A question asking the model to retrieve the value given the key:
       "What is the special magic number for {key} mentioned in the
        provided text?"
4. An expected answer = the value, scored by exact substring match.

The (key, value, position) for each sample is fully determined by the
(seed, sample_id) pair; same seed produces identical samples.

The haystack source is configurable. The default Moby Dick corpus is
sha256-pinned in METHODOLOGY.md so that anyone reproducing the run is
operating on identical bytes.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import tiktoken

# Public-domain US English nouns and adjective words used for the key.
# Drawn from RULER's published key vocabulary (a small set of common
# English nouns). Keeping the set small keeps samples sharply distinct.
_KEY_WORDS = [
    "abalone", "cardamom", "delight", "egret", "fennel", "garnet",
    "hazelnut", "ivory", "jasmine", "kestrel", "labyrinth", "marble",
    "nectarine", "obsidian", "peregrine", "quartz", "rhodium", "sable",
    "tamarind", "umber", "verbena", "willow", "xenolith", "yarrow",
    "zinnia", "amber", "basalt", "cobalt", "drift", "emerald",
]

# Each sample uses a 7-digit value so that the exact-match check is
# unambiguous and the same digit count is used across all samples.
_VALUE_MIN = 1_000_000
_VALUE_MAX = 9_999_999

NEEDLE_TEMPLATE = "One of the special magic numbers for {key} is: {value}."
QUESTION_TEMPLATE = (
    "What is the special magic number for {key} mentioned in the provided text? "
    "Answer with only the number."
)


@dataclass
class NIAHSample:
    """A single NIAH-single sample.

    The full prompt sent to the model is ``haystack + "\n\n" + question``.
    """

    sample_id: int
    seed: int
    key: str
    value: str
    needle_position_fraction: float
    target_input_tokens: int
    actual_input_tokens: int
    haystack_sha256: str
    haystack: str
    question: str
    expected_answer: str

    @property
    def prompt(self) -> str:
        return self.haystack + "\n\n" + self.question

    def to_dict(self) -> dict:
        return asdict(self)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_corpus_words(corpus_path: Path) -> list[str]:
    """Load corpus and split into whitespace-delimited words for cheap recombination."""
    text = corpus_path.read_text(encoding="utf-8")
    return text.split()


def _build_haystack(
    corpus_words: list[str],
    target_tokens: int,
    encoding: tiktoken.Encoding,
    sample_seed: int,
) -> str:
    """Assemble a haystack of approximately target_tokens tokens.

    Strategy: take a random contiguous slice of the corpus (seeded), then
    tile or truncate until we reach the target token count. Uses tiktoken
    cl100k for token approximation; actual Claude tokens may differ by a
    few percent. The resulting token count is recorded per sample so the
    reported "context length" is the actual measured one, not the target.
    """
    rng = random.Random(sample_seed)

    if not corpus_words:
        raise ValueError("corpus is empty")

    # Repeated approximation. We over-allocate slightly then trim by token
    # count at the end.
    chars_per_token = 4  # cl100k-ish approximation
    target_chars = target_tokens * chars_per_token

    pieces: list[str] = []
    char_count = 0
    while char_count < target_chars:
        # Pick a random starting word; copy a chunk of ~5000 words.
        start = rng.randint(0, max(0, len(corpus_words) - 5000))
        chunk = " ".join(corpus_words[start : start + 5000])
        pieces.append(chunk)
        char_count += len(chunk) + 1

    raw = "\n\n".join(pieces)

    # Trim to target tokens.
    tokens = encoding.encode(raw)
    if len(tokens) > target_tokens:
        tokens = tokens[:target_tokens]
    return encoding.decode(tokens)


def _insert_needle(haystack: str, needle: str, position_fraction: float) -> str:
    """Insert a needle at the given fractional position, snapped to a sentence boundary."""
    if not 0.0 <= position_fraction <= 1.0:
        raise ValueError("position_fraction must be in [0, 1]")
    target_idx = int(len(haystack) * position_fraction)
    # Snap forward to the next sentence boundary (period followed by whitespace).
    match = re.search(r"\.\s+", haystack[target_idx:])
    insert_at = target_idx if match is None else target_idx + match.end()
    return haystack[:insert_at] + needle + " " + haystack[insert_at:]


def generate_samples(
    *,
    n_samples: int,
    target_input_tokens: int,
    seed: int,
    corpus_path: Path,
    encoding_name: str = "cl100k_base",
) -> list[NIAHSample]:
    """Generate a deterministic list of NIAH-single samples.

    Args:
        n_samples: number of samples to generate.
        target_input_tokens: approximate target context length in tokens
            (measured under tiktoken cl100k; Claude's tokenizer differs slightly).
        seed: master seed; same seed produces identical samples.
        corpus_path: path to a UTF-8 text file used as haystack source.
        encoding_name: tiktoken encoding used for length control.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    corpus_words = _load_corpus_words(corpus_path)
    corpus_text = corpus_path.read_text(encoding="utf-8")
    corpus_sha = _sha256(corpus_text)

    master_rng = random.Random(seed)
    samples: list[NIAHSample] = []

    for i in range(n_samples):
        sample_seed = master_rng.randint(0, 2**31 - 1)
        sample_rng = random.Random(sample_seed)

        key = sample_rng.choice(_KEY_WORDS)
        value = str(sample_rng.randint(_VALUE_MIN, _VALUE_MAX))
        position_fraction = sample_rng.uniform(0.1, 0.9)

        haystack = _build_haystack(
            corpus_words=corpus_words,
            target_tokens=target_input_tokens,
            encoding=encoding,
            sample_seed=sample_seed,
        )
        needle = NEEDLE_TEMPLATE.format(key=key, value=value)
        haystack_with_needle = _insert_needle(haystack, needle, position_fraction)
        question = QUESTION_TEMPLATE.format(key=key)

        full_prompt_tokens = len(encoding.encode(haystack_with_needle + "\n\n" + question))

        samples.append(
            NIAHSample(
                sample_id=i,
                seed=sample_seed,
                key=key,
                value=value,
                needle_position_fraction=position_fraction,
                target_input_tokens=target_input_tokens,
                actual_input_tokens=full_prompt_tokens,
                haystack_sha256=corpus_sha,
                haystack=haystack_with_needle,
                question=question,
                expected_answer=value,
            )
        )

    return samples


def write_samples_jsonl(samples: list[NIAHSample], path: Path) -> None:
    """Write samples to a JSONL file (one sample per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(json.dumps(s.to_dict()) + "\n")


def read_samples_jsonl(path: Path) -> list[NIAHSample]:
    """Read samples written by write_samples_jsonl."""
    samples: list[NIAHSample] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            samples.append(NIAHSample(**d))
    return samples
