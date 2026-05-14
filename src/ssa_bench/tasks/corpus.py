"""Corpus management.

The default haystack corpus is Moby Dick from Project Gutenberg, public
domain in the US. We download once, cache locally, and sha256-verify on
every subsequent read so the bytes are stable across runs.

Source: Project Gutenberg, ebook #2701.
URL:    https://www.gutenberg.org/files/2701/2701-0.txt

We intentionally do NOT vendor the corpus in the repo. The cache lives in
``~/.cache/ssa_bench/`` by default and is gitignored.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

DEFAULT_CORPUS_URL = "https://www.gutenberg.org/files/2701/2701-0.txt"
DEFAULT_CORPUS_NAME = "moby_dick.txt"

# sha256 of the Project Gutenberg Moby Dick text file as of 2026-05-13.
# If Gutenberg updates the file, this hash will change and we should pin
# to a new value or vendor the file explicitly. The hash is verified on
# every load; a mismatch raises an error rather than silently producing
# different samples.
EXPECTED_SHA256 = (
    # NOTE: populated automatically on first successful download; the
    # initial download writes the observed sha to .corpus_sha256 next to
    # the cached file, and subsequent loads pin against it. This avoids
    # hard-coding a hash before we've fetched the file once.
    ""
)


def default_cache_dir() -> Path:
    return Path.home() / ".cache" / "ssa_bench"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_corpus(
    url: str = DEFAULT_CORPUS_URL,
    cache_dir: Path | None = None,
    name: str = DEFAULT_CORPUS_NAME,
) -> Path:
    """Download the corpus if not cached; verify sha256 on every load.

    Returns the path to the cached corpus file.

    On first download we record the observed sha256 to a sidecar file
    ``{name}.sha256``; subsequent loads pin to that value and fail loudly
    if the file changes underneath us.
    """
    cache_dir = cache_dir or default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / name
    sha_path = cache_dir / f"{name}.sha256"

    if not path.exists():
        with urllib.request.urlopen(url) as response:
            data = response.read()
        path.write_bytes(data)

    observed_sha = _sha256_file(path)

    if sha_path.exists():
        pinned_sha = sha_path.read_text().strip()
        if observed_sha != pinned_sha:
            raise RuntimeError(
                f"Corpus sha256 mismatch:\n"
                f"  cached file: {path}\n"
                f"  pinned sha:  {pinned_sha}\n"
                f"  actual sha:  {observed_sha}\n"
                f"Delete the cached file to re-download, or update the pinned sha."
            )
    else:
        sha_path.write_text(observed_sha + "\n")

    return path
