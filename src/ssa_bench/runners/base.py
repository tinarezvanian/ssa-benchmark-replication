"""Runner protocol and shared dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class RunnerRequest:
    """A single inference request."""

    sample_id: int
    prompt: str
    max_output_tokens: int = 64
    temperature: float = 0.0
    system: str | None = None


@dataclass
class RunnerResponse:
    """A single inference response."""

    sample_id: int
    text: str
    input_tokens: int
    output_tokens: int
    model_version: str
    error: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class Runner(Protocol):
    """Common interface implemented by every provider runner."""

    name: str
    model: str

    def run_one(self, request: RunnerRequest) -> RunnerResponse:
        """Run a single request synchronously."""
        ...

    def run_batch(self, requests: list[RunnerRequest]) -> list[RunnerResponse]:
        """Run requests via the provider's batch API and wait for completion."""
        ...
