"""Model adapter layer. Each runner exposes a uniform `run_one` / `run_batch` interface."""

from .base import Runner, RunnerRequest, RunnerResponse

__all__ = ["Runner", "RunnerRequest", "RunnerResponse"]
