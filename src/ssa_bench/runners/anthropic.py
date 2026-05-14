"""Anthropic runner.

Supports both the regular Messages API (for smoke-testing) and the Batch
API (for full benchmark runs at 50% list price).

Requires the ``anthropic`` package and an ``ANTHROPIC_API_KEY`` env var.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import anthropic

from .base import RunnerRequest, RunnerResponse

log = logging.getLogger(__name__)

# Pinned model aliases. Update here when versions are bumped; the active
# model_version is recorded on every response and committed to the run
# artifacts.
MODEL_ALIASES: dict[str, str] = {
    "claude-opus-4-6": "claude-opus-4-6-20251015",
    "claude-sonnet-4-6": "claude-sonnet-4-6-20251022",
}


class AnthropicRunner:
    """Anthropic Messages + Batch runner."""

    name = "anthropic"

    def __init__(self, model: str, *, api_key: str | None = None):
        self.model = MODEL_ALIASES.get(model, model)
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def run_one(self, request: RunnerRequest) -> RunnerResponse:
        """Single synchronous Messages call."""
        try:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "max_tokens": request.max_output_tokens,
                "temperature": request.temperature,
                "messages": [{"role": "user", "content": request.prompt}],
            }
            if request.system is not None:
                kwargs["system"] = request.system
            msg = self._client.messages.create(**kwargs)
            text = "".join(
                block.text for block in msg.content if getattr(block, "type", None) == "text"
            )
            return RunnerResponse(
                sample_id=request.sample_id,
                text=text,
                input_tokens=msg.usage.input_tokens,
                output_tokens=msg.usage.output_tokens,
                model_version=msg.model,
                raw=msg.model_dump(mode="json"),
            )
        except Exception as exc:
            log.exception("anthropic run_one failed for sample %s", request.sample_id)
            return RunnerResponse(
                sample_id=request.sample_id,
                text="",
                input_tokens=0,
                output_tokens=0,
                model_version=self.model,
                error=f"{type(exc).__name__}: {exc}",
            )

    def run_batch(
        self,
        requests: list[RunnerRequest],
        *,
        poll_interval_seconds: float = 30.0,
        timeout_seconds: float = 24 * 3600,
    ) -> list[RunnerResponse]:
        """Submit a batch, poll until done, return responses ordered by sample_id."""
        from anthropic.types.messages.batch_create_params import Request

        batch_requests = []
        for req in requests:
            params: dict[str, Any] = {
                "model": self.model,
                "max_tokens": req.max_output_tokens,
                "temperature": req.temperature,
                "messages": [{"role": "user", "content": req.prompt}],
            }
            if req.system is not None:
                params["system"] = req.system
            batch_requests.append(
                Request(custom_id=f"sample-{req.sample_id}", params=params)
            )

        log.info("submitting batch of %d requests", len(batch_requests))
        batch = self._client.messages.batches.create(requests=batch_requests)
        batch_id = batch.id
        log.info("batch submitted: %s", batch_id)

        start = time.monotonic()
        while True:
            batch = self._client.messages.batches.retrieve(batch_id)
            if batch.processing_status == "ended":
                break
            if time.monotonic() - start > timeout_seconds:
                raise TimeoutError(f"batch {batch_id} did not finish in {timeout_seconds}s")
            counts = batch.request_counts
            log.info(
                "batch %s status=%s succeeded=%d errored=%d processing=%d",
                batch_id,
                batch.processing_status,
                counts.succeeded,
                counts.errored,
                counts.processing,
            )
            time.sleep(poll_interval_seconds)

        results_by_sample_id: dict[int, RunnerResponse] = {}
        for result in self._client.messages.batches.results(batch_id):
            custom_id = result.custom_id
            sample_id = int(custom_id.removeprefix("sample-"))
            if result.result.type == "succeeded":
                msg = result.result.message
                text = "".join(
                    block.text for block in msg.content if getattr(block, "type", None) == "text"
                )
                resp = RunnerResponse(
                    sample_id=sample_id,
                    text=text,
                    input_tokens=msg.usage.input_tokens,
                    output_tokens=msg.usage.output_tokens,
                    model_version=msg.model,
                    raw=msg.model_dump(mode="json"),
                )
            else:
                err_type = result.result.type
                err_msg = ""
                if hasattr(result.result, "error"):
                    err_msg = str(result.result.error)
                resp = RunnerResponse(
                    sample_id=sample_id,
                    text="",
                    input_tokens=0,
                    output_tokens=0,
                    model_version=self.model,
                    error=f"{err_type}: {err_msg}",
                )
            results_by_sample_id[sample_id] = resp

        ordered: list[RunnerResponse] = []
        for req in requests:
            if req.sample_id in results_by_sample_id:
                ordered.append(results_by_sample_id[req.sample_id])
            else:
                ordered.append(
                    RunnerResponse(
                        sample_id=req.sample_id,
                        text="",
                        input_tokens=0,
                        output_tokens=0,
                        model_version=self.model,
                        error="missing_from_batch",
                    )
                )
        return ordered
