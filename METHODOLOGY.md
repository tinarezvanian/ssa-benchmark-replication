# Methodology

This document is the source of truth for *how* every published result in this repo was produced. It is committed before results are run and pinned to a tag when each writeup is published.

## Principles

1. **Pre-register the runs.** Model versions, temperatures, sample sets, and seeds are committed to the repo before any benchmark output is recorded. Adjustments after seeing data are tracked as separate commits with an explicit reason.
2. **One prompt, all models.** Task prompts come verbatim from the canonical benchmark definitions (NVIDIA RULER, public MRCR, SWE-Bench Verified). No model-specific prompt tuning between runs.
3. **Full raw artifacts.** Every model response, judge verdict, and generated patch is committed alongside the score summaries.
4. **Honest sample counts.** Where API costs or quotas force smaller sample sizes than the source paper, the reduced size is reported in the same line as the score, not in a footnote.
5. **Negative results are also results.** If a benchmark behaves worse than published numbers (for any model), it is published with the same prominence.

## Model configurations

Pinned at the version published, not the rolling alias. To be updated as releases ship.

| Model | API alias | Pinned version | Temperature | top_p | max_output_tokens |
|---|---|---|---|---|---|
| Claude Opus 4.6 | anthropic / claude-opus-4-6 | (TBD on Phase 1a) | 0.0 | 1.0 | 4096 |
| Gemini 2.0 Pro 2M | google / gemini-2.0-pro-2m | (TBD) | 0.0 | 1.0 | 4096 |
| GPT-5 | openai / gpt-5 | (TBD) | 0.0 | 1.0 | 4096 |
| SubQ SSA (Phase 2) | subquadratic / ssa-latest | (TBD) | 0.0 | 1.0 | 4096 |

Temperature 0.0 is used uniformly because every benchmark we run is deterministic-target: there is a known correct answer (needle present / absent, patch passes tests / not, exact token retrieved / not). Sampling diversity is not relevant.

## Benchmark-specific methodology

### RULER at 128K

- **Source:** NVIDIA/RULER, public.
- **Context length:** 131,072 tokens.
- **Task types:** All 13 (NIAH single/multikey/multivalue, QA single-hop / multi-hop, CWE, FWE, VT).
- **Samples per task:** 100, seeded.
- **Judge:** For QA and word-extraction tasks, Claude Opus 4.6 with the RULER paper's published judge prompt. Judge verdicts committed per sample.
- **Reference numbers (Appen):** Overall 95.6% on the QA + extraction LLM-judged subset. NIAH single = 100%, NIAH multikey degrades by design.

### MRCR 8-needle at 1M

- **Source:** Public MRCR dataset.
- **Context length range:** 524,288–1,048,576 tokens (the largest tier).
- **Samples:** 100 from the published 8-needle bucket.
- **Scoring:** All-or-nothing exact match across all 8 needles (Appen reports a bimodal error pattern that this scoring captures).
- **Reference numbers (Appen):** 86.2% at 1,048,576 tokens, 8-needle hardest tier.

### SWE-Bench Verified

- **Source:** princeton-nlp/SWE-bench, Verified split.
- **Eval:** Docker per-task, run all repository tests, no partial credit.
- **Agent loop:** (TBD on Phase 1c — likely `swe-agent` or a minimal custom loop; pinned by commit hash before runs.)
- **Thinking budget:** "Extended thinking" / equivalent enabled to match Appen's configuration.
- **Reference numbers (Appen):** 81.8% resolved on SWE-Bench Verified.

## What is explicitly *not* attempted here

- **Wall-clock latency vs FlashAttention-2.** Requires SSA kernel source and matched B200 hardware; covered by the Appen report. No claims will be published from this repo about kernel-level efficiency.
- **FLOP measurement.** Same gating as above.
- **Training-data audits.** No claim that any model's training corpus did or did not include any benchmark dataset.

## Reproducibility commitments

For every published number:

1. The commit hash of the runner is included in the writeup.
2. The full raw model output for every sample is in the repo.
3. The judge model's exact verdict per sample is committed.
4. `.env.example` lists every environment variable required to re-run.
5. A single `./run.sh <benchmark> --model <model>` invocation reproduces the run.

## Updates to this document

Material changes to methodology between runs are recorded in `METHODOLOGY-CHANGELOG.md` (added when the first change happens) with rationale.
