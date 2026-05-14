# Roadmap

Working notes on phase ordering and what "done" looks like for each step. The README is the public-facing version of this; this doc is for tracking.

## Phase 1: Public-models baselines

Goal: a runnable harness that scores Claude Opus 4.6, Gemini 2.0 Pro 2M, and GPT-5 on RULER 128K, MRCR 8-needle at 1M, and SWE-Bench Verified — with full methodology, raw outputs, and per-sample judge verdicts committed.

### 1a — RULER 128K

- [ ] Clone NVIDIA/RULER into `benchmarks/ruler/upstream/` as a submodule, pinned commit
- [ ] Adapter layer in `runners/` exposing a uniform `(model, prompt, max_tokens) -> response` interface backed by Anthropic, Google, OpenAI SDKs
- [ ] Sample selector: deterministic seed, 100 samples per task type
- [ ] Judge runner using Claude Opus 4.6 with the RULER paper's published judge prompt; per-sample verdicts written to `runs/ruler/<date>/<model>/judgments.jsonl`
- [ ] Score aggregator: per-task and overall, with sample size and CI
- [ ] `./run.sh ruler --model <model>` end-to-end smoke test on 5 samples per task

**Deliverable:** running `./run.sh ruler --model claude-opus-4-6` produces a results JSON the writeup can quote.

### 1b — MRCR 8-needle at 1M

- [ ] Pull the MRCR public dataset to `data/mrcr/`, gitignore the raw, commit a sha256
- [ ] Reuse adapter layer; only Gemini 2.0 Pro 2M can run the 1M tier
- [ ] Scoring: all-8 exact match
- [ ] `./run.sh mrcr --model gemini-2-0-pro-2m --tokens 1048576`

**Deliverable:** a single number with sample size and seed for Gemini at 1M, 8-needle.

### 1c — SWE-Bench Verified

- [ ] Vendor princeton-nlp/SWE-bench at pinned commit
- [ ] Pick agent loop (likely `swe-agent` initially; revisit if cost-prohibitive)
- [ ] Docker eval pipeline per task
- [ ] `./run.sh swe-verified --model <model>` for a configurable subset (cost-aware default = 50 tasks)

**Deliverable:** resolved % on a documented sample of SWE-Bench Verified for one or two baseline models. Full set as budget allows.

### 1d — Phase 1 writeup

- [ ] Blog post: *How I'm setting up to reproduce Appen's SSA benchmark*
- [ ] Cross-post to LinkedIn
- [ ] Drop a non-promotional link in SubQ Discord as a community resource
- [ ] Update the resume's projects section to cite this repo with the Phase 1 results URL

## Phase 2: SubQ comparison (blocked on API access)

### 2a — Add SubQ as backend

- [ ] Add `subquadratic` provider to the adapter layer
- [ ] Pin SubQ model version + API base URL
- [ ] Smoke test all three benchmarks against SubQ on 5 samples each

### 2b — Re-run all benchmarks

- [ ] Re-run RULER 128K on SubQ
- [ ] Re-run MRCR 8-needle at 1M on SubQ
- [ ] Re-run SWE-Bench Verified on SubQ at the same sample size used in Phase 1

### 2c — Phase 2 writeup

- [ ] Blog post: *Reproducing Appen's SSA benchmark — what the API tier actually shows*
- [ ] Side-by-side tables with the Appen numbers and the Phase 1 baselines
- [ ] Honest discussion of any deltas from the Appen report

## Non-goals (intentional)

- Wall-clock latency vs FlashAttention-2 (requires kernel access, B200 hardware)
- FLOP counts
- Training-data audits
- Adversarial benchmarks crafted to find weaknesses

## Cost budget (rough estimate, to be refined before 1a runs)

| Benchmark | Token I/O per run | Models | Est. cost per pass |
|---|---|---|---|
| RULER 128K (13 tasks × 100 samples × ~128K input + ~1K output) | ~170M input + 1.3M output per model | 3 baselines | $400–800 per full pass |
| MRCR 1M (100 samples × ~1M input + ~1K output) | ~100M input + 100K output | 1 baseline (Gemini) | $30–60 per pass |
| SWE-Bench Verified (cost-aware 50 of 500 tasks × ~30K input × ~10 agent turns) | ~15M input + ~150K output | 2 baselines | $50–200 per pass |

Total Phase 1 ballpark: **$500–1100** for one clean pass per model, plus judge calls. Budget needs to be confirmed before commit.
