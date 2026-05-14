# SSA Benchmark Replication

An independent, in-progress effort to reproduce [Appen's published benchmark](https://www.appen.com/whitepapers/benchmarking-subquadratics-latest-model-ssa-kernel) of Subquadratic's Sparse Self-Attention (SSA) model from the API side, with full methodology, code, and per-model results.

> **Status:** Phase 1 in progress. Baseline runs on Claude Opus 4.6, Gemini 2M, and GPT-5 (or equivalent SOTA models) are being set up. SubQ access is on the public beta waitlist; Phase 2 (adding SubQ as a backend) starts when access lands.

## Why this exists

Appen published a striking efficiency and quality benchmark of SubQ's SSA kernel: **56× faster than FlashAttention-2 at 1M tokens**, with **86.2%** on MRCR 8-needle at 1M tokens, and **81.8%** on SWE-Bench Verified. Those are notable numbers if they hold across independent replications and across different evaluators.

Two layers of reproducibility are interesting here, and they have different access requirements:

1. **Kernel-level efficiency** (wall-clock vs FlashAttention-2, FLOP counts) — requires direct access to SSA kernel code and matched hardware (B200). Out of scope for this repo.
2. **API-level retrieval and code quality** (RULER, MRCR, SWE-Bench Verified) — requires only API access to the model. **This is what this repo reproduces.**

The retrieval and code-quality benchmarks are the ones developers care about most when picking a model. They are also the ones an independent third-party can validate without a formal evaluator agreement.

## Scope

| Benchmark | Public dataset? | Tier | Status |
|---|---|---|---|
| **RULER** | Yes ([NVIDIA/RULER](https://github.com/NVIDIA/RULER)) | 128K tokens, all 13 task types, 100 samples each | Harness scaffold |
| **MRCR** | Yes | 8-needle, 1,048,576 tokens, 100 samples | Harness scaffold |
| **SWE-Bench Verified** | Yes ([princeton-nlp/SWE-bench](https://github.com/princeton-nlp/SWE-bench)) | Full Verified split, Dockerized patch eval | Harness scaffold |

## Models under evaluation

### Phase 1 (baselines — no SubQ access required)

| Model | Provider | Max context | Why included |
|---|---|---|---|
| Claude Opus 4.6 | Anthropic | 200K | Strongest 128K-class long-context model |
| Gemini 2.0 Pro / 2M | Google | 2M | Only widely available baseline reaching 1M+ context |
| GPT-5 | OpenAI | 256K | Reference baseline at the 128K tier |

### Phase 2 (add when SubQ API access lands)

| Model | Provider | Max context |
|---|---|---|
| SubQ SSA latest | Subquadratic | 1M+ |

## Methodology

See [METHODOLOGY.md](./METHODOLOGY.md) for the full version. Headline commitments:

- **Pre-registered configurations.** Model versions, temperatures, sample sets, and seeds are committed to the repo *before* any results are written down.
- **No prompt engineering between runs.** Same prompts for all models, taken from the published benchmarks' canonical task definitions.
- **LLM-as-judge transparency.** Where the source benchmark uses LLM judges (Claude Opus 4.6 for RULER QA tasks in the Appen study), the judge model, prompt, and per-sample verdicts are committed alongside the raw model outputs.
- **Full raw outputs.** Every model response, every judge verdict, every patch is committed. Anyone can re-score with a different judge or rubric.

## Reproducibility

Every result published from this repo is paired with the exact commit hash of the runner that produced it. To re-run any reported number:

```bash
git checkout <commit>
cp .env.example .env  # fill in API keys
./run.sh <benchmark> --model <model>
```

(Runner scripts land with Phase 1a — RULER setup.)

## Roadmap

### Phase 1: Public-models baselines

- [ ] **1a.** RULER 128K harness — runners for Claude Opus 4.6, Gemini 2M, GPT-5
- [ ] **1b.** MRCR 8-needle at 1M-token harness — runner for Gemini 2M (only baseline reaching 1M)
- [ ] **1c.** SWE-Bench Verified harness — Docker patch-eval, agent loop
- [ ] **1d.** Publish Phase 1 writeup: *How I'm setting up to reproduce Appen's SSA benchmark*

### Phase 2: SubQ comparison (blocked on API access)

- [ ] **2a.** Add SubQ as backend across all three benchmarks
- [ ] **2b.** Re-run all benchmarks
- [ ] **2c.** Publish Phase 2 writeup: *Reproducing Appen's SSA benchmark — what the API tier actually shows*

## Anti-goals

- **Not a competitor benchmark.** This isn't trying to find SSA's weakest case or pick fights with published numbers. It's trying to verify that the published numbers hold under a second independent evaluation.
- **Not a kernel benchmark.** No claims about wall-clock speed or FLOP counts will be made from this repo. Those require kernel access and matched hardware; the Appen report covers them.
- **Not a marketing artifact.** Negative or null results will be published with the same prominence as positive ones.

## Author

Tina Rezvanian · [tinarezvanian.com](https://github.com/tinarezvanian) · [Past the Quadratic Wall](https://github.com/tinarezvanian/LLMs/blob/main/docs/scaling_attention/main.pdf) (companion primer on long-context LLMs)

## License

Code: MIT. Methodology and writeups: CC BY 4.0. Raw model outputs: provided as-is for replication.
