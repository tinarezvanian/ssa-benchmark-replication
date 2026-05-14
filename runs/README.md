# Run artifacts

Each run lives in its own dated, model-tagged subdirectory:

```
runs/2026-05-14-niah-single-claude-opus-4-6/
├── config.json       # exact runtime configuration (model, seed, n, tokens, mode)
├── samples.jsonl     # generated samples (deterministic given seed)
├── responses.jsonl   # raw model responses including token usage and any errors
├── verdicts.jsonl    # per-sample correctness verdicts
├── summary.json      # aggregate accuracy + 95% Wilson CI
└── results.md        # human-readable summary
```

All of these are committed: anyone can re-score with a different rubric
or audit any individual sample without re-running the model.

`samples.jsonl` and `responses.jsonl` can grow large at 128K-token
contexts. They are still committed in full so a reader can verify the
exact input the model saw on any given sample.
