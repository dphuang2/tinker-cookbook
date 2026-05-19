# 0186 phaseB-rft-smoke-eval (RFT pipeline check)

**Hypothesis**: SFT on 17 RFT positives, starting from fresh LoRA,
should produce a model with the same shape as the positives — high
interleaving, ~12k tokens, but improved efficiency vs OPSD because
all positives meet a 0.3+ score threshold.

**Caveats**:
- Trained from scratch (no OPSD warm-start; rft_train.py needs a
  *state* path to resume, which OPSD didn't save before 42cce7b).
- 17 datums is tiny — basically a sanity check.

**Result vs OPSD 0181 step_2**:
| metric              | OPSD step_2 | RFT-from-scratch 5 steps | Δ |
|---------------------|------------:|-------------------------:|---:|
| accuracy            | 0.842       | 0.810                    | −3.2 pp |
| mean_split_balance  | 0.443       | 0.432                    | −0.011 |
| mean_total_tokens   | 12078       | 12988                    | +7.5% (worse) |
| efficiency          | 0.455       | 0.423                    | −0.032 |
| primary_score       | 0.2746      | **0.2433**               | **−0.031** |

**Status**: `keep` (infra). Worse than OPSD because (a) tiny dataset,
(b) from-scratch instead of warm-start. Pipeline works though:
- 17 SFT datums built from positives JSONL ✓
- cross_entropy loss converged (mean_lp -0.42 to -0.45) ✓
- 3 sampler checkpoints saved ✓

**To beat OPSD, we need either**:
1. **Scale**: 500-1000 problems × 4 samples → ~300-500 positives
2. **Warm-start**: chain from OPSD state checkpoint (requires
   re-running OPSD with the new state-saving committed in 42cce7b)
3. **Both** (the real Phase B run)

**Next experiment (0187): rerun a bigger RFT sample** — bump
`n_problems` to 200 and tighten threshold to 0.4. Targets ~150
positives. Same sampler (OPSD 0181 step_2). Then RFT-from-scratch
on those.

After that (0188): rerun OPSD with state-saving to produce a state
path, then RFT-warm-start on top.
