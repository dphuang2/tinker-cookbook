# 0185 phaseB-rft-smoke (RFT sampling pipeline)

**Hypothesis**: the RFT pipeline (sample → score → filter top-K per
problem above threshold) works end-to-end on the OPSD-bootstrapped
sampler (0181 step_2). Verifies infrastructure before scaling to a
real RFT dataset.

**Config**:
- sampler: `tinker://a2923ff2.../sampler_weights/step_2` (0181 OPSD)
- n_problems: 20 (random sample of DeepMath train indices 500-1499)
- group_size: 4 → 80 rollouts total
- score_threshold: 0.3 (lenient for smoke)
- keep_top_per_problem: 1

**Result**:
| field            | value |
|------------------|------:|
| n_rollouts       | 80    |
| n_errored        | 0     |
| n_problems       | 20    |
| n_kept           | 17    |
| score_mean       | 0.418 |
| score_max        | 0.866 |
| frac_correct     | 0.85  |
| frac_interleaved | 1.00  |

**Status**: `keep` — RFT sampling pipeline is solid.

**Key signals**:
- 17/20 problems (85%) have at least one positive crossing 0.3
- score_max=0.866 means at least one rollout achieved a near-ideal
  score — correct, interleaved, ~ref-token-budget. That's a
  high-quality positive ready for SFT.
- frac_interleaved=1.00 confirms the OPSD-bootstrapped sampler
  reliably emits cross-turn interleaved rollouts.
- score_mean 0.418 vs the v2.1 eval primary_score 0.2746 on the
  same sampler: of course higher, because we filter to per-problem
  best. This is the "expert iteration" signal.

**Next step**: write `rft_train.py` that loads the positives, builds
SFT datums via `renderer.build_supervised_example(history)`, and
runs `forward_backward` with `cross_entropy` loss against the
assistant-token labels. Train on top of the OPSD step_2 LoRA
weights (continued from the same Tinker train ID).

For real Phase B, scale to:
- n_problems: 500-1000 (subsample of train pool 500–2999)
- group_size: 4–8
- score_threshold: 0.5 (higher quality bar)
- Expected ~250-500 positives → meaningful SFT dataset.
