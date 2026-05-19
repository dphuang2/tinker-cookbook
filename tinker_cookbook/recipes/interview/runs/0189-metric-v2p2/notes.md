# 0189 metric-v2.2 — formula tightening

**Change**: drop the `0.5 +` floor in primary_score. v2.2 formula:
```
primary_score = accuracy × interleaving_rate × mean_split_balance × efficiency_factor
```

All four axes are now multiplicative — abandoning any one (e.g.,
dropping interleaving to ~0) tanks the score even if the others
are perfect. This closes the 0188 Goodhart.

Also updated rft_sample.py per-rollout score to the same formula so
RFT iteration can't again select "skip the tool" positives.

**v2.2 recomputed leaderboard (no re-eval needed; metric is a pure
formula over fields already saved in summaries)**:

| run                                      | acc   | int_r | bal   | eff   | v2.1   | **v2.2**  |
|------------------------------------------|------:|------:|------:|------:|-------:|----------:|
| v2.1 base (0172) 0105 prompt-only        | 0.862 | 0.280 | 0.029 | 0.793 | 0.3446 | 0.0056    |
| OPSD 0181 step_2                         | 0.842 | 0.976 | 0.443 | 0.455 | 0.2744 | 0.1656    |
| OPSD 0182 step_4 concise teacher         | 0.842 | 0.974 | 0.441 | 0.449 | 0.2702 | 0.1624    |
| OPSD 0183 step_4 kl=0.25                 | 0.814 | 0.972 | 0.429 | 0.445 | 0.2566 | 0.1510    |
| OPSD 0184 step_4 mode=B                  | 0.828 | 0.976 | 0.419 | 0.422 | 0.2462 | 0.1429    |
| RFT 0186 from-scratch 17 positives       | 0.810 | 0.968 | 0.432 | 0.423 | 0.2430 | 0.1433    |
| **RFT 0187 from-scratch 125 positives**  | 0.834 | 0.638 | 0.515 | 0.672 | 0.3723 | **0.1841** |
| RFT 0188 expert iter (Goodhart)          | 0.770 | 0.022 | 0.302 | 1.000 | 0.3876 | 0.0051    |

**v2.2 standings**:
- **0187 RFT (1st iter) — 0.1841 NEW BEST**
- OPSD step_2 — 0.1656 (close second)
- v2.1 base & 0188 — both ~0.005 (both effectively non-interleaving)

The v2.2 metric correctly identifies 0187 as the best
behaviorally-correct result and exposes the v2.1 baseline as
near-zero on the joint objective.

**Next experiment**: RFT iteration on 0187 with v2.2 scoring in
rft_sample.py. Now that "skip the tool" rollouts score 0, the
filter forces selection of correct + interleaved + balanced
rollouts, breaking the 0188 Goodhart. Sample from 0187 step_32_final
again with `score_threshold=0.2` (modest since pure-mult scores are
much smaller than v2.1).
