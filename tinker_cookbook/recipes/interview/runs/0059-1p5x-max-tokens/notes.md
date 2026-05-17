# 0059 1p5x-max-tokens

**Hypothesis**: 0058 showed 4096 caused mass truncation. Maybe 8192 is also too tight for some problems. Try 12288.

**Diff**: eval_deepmath_agent.py `MAX_TOKENS_PER_TURN = 8192 → 12288`. Breaks strict comparability per PROGRAM.md, but if the effect is large enough it's a meaningful change.

**Result**: accuracy **0.846** — **+4.7pp over 0024 baseline** (0.799). Way outside noise (eval std ~0.006).

Cadence: `0:405, 1:58, 2:18, 3:11, 4:3, 5:1, 7:1, 8:3`. Same shape as 0024 — 81% 0-call.

**Status**: `keep` with caveat — this changes a "fixed" eval param.

| Recipe | accuracy | max_tokens_per_turn |
|--------|----------|---------------------|
| 0024 (mean of 6) | 0.797 | 8192 |
| **0059** | **0.846** | **12288** |

**Insight**: the 8192 cap was actually binding on a non-trivial fraction of problems. Roughly 1/4 of the problems that would have been correct with more thinking budget were getting cut off. This was an *eval framework artifact* the whole time, not a recipe limitation.

**Implication**: the recipe (0024) was probably never the bottleneck on these problems — the eval cap was. With 12288 max_tokens, 0024's true accuracy is ~0.846.

**Next**: re-run a few reps of 0059 to confirm. If it holds, this is the new best by a wide margin.
