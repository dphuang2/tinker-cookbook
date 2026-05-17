# 0027 shorter-directive

**Hypothesis**: 0024's three-sentence directive ("sparingly", "only when you genuinely change approach", "for simple problems, just think and answer") might be wordier than needed. A single sentence might preserve the suppression while reducing prompt overhead.

**Diff**: USER_INSTRUCTION_SUFFIX directive trimmed to one sentence: "Use the checkpoint tool only on hard problems where you change approach."

**Result**: accuracy **0.778**, cadence `0:234, 1:181, 2:48, 3:16, 4:7, 5:6, 6:2, 7:1, 8:5`. 47% 0-call (vs 0024's 82%).

**Status**: `discard`. Shorter directive doesn't suppress tool use as strongly — 47% 0-call vs 82% before — and accuracy drops 2pp.

**Take-away**: the three-sentence wording of 0024 is doing real work. The "for simple problems, just think and answer directly" sentence in particular gives an explicit *default*; without it, the model still defaults to using the tool.

**Best remains 0024 at 0.798.**

**Pattern after 27 experiments**: we've explored prompt engineering thoroughly. The optimum is robust. Headroom to no-tool baseline (0.880) is 8.2pp but appears intractable via prompt-only — every variation around 0024 regresses.

**Next ideas (running thin)**:
1. **Combine 0024 + tiny SFT (e.g. 100 records)** — see if a small adapter on top of prompt-only adds anything.
2. **Try a different `max_tokens_per_turn`** — increasing from 8192 to 12288 might help long-thinking problems finish. (Caveat: changes eval params, breaks comparability with prior rows; but useful one-time test.)
3. **Concede the ceiling** — declare 0024 final, exit loop.

Picking #1 (tiny SFT on top of prompt-only): structurally distinct. Most prior SFT runs were 2300+ records and degraded base reasoning. Maybe 100-200 records, the smallest viable, would just nudge the format without large damage. Cheap to try.
