# 0165 trimmed-between-steps (NEW BEST)

**Hypothesis**: the 0163 prompt was too long; the verbosity itself
was costing accuracy. Trim while keeping "*between* reasoning steps"
and "about three checkpoints" as the load-bearing phrases.

**Diff**: `USER_INSTRUCTION_SUFFIX` cut from ~85 words to ~40, kept
"*between* reasoning steps", "pause, summarize where you are in one
sentence, then keep thinking", "about three checkpoints spread
through your work is typical", "Write the boxed answer as soon as you
have it."

**Result vs 0163 baseline**:
- accuracy:          **0.812**  (was 0.736, **+7.6 pp**)
- in_think_rate:     0.004   (same)
- turn_split_rate:   0.316   (was 0.396, **−8.0 pp**)
- interleaving_rate: 0.316   (was 0.398, **−8.2 pp**)
- **primary_score:   0.5343** (was 0.5145, **+0.020 → new best**)

**Status**: `keep` — new v2 best.

**Observations**:
1. Accuracy is the dominant lever. Cutting prompt length recovered
   7.6pp accuracy by losing only 8pp interleaving — net positive on
   primary_score.
2. Cadence tail tamed: only 6 rollouts at exactly 24 calls (was 47);
   max single rollout is 43 (was 40). The shorter directive doesn't
   trigger the "checkpoint on every micro-step" failure mode.
3. `in_think_rate` is still pinned at ~0.004. Prompt length doesn't
   move it. Confirms the chat-template prior is the real obstacle.

**Where we are**:
- v1 ceiling (0161 baseline): 0.5(0)–0.4466
- 0162 explicit-between-steps:  0.5058
- 0163 cap-at-four:             0.5145
- 0164 exactly-three:           0.2280  (discard)
- **0165 trimmed:               0.5343**

Prompt-only is plateauing around 0.53. Without breaking the in_think
wall, we cap around (accuracy × 0.66) ≈ 0.55 even at perfect
turn-split. To clear that, need a format change or RL.

**Next idea**: one more cheap prompt swing — try moving the
placement directive from the user suffix into the SYSTEM_PROMPT
(currently barely says anything about cadence). If system-prompt
placement works better than per-turn user suffix, that'd inform RL
prompt design. Otherwise, move to RL infra.
