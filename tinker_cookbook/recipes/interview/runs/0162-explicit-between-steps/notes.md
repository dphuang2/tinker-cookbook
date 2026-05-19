# 0162 explicit-between-steps

**Hypothesis**: explicitly telling the model "Use the checkpoint tool
*between* reasoning steps — pause your thinking, call the tool, then
continue. Don't batch them all at the end." will lift interleaving
without an SFT-induced accuracy collapse.

**Diff**: `USER_INSTRUCTION_SUFFIX` only — added the
`*between* reasoning steps` directive and the explicit anti-batch line.

**Result vs 0161 baseline (in parens)**:
- accuracy:          **0.794**  (was 0.874, **−8.0 pp**)
- in_think_rate:     **0.018**  (was 0.000, **+1.8 pp**)
- turn_split_rate:   **0.268**  (was 0.022, **+24.6 pp**)
- interleaving_rate: **0.274**  (was 0.022, **+25.2 pp**)
- **primary_score:   0.5058**  (was 0.4466, **+0.059 → keep**)

**Status**: `keep`. First v2 row to beat the prompt-only ceiling.

**Observations**:
1. The simple text change "between reasoning steps … don't batch them
   all at the end" produced a real shape change. The chat-template
   prior is breakable from the prompt alone — at least at the
   turn-split level.
2. `in_think_rate` is still tiny (1.8%). Mid-`<think>` tool calls
   remain suppressed by Qwen3's chat-template prior. To meaningfully
   move that, we'd need a custom renderer or special tokens.
3. Cadence variety exploded — tail now includes 24-call, 40-call
   rollouts. Some of those are runaway loops eating turns. The
   accuracy regression likely comes from (a) longer rollouts hitting
   max_turns=8 mid-derivation, (b) the verbose checkpoint pattern
   pushing the model away from its native concise reasoning.
4. The mode at cadence=3 still dominates (272/500), but only 42/500
   skip the tool now (was 259/500). The directive flipped most "skip"
   rollouts into "use it ≥3 times spread out."

**Next idea**: anti-spam clause. Try a recipe that keeps the
"between steps" language but adds "use it sparingly — three or four
times total". The 24+ cadence outliers are likely the accuracy drag.
