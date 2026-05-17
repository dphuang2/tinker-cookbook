# 0023 optional-tool-prompt

**Hypothesis**: 0022 hit 0.792 by telling the model to use the tool sparingly. Going further — telling it the tool is OPTIONAL, has no effect on grading, and "most problems should have 0 calls" — should suppress tool use even more and recover more of the 8.8pp gap to no-tool baseline.

**Diff**: sft_train.py SYSTEM_PROMPT rewritten:
> "Solve the math problem. Your only goal is to arrive at the correct answer. The checkpoint tool is OPTIONAL -- it exists in case you find it useful for tracking progress on a hard problem, but it has no effect on grading. Most problems should be solved with 0 tool calls. Only call checkpoint if it genuinely helps your reasoning -- never out of obligation. Always end with the boxed answer."

Same prompt-only setup as 0022.

**Result**: accuracy **0.788**, cadence `0:439, 1:35, 2:17, 3:5, 4:2, 8:2`. 88% emit 0 calls. Slightly worse than 0022 (0.792) by 0.4pp.

| Recipe | accuracy | 0-call % |
|--------|----------|----------|
| 0020 (no system prompt) | 0.774 | 45% |
| **0022 (sparing prompt)** | **0.792** | 79% |
| 0023 (optional prompt) | 0.788 | 88% |
| no-tool baseline | 0.880 | n/a |

**Status**: `discard`. Suppression peaks at 0022's framing; pushing further suppresses but doesn't recover accuracy. Local maximum.

**Take-away**: there's a sweet spot in the system prompt around 0022's "sparing but used". Both ends (no prompt, very-optional prompt) underperform it.

**Cadence concern**: at 12% tool-call rate, we're getting close to "model rarely uses the tool". Still has 60/500 problems with 1+ calls (12%) — borderline but still demonstrates the behavior. 0022's 21% is healthier.

**Best remains 0022 at 0.792.** Gap to no-tool baseline: 8.8pp.

**Next ideas**:
1. **Symmetric variation**: try a system prompt that's between 0020 (no prompt) and 0022 (sparing) — a milder anti-distraction. May land somewhere between 0.774 and 0.792 but probably no improvement.
2. **Add a brief instruction before the math problem itself** — instead of the system prompt, prepend to the user message. Tests whether positioning matters.
3. **Combine 0022 + tighter eval params** — e.g. lower temperature for more deterministic answers. (BUT this changes eval params, breaks comparability.)

Picking #2 (move the cadence directive into the user message rather than system prompt). It tests whether the prompt-position matters for biasing tool use. If neutral or better, we learn something. Cheap.
