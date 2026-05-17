# 0021 minimal-description

**Hypothesis**: 0020's tool description (long, from 0011 era) may add cognitive load that distracts the base model. A minimal description should let the model focus on math.

**Diff**: sft_train.py PROGRESS_TOOL_SPEC trimmed to:
- description: "Record a brief progress note while reasoning."
- message description: "One short sentence on your current progress."

Ran prompt-only eval (SAMPLER_PATH=base).

**Result**: accuracy **0.764**, cadence `0:287, 1:131, 2:30, 3:14, 4:13, ..., 8:16`. Worse than 0020 (0.774) by 1pp, with fewer tool calls (57% 0-call vs 45%).

| Variant | accuracy | 0-call % | description length |
|---------|----------|----------|---------------------|
| **0020** | **0.774** | 45% | long (0011-style) |
| 0021 | 0.764 | 57% | minimal |

**Status**: `discard`. Minimal description hurts both metrics. The longer description in 0020 is doing useful work — biasing the model toward more tool use without sacrificing accuracy.

**Take-away**: tool description content matters, more is better here (within reason). The 0011-style description that we evolved over many SFT experiments turns out to also be the right prompt-only description.

**Best so far**: 0020 (prompt-only, 0011 description) at 0.774 / 0 records.

**Next ideas**:
1. **Add system-prompt instruction** ON TOP OF the 0020 description — bias the model further toward useful tool calls without changing the spec. Cheap test.
2. **Raise max_turns** — 10-16 of the 8-call problems hit the cap. Allowing more turns might let them finish. (Caveat: this changes a "fixed" eval parameter; need to document the trade-off.)
3. **Slight LoRA fine-tune** on 100-200 records (very small) starting from 0020-style behavior — see if a tiny adapter can recover any of the 10pp gap to 0.880 without the format collapse seen in our larger SFT runs.

Picking #1 (system prompt + 0020 description). Same setup as 0020 plus a brief SYSTEM_PROMPT directive. Cheap, structurally distinct from prior SFT system-prompt experiments because 0016 was on SFT not prompt-only.
