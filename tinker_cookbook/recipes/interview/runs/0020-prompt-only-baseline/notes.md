# 0020 prompt-only-baseline

**Hypothesis**: with 19 SFT experiments hitting a 0.740 ceiling and 14pp below no-SFT (0.880), measure what base Qwen3 does when just exposed to the `checkpoint` tool spec in the prompt — no training. Establishes whether SFT is adding value at all, or just adding format consistency at the cost of accuracy.

**Diff**:
- eval_deepmath_agent.py: added handling for `SAMPLER_PATH=base`, which creates a sampling client without `model_path` (uses base model directly).
- Ran eval with `SAMPLER_PATH=base`. No training step at all. Eval prompt still includes the `checkpoint` tool spec from `PROGRESS_TOOL_SPEC`.

**Result**: accuracy **0.774**.

Cadence: `0:224, 1:203, 2:35, 3:15, 4:6, 5:2, 6:5, 8:10`. Reasonable distribution: 45% 0-call, 41% 1-call, healthy tail.

| Recipe | accuracy | training_records |
|--------|----------|------------------|
| no SFT, no tool (vanilla baseline) | 0.880 | 0 |
| **0020 no SFT, tool in prompt** | **0.774** | **0** |
| 0011 SFT (best SFT) | 0.740 | 2302 |
| v3 SFT (original) | 0.708 | 2302 |

**Status**: `keep`. **MAJOR finding**. Prompt-only beats every SFT recipe we tested by 3.4pp+. The optimal "recipe" within scope of cheap experiments is: **no training, just put the tool spec in the prompt**.

This satisfies both primary goal (accuracy) AND secondary goal (minimize training data: 0 records is the floor).

**Implication**: SFT-LoRA on Kimi-generated teacher data is actively *harmful* to accuracy on DeepMath, regardless of recipe knobs. The base model already knows how to use a well-described tool. The SFT process replaces some of its natural reasoning with format-mimicry, costing 3-15pp depending on knobs.

**What SFT might still be good for**:
- Higher cadence: 0020 has 45% 0-call problems vs 0011's 28%. If you want every problem to emit at least one checkpoint, SFT biases toward that.
- The user goal was "interleave progress updates while thinking" — 0020 still does this on 55% of problems with the tool spec alone.

**Take-away of the whole loop**: the right recipe was hiding in plain sight — exposing the tool in the prompt to base Qwen3. 20 experiments to land on "don't train" is a worthwhile negative result.

**Next ideas (if continuing)**:
1. **Strengthen the prompt-only baseline** further — try different tool descriptions / system prompts on top of the base model. Maybe accuracy can recover closer to 0.880.
2. **Eval cap analysis** — 10 problems hit the 8-call cap. If we raised `max_turns`, those might resolve. Or shrink them by tightening the description.
3. **Investigate whether SFT can complement prompt-only** — e.g., a very small LoRA trained on much fewer (e.g. 100) high-quality records, only adjusting format, not reasoning. Might recover the 11pp gap to 0.880 without the SFT cost.

Picking #1 (tighten the tool description for prompt-only). Cheap to test — same eval, different description. We had a winner (0011's description), now combine prompt-only + 0011 description vs prompt-only + bare description.
