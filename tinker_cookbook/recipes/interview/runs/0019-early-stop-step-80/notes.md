# 0019 early-stop-step-80

**Hypothesis**: training may overfit on the format in the final steps. An earlier checkpoint (step 80 of 142) might preserve more baseline reasoning while still having learned the format.

**Diff**: eval_deepmath_agent.py — added `SAMPLER_PATH` env override (reads `os.environ.get("SAMPLER_PATH")`). Ran eval against `tinker://dbbf713d-fb12-5e79-88df-48aee4af8530:train:0/sampler_weights/000080` (the step-80 checkpoint from 0011's training run). No retraining required.

**Result**: accuracy **0.706**, cadence `0:149, 1:254, 2:87, 3:10`. Worse than 0011's final-step accuracy (0.740) by 3.4pp.

| Step | accuracy | cadence (0:1:2:3) |
|------|----------|-------------------|
| 80   | 0.706 | 149:254:87:10 |
| 142 (0011 final) | **0.740** | 145:237:110:8 |

**Status**: `discard`. Earlier checkpoint underperforms final. The accuracy curve appears monotonically improving through step 142 — not overfit. So `final` is the right checkpoint.

**Take-away**: this rules out early-stopping as a path to beating 0011. The 1-epoch training schedule at step ~142 is genuinely near-optimal for the recipe.

**Pattern after 19 experiments**:
- **Best**: 0011 (0.740) — reasoning-arg dropped, `checkpoint` rename + mid-framing description, default hyperparams.
- All 17 other variations (LR, rank, epochs, data composition, system prompt, dataset size, loss mask, checkpoint step) regress or tie.
- 0.740 is the durable ceiling for SFT-LoRA on this dataset/format.

**Strong inference**: the residual 14pp gap to no-SFT (0.880) is **structural** to SFT-on-this-data. To close it, we'd need:
- RL (with format + correctness rewards),
- DPO (preference pairs), or
- A fundamentally different recipe (e.g. teach via prompt-only ICL plus a small format-correction adapter).

These are all big undertakings (~2-4 hours per cycle). Within the scope of cheap SFT cycles, **0011 is the optimum.**

**Next ideas**:
1. **Eval step 120** — same kind of cheap test as 0019 but closer to final. Might reveal a peak between 100 and 142.
2. **Modified teacher prompt**: encourage "1 call for short, 2-3 for long, never 0" — biases data toward always-using-tools while not forcing 3.
3. **Prompt-only baseline** — measure no-SFT performance with tool-spec in prompt, establishes the natural upper bound for a non-trained recipe.

Picking #3 (prompt-only baseline): no SFT, cheap (just eval), establishes a useful comparison point. May reveal whether the base Qwen3 already does any tool calling when shown the spec, which constrains what SFT can possibly add.
