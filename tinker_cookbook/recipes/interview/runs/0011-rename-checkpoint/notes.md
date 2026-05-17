# 0011 rename-checkpoint

**Hypothesis**: the tool name `progress_update` and the description ("so the user can follow along") frames the tool as user-facing communication, which the model may suppress on problems where it thinks "the user doesn't need an update". Renaming to `checkpoint` with a framing of "for YOUR OWN bookkeeping" should reduce that suppression instinct.

**Diff**:
- teacher_rewrite.py: reverted to the 0002-era cadence prompt (~1 per 4000 chars, up to 3, fewer is better) — needed because 0009/0010 left it on a sparser variant.
- sft_train.py: ToolSpec renamed `progress_update` → `checkpoint`; description rewritten to emphasize self-bookkeeping ("for YOUR OWN bookkeeping while you work through the problem -- use it whenever you finish a logical subtask, switch approach, or want to consolidate progress. Call it freely; the user will read the summaries to follow along.").
- Regenerated sft_dataset.json. Cadence matches 0002 (~64% 3-call, ~24% 2-call).

eval_deepmath_agent.py parses tool calls by structure regardless of name, so no eval change needed.

**Training**: 142 steps, final NLL 0.239.

**Result**: accuracy **0.740**, cadence `0:145, 1:237, 2:110, 3:8`. **+0.4pp over 0002 (0.736)**.

| Run | accuracy | cadence (0:1:2:3+) | tool name |
|-----|----------|--------------------|-----------|
| 0002 | 0.736 | 138:250:102:10 | progress_update |
| **0011** | **0.740** | 145:237:110:8 | **checkpoint** |

Cadence is essentially identical. The accuracy bump is small (within noise of a 500-problem eval — ±2pp is ~95% CI for 0.7) but it doesn't regress, so the tool-rename is at least neutral.

**Status**: `keep`. Marginal but not negative; tool naming is now `checkpoint` going forward. The framing change at least doesn't hurt.

**Observation**: this is the first improvement (or non-regression) we've seen in 9 experiments since 0002. The accuracy gap from baseline (0.880) remains ~14pp.

**Next ideas**:
1. **Tighter description** — push "self-bookkeeping" framing even harder, or explicitly say "the user does not see these".
2. **System prompt + checkpoint name** — maybe the system prompt's tools instructions interact with the name. Try a custom system prompt.
3. **Combine 0011 + smaller `lora_rank`** — 0005's small-rank attempt may have been mismeasured against the old tool name; retry with `checkpoint`.

Picking #3 (lora_rank=16, intermediate between 8 and 32) — easy to test, complementary to the format-rename win.
