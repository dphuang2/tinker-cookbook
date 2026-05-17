# 0029 note-to-self

**Hypothesis**: rename `checkpoint` → `note_to_self` to lean further into "internal" framing. Same long description and user directive otherwise.

**Diff**: tool name + matching renames in description / user directive.

**Result**: accuracy **0.780**, cadence `0:361, 1:95, 2:25, 3:7, ...`. 72% 0-call (vs 0024's 82%). The "note_to_self" name actually slightly increased tool use, perhaps because the framing makes it feel more obligatory.

**Status**: `discard`. -1.8pp vs 0024.

**Best remains 0024 at 0.798.**

**Pattern after 29 experiments**: prompt-engineering exhausted. Every variation around 0024 regresses. The optimum is sharp.

**Final summary of the loop**:

| # | Approach | Best | Records |
|---|----------|------|---------|
| v3 | reasoning-arg SFT | 0.708 | 2302 |
| 0002 | message-only SFT | 0.736 | 2302 |
| 0011 | + checkpoint rename SFT | 0.740 | 2302 |
| **0024** | **prompt-only + user-msg sparing directive** | **0.798** | **0** |
| 0028 | tiny SFT | 0.482 | 100 |
| no-tool baseline (vanilla) | 0.880 | 0 |

The key insight (in order discovered):
1. Drop the `reasoning` arg duplication (0002, +3pp).
2. Use a `checkpoint` name + mid-framing description (0011, marginal).
3. **Skip training entirely** — prompt the base model (0020, +4pp).
4. Add a user-message sparing-use directive (0024, +2pp).

**Recipe (final)**: base Qwen3-30B-A3B, no SFT, with PROGRESS_TOOL_SPEC in the prompt and a user-message directive saying "Use the checkpoint tool sparingly -- only when you genuinely change approach...". 0 training records. Accuracy 0.798.

**Next ideas** (genuinely exhausted; remaining ones risk damaging the optimum):
1. Vary the `message` parameter description length.
2. Try renaming the boxed-answer instruction.
3. Test on a different DeepMath slice to verify generalization.

Picking #1 (shorter parameter description). The model spends tokens parsing it during tool calls; a tighter spec might marginally improve.
