# 0136 0105-variance-rerun-11 — RECIPE CORRUPTION BUG

**Result**: accuracy **0.824** (looked like a wild outlier).

**Status**: `crash` (autoresearcher bug, not recipe variance).

**Root cause**: in 0135 I renamed tool `checkpoint` → `step` using `replace_all`. When 0135 reverted (`step` → `checkpoint`), the `replace_all` of "step" → "checkpoint" was over-eager and corrupted:
- `Think step by step` → `Think checkpoint by checkpoint` (nonsense)
- `multi-step problems` → `multi-checkpoint problems`
- `multi-step problem` → `multi-checkpoint problem`

The recipe ran with the corrupted text in 0136. The model dropped to 0.824 with heavy tool overuse (40 calls in a single problem!) because "Think checkpoint by checkpoint" is nonsensical and probably interpreted as "use the checkpoint tool repeatedly".

**Fix**: manually restored "Think step by step" and "multi-step problem".

**Lesson**: `replace_all` of a 4-letter common word like "step" is hazardous. Use targeted edits instead.

**Action**: this sample is **EXCLUDED** from the 0105 variance estimate. Recipe is back to canonical.

**Best remains 0105 / canonical recipe at 10-sample mean 0.8732.**
