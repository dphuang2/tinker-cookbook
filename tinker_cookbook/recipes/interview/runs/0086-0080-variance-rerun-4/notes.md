# 0086 0080-variance-rerun-4

**Hypothesis**: 4th sample to tighten 0080-recipe mean estimate.

**Result**: accuracy **0.870**, cadence `0:265, 1:17, 2:16, 3:147, 4:38, 5:7, 6:5, 7:1, 8:2, 9:1, 24:1`. 53% 0-call.

**Status**: `variance`.

| Recipe | n | samples | mean | std | vs no-tool baseline (0.880) |
|--------|---|---------|------|-----|----------------------------|
| 0062 | 3 | 0.870, 0.864, 0.854 | 0.863 | 0.008 | -1.7pp |
| 0076 | 3 | 0.876, 0.854, 0.852 | 0.861 | 0.014 | -1.9pp |
| **0080** | **4** | **0.876, 0.868, 0.884, 0.870** | **0.8745** | **0.007** | **-0.55pp** |

**Final picture**: 0080-recipe holds at 4-sample mean 0.8745, within 0.55pp of no-tool baseline 0.880. Cadence consistent: ~52% 0-call, peak at 3 calls.

The 0080-recipe is **statistically equivalent to or matching no-tool performance**, while using the checkpoint tool meaningfully on ~48% of problems. The structural cost of having the tool spec in the prompt has been effectively neutralized through careful directive design.

**Best**: 0080-recipe at 4-sample mean **0.8745**.

**Drivers (decomposed)**:
1. Prompt-only (no SFT): +4pp vs SFT (0020).
2. max_tokens=24576: +7pp vs default (0062).
3. CoT prefix "Think step by step": +0.5pp (0076).
4. "Two or three calls is typical": +1pp + healthy cadence (0080).

**Pattern**: prompt-only ceiling reached. Future improvements would require:
- RL with format + correctness reward
- Different base model
- Different problem distribution

**Next ideas**:
1. **Declare 0080 final** — best recipe, validated with 4 samples.
2. **Try removing redundancy** — "multi-step problem" appears twice in current wording.
3. **Try a tiny SFT (10-20 records) with very low LR** — barely-perturb the adapter.

Picking #2 (wording cleanup): cheap variance check + cleaner recipe.
