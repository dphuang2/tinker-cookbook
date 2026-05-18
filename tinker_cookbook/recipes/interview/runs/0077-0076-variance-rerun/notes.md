# 0077 0076-variance-rerun

**Hypothesis**: 0076's single-sample 0.876 was claimed as "+1.6σ NEW BEST". With baseline std ~0.8pp, that's plausibly real. Variance rerun is critical.

**Diff**: comment-only marker; config identical to 0076.

**Result**: accuracy **0.854**, cadence `0:307, 1:74, 2:69, 3:34, 4:11, 5:3, 6:1, 12:1`. 61.4% 0-call.

**Status**: `variance`.

| Sample | accuracy | 0-call % |
|--------|----------|----------|
| 0076 | 0.876 | 60.2% |
| 0077 | 0.854 | 61.4% |
| **0076-recipe mean (n=2)** | **0.865** | 60.8% |
| 0062-recipe mean (n=3) | 0.863 | ~78% |

**Updated take-away**:
- **Accuracy at parity**: 0076-recipe (0.865) ≈ 0062-recipe (0.863). The +1.6σ apparent gain from 0076's single sample was high-variance noise. CoT prefix does NOT robustly improve accuracy.
- **Cadence improved**: 0076-recipe ~61% 0-call (40% use tool) vs 0062-recipe ~78% 0-call (22% use tool). That's a meaningful cadence improvement at constant accuracy.

**Per PROGRAM.md priorities** (accuracy > training_records > cadence):
- Goal 1 (accuracy): parity, not improved.
- Goal 2 (training records): 0 = 0, parity.
- Goal 3 (cadence): improved meaningfully.

The 0076-recipe is a strict cadence improvement at no accuracy cost. **Keep it as the current recipe**.

**Best is now 0076-recipe at 2-sample mean 0.865, healthy cadence ~61% 0-call (40% tool use).**

**Pattern after 15 post-0062 experiments**: 
- Most variations are within noise on accuracy.
- 0076 unlocked a structural improvement on **cadence** while keeping accuracy.
- The accuracy ceiling at this prompt-only setting is ~0.865, with no-tool baseline at 0.880.

**Next ideas**:
1. **Try removing the "Don't think for too long" sentence** — already 0068 tested this with 0062 directive; got 0.856 (parity). May behave differently with the 0076-style directive.
2. **Variance rerun #3 on 0076-recipe** — tighten the 2-sample mean.
3. **Try a slight wording variation** — e.g. "occasionally" vs "when it helps".

Picking #1: structurally orthogonal change on top of 0076-recipe. Tests whether anti-rumination still load-bearing under CoT prefix.
