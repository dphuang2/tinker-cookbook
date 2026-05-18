# 0079 0076-variance-rerun-3

**Hypothesis**: 3rd sample to tighten 0076-recipe variance estimate.

**Result**: accuracy **0.852**, cadence `0:301, 1:82, 2:76, 3:27, 4:7, 5:3, 6:2, 8:1, 11:1`. 60.2% 0-call.

**Status**: `variance`.

| Recipe | n | samples | mean | std |
|--------|---|---------|------|-----|
| 0062 | 3 | 0.870, 0.864, 0.854 | 0.863 | 0.008 |
| 0076 | 3 | 0.876, 0.854, 0.852 | **0.861** | **0.014** |

**Critical update**: 0076-recipe std is wider (1.4pp) than 0062-recipe (0.8pp). Possibly because CoT prefix adds variance in reasoning paths.

**Recipe comparison (final)**:
- **Accuracy**: 0062 (0.863) ≈ 0076 (0.861). **Statistical tie.**
- **Cadence**: 0076 (~61% 0-call, 40% tool use) better than 0062 (~78% 0-call, 22% tool use).
- **Training records**: both 0.

**Per PROGRAM.md priorities** (accuracy > training_records > cadence):
- Goal 1 accuracy: tie.
- Goal 2 records: tie.
- Goal 3 cadence: 0076 wins.

**Best is 0076-recipe** by cadence tiebreaker. 3-sample mean 0.861, 40% tool use.

**Final summary after ~17 post-0062 experiments**:
- Prompt-engineering ceiling on accuracy: ~0.862, irrespective of recipe nuances.
- No-tool baseline: 0.880.
- Structural gap: ~1.8pp from having the tool spec in the prompt.
- Tool spec wording, name, suppressor, anti-rumination, CoT prefix all explored.
- Conclusion: prompt-only is saturated. To improve from here would need:
  - RL with format + correctness reward (untried; expensive).
  - A different recipe paradigm (DPO? Constitutional fine-tuning?).

**Next ideas (running thin)**:
1. **Declare 0076 final**.
2. **Try one more orthogonal: smaller MIN_TOTAL_THINKING_CHARS filter via SFT** — already tried via 0028 (catastrophic). 
3. **Add reasoning-length cap** — limit per-turn thinking; might counter the wider variance.

Picking #3: structurally distinct, addresses the variance directly.
