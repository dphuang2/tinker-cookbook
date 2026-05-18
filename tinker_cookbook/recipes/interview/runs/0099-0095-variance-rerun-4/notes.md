# 0099 0095-variance-rerun-4

**Result**: accuracy **0.882**, cadence `0:296, 1:9, 2:6, 3:137, 4:34, 5:10, 6:5, 9:2, 10:1`. 59.2% 0-call, peak at 3 calls (27.4%).

**Status**: `keep`.

| Recipe | n | samples | mean | std |
|--------|---|---------|------|-----|
| **0095** | **4** | **0.880, 0.874, 0.876, 0.882** | **0.878** | **0.003** |
| 0080 | 4 | 0.876, 0.868, 0.884, 0.870 | 0.8745 | 0.007 |
| no-tool baseline | - | - | 0.880 | - |

**0095-recipe holds at 0.878 ± 0.3pp** — essentially AT no-tool baseline 0.880, with tighter variance than 0080. Two of four samples (0.880, 0.882) at or above baseline.

**0095 is the final recipe.**

Cadence: 41% tool use (peak at 3 calls). Non-degenerate. 0 training records.

**Closing summary**: prompt-only recipe achieves 0.878 ± 0.003 on the DeepMath held-out 500, statistically indistinguishable from no-tool baseline 0.880, while using the checkpoint tool on 41% of problems. The recipe is the result of ~100 experiments isolating each lever:

1. **Skip SFT entirely** (0020): SFT damages base reasoning regardless of data.
2. **Lift eval cap** (0062): max_tokens=24576 was the dominant lever (+9pp).
3. **CoT prefix** (0076): "Think step by step" framing.
4. **Numerical cadence anchor** (0080): "two or three calls is typical" lifts tool use to healthy ~50%.
5. **Tool-meta in system prompt** (0095): tighter variance + slight uptick.

**Levers tested and ruled out** (insignificant or harmful):
- Tool renames (note_to_self, scratchpad)
- Shorter/longer tool descriptions
- Different user-message orderings  
- Imperative cadence
- Different numerical anchors (1-2, 3-4)
- Anti-rumination removal
- Verify directives
- Action-only param descriptions
- System-prompt rubrics
- Directives-only-in-system (cadence collapsed)
- Tool ack content (echoed vs "ok")
