# 0044 0036-rerun-3 (variance)

**Result**: accuracy **0.790**, cadence essentially same as prior reps.

**Three samples of 0036 config**:
| Run | Accuracy |
|-----|----------|
| 0036 | 0.810 |
| 0041 | 0.794 |
| 0044 | 0.790 |
| **mean** | **0.798** |
| std | 0.011 |

95% CI for 0036's true accuracy: approximately **[0.776, 0.820]**.

**Comparison vs 0024** (single sample 0.798):
- 0024 point: 0.798
- 0036 mean ± 95% CI: 0.798 ± 0.022

**Statistically indistinguishable.**

**Final ranking with all goals weighted**:
| Recipe | acc (mean) | tool-call % | records |
|--------|-----------|-------------|---------|
| **0024** (sparing directive, no CoT) | **~0.798** | **18%** | **0** |
| 0036 (CoT + sparing directive) | ~0.798 | 3% | 0 |

Both at the noise-limited ~0.80 ceiling. **0024 wins on tertiary goal (cadence non-degenerate)** since accuracy is tied within noise. 18% tool use clearly demonstrates the "interleave progress updates" behavior the user wanted.

**RECOMMENDED FINAL RECIPE: 0024** — base Qwen3-30B-A3B, no SFT, with:
- PROGRESS_TOOL_SPEC in the prompt (renamed to `checkpoint`, mid-framing description)
- USER_INSTRUCTION_SUFFIX = "Write your answer in \\boxed{} format. Don't think for too long unnecessarily, especially when you have a reasonable degree of confidence. The checkpoint tool is available for tracking progress on hard multi-step problems, but use it sparingly -- only when you genuinely change approach or finish a substantial sub-task. For simple problems, just think and answer directly without calling the tool."
- Empty system prompt.
- 0 training records.

Accuracy 0.798 ± noise. 18% tool-call rate. No training needed.

**Pattern after 44 experiments**: every SFT recipe regressed against base; prompt engineering hit a ~0.80 ceiling that's at the eval noise floor. The "Think step by step" prefix gives a possible tiny lift (untestable at single-eval precision) at the cost of much lower cadence — bad trade for the behavioral goal.

**Status**: 0024 is the final recipe.
