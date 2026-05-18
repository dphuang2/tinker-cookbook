# 0076 cot-with-soft-cadence — NEW BEST

**Hypothesis**: 0064 showed CoT prefix gives real +1pp accuracy but the "use sparingly / for simple problems just answer directly" suppressor collapsed cadence to 95% 0-call. Can we keep CoT and restore cadence by softening the tool directive?

**Diff**: USER_INSTRUCTION_SUFFIX rewritten to:
- KEEP "Think step by step, then write your final answer in \boxed{} format." (0064 CoT prefix)
- KEEP "Don't think for too long unnecessarily..." (anti-rumination)
- REPLACE "use it sparingly -- only when you genuinely change approach... for simple problems, just think and answer directly" with permissive "Use the checkpoint tool when it helps you organize hard multi-step problems -- one or two calls is plenty for a typical problem."

**Result**: accuracy **0.876**, cadence `0:301, 1:91, 2:64, 3:31, 4:12, 18:1`. 60.2% 0-call (healthy!).

| Run | accuracy | 0-call % | description |
|-----|----------|----------|-------------|
| 0062 / mean | 0.863 | 76% | no CoT, suppressor (prior best) |
| 0064 | 0.876 | 95.4% | CoT + suppressor (degenerate) |
| **0076** | **0.876** | **60.2%** | **CoT + permissive (NEW BEST)** |
| no-tool baseline | 0.880 | 100% | upper bound |

**Status**: `keep`. **+1.3pp / +1.6σ vs 0.863 baseline AND healthy cadence.** Same accuracy as 0064 with vastly healthier tool use (60% 0-call vs 95%).

**Critical insight**: the +1.3pp gain is from the **CoT prefix** alone (NOT from suppressing tool use). The cadence collapse in 0064 was a separate side effect of the strong "for simple problems, just answer directly" sentence. By replacing that with permissive language, we get both:
- accuracy: +1pp (from CoT framing)
- cadence: healthy 60% 0-call, 40% tool use (gain over even 0062's 24%)

This is the true Pareto improvement. CoT prefix unlocks accuracy; permissive guidance unlocks cadence.

**Note**: cadence has one outlier at 18 tool calls in a single problem (max_turns=8 limits agent loop turns, but each turn can emit multiple parallel tool_calls). Not a quality concern.

**Best is now 0076 at 0.876, healthy cadence.**

**Next ideas**:
1. **Variance rerun 0076** — single-sample 0.876 needs corroboration. With std 0.8pp, the gain is +1.6σ (~94% likely real, but worth verifying).
2. **Tighten the cadence directive** — try "two or three calls" vs "one or two calls" wording.
3. **Verify with no-tool comparison** — if 0076 ≈ 0.876 and no-tool baseline ≈ 0.880, gap is only 0.4pp (essentially closed).

Picking #1: variance rerun is critical before any further variation. Most important data point.
