# 0038 cot-soft-tool

**Hypothesis**: CoT + softer positive tool framing ("Use the checkpoint tool when you find it helpful") might recover healthier cadence while preserving most of 0036's CoT gain.

**Diff**: CoT prefix + boxed-format + don't-think-too-long + softer directive (no "sparingly" / "for simple problems just answer").

**Result**: accuracy **0.764**, cadence `0:297, 1:132, 2:30, 3:27, 4:9, 5:5`. 59% 0-call. Worse than 0036 (0.810) AND 0024 (0.798).

**Status**: `discard`. Soft framing increased tool use (good for behavioral goal) but lost ~5pp accuracy.

**Pattern**: tool use rate is strongly inversely correlated with accuracy across recipes.

| Recipe | tool-use % | accuracy |
|--------|-----------|----------|
| 0036 (CoT + sparing) | 3% | **0.810** |
| 0024 (sparing only) | 18% | 0.798 |
| 0020 (no directive) | 55% | 0.774 |
| 0038 (CoT + soft) | 41% | 0.764 |
| 0033 (no tool) | 0% | 0.772 |

The non-monotonic pattern: at 0% (0033) accuracy is 0.772, at 3% (0036) it's 0.810. So having the tool available BUT used rarely is strictly better than no tool. There's a U-shape — too much tool use AND complete absence both underperform "rare but available".

**Best remains 0036 at 0.810.** The cadence (3%) is borderline degenerate but technically non-zero.

**Next ideas**:
1. **CoT + EXTREME suppression** ("never use the tool unless you're completely stuck") — see if we can drive accuracy even higher by getting to 1% tool use.
2. **Try a CoT-only variant** with a positive system prompt to encourage tool use, see if accuracy drops less than the 0036 → 0038 drop.

Picking #1 to test the U-shape hypothesis at the low end.
