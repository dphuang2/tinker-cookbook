# 0067 verbose-tool-spec

**Hypothesis**: from 0066, verbose tool spec acts as latent reasoning scaffolding. Test by adding explicit "strong problem-solvers naturally checkpoint after setting up notation, after deriving an intermediate identity, ..., before attempting a verification" rubric. More rubric → more accuracy?

**Diff**: appended 3 sentences of explicit problem-solving rubric language to PROGRESS_TOOL_SPEC description.

**Result**: accuracy **0.856**, cadence `0:360, 1:84, 2:28, 3:14, 4:3, 5:2, 6:2, 7:2, 8:5`. 72% 0-call.

| Run | accuracy | spec verbosity |
|-----|----------|----------------|
| 0066 | 0.850 | minimal (~10 words) |
| 0062 | **0.870** | medium (~60 words) |
| 0067 | 0.856 | maximal (~100 words) |

**Status**: `discard`. -1.4pp vs 0062. The latent-rubric hypothesis is disconfirmed: more verbosity doesn't help, and the curve is non-monotonic with a peak at medium length.

**Interpretation**: 0062's wording sits at an inflection point. Too short (0066) and you lose the reasoning prompt; too long (0067) and the spec starts to feel like a directive to use the tool more, which competes with the "use it sparingly" user-message guidance and likely confuses cadence/accuracy.

**Take-away**: 0062 is at the local optimum for tool-spec wording. Further variations along this axis regress.

**Action**: revert PROGRESS_TOOL_SPEC to 0062's wording.

**Best remains 0062 at 0.870.**

**Next ideas**:
1. **Variance re-run on 0062** — single-sample 0.870 needs corroboration. Std ~0.6-1.7pp per single sample. Critical for declaring final.
2. **Try removing the "Don't think for too long unnecessarily" sentence** from USER_INSTRUCTION_SUFFIX — may be under-using budget on hard problems given high max_tokens=24576.
3. **Inject reasoning rubric into SYSTEM_PROMPT** instead of tool spec.

Picking #2: with 24576 budget unlocked, the "don't think too long" sentence (originally added when max_tokens=8192) may now be counter-productive. Removing it tests whether unleashing the thinking budget helps.
