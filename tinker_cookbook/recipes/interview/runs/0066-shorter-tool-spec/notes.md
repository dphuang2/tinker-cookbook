# 0066 shorter-tool-spec

**Hypothesis**: trim the PROGRESS_TOOL_SPEC description from ~60 words to one sentence (and parameter description similarly). Reducing prompt overhead should free thinking budget and improve accuracy.

**Diff**: tool description shortened from 5-sentence "Pause your thinking ... read along" to single sentence "Record a one-sentence checkpoint of your current reasoning state." Parameter description trimmed from full u-substitution example to "One short first-person sentence."

**Result**: accuracy **0.850**, cadence `0:392, 1:61, 2:28, 3:8, 4:3, 5:1, 6:1, 7:2, 8:4`. 78% 0-call (similar shape to 0062).

| Run | accuracy | tool spec verbosity |
|-----|----------|---------------------|
| 0062 | **0.870** | full (5 sentences, encouraging "call freely") |
| 0066 | 0.850 | minimal (1 sentence) |

**Status**: `discard`. -2pp vs 0062. The terse spec actually hurts accuracy.

**Interpretation**: the longer description in 0062 wasn't just bookkeeping — phrases like "use it whenever you finish a logical subtask, switch approach, or want to consolidate progress" likely act as inline thinking-rubric cues that nudge the model toward more structured reasoning even when it doesn't call the tool. Removing them reduces accuracy. The tool description doubles as reasoning scaffolding.

**Take-away**: prompt-tokens-as-overhead is the wrong mental model — verbose tool specs can act as latent prompts that improve reasoning quality. Shorter ≠ better.

**Action**: revert PROGRESS_TOOL_SPEC to 0062's verbose wording.

**Best remains 0062 at 0.870.**

**Next ideas**:
1. **Variance re-run on 0062** — confirm 0.870 is robust, not a high-variance peak. Single-sample std ~0.6-1.7pp; needs corroboration before declaring final.
2. **Inverted hypothesis test**: make tool spec even MORE verbose with explicit reasoning-rubric language. If verbose helps, even-more-verbose might help more.
3. **System prompt with reasoning rubric** — push the latent rubric into SYSTEM_PROMPT instead of tool spec. Decouples from cadence pressure.

Picking #2 (more verbose spec): direct test of the latent-rubric hypothesis. Cheap, structurally different from earlier runs.
