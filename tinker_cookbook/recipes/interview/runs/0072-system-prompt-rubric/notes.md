# 0072 system-prompt-rubric

**Hypothesis**: move reasoning emphasis to SYSTEM_PROMPT (orthogonal channel from user-message directive). Test: "You are a careful mathematician. Solve each problem rigorously: set up notation, derive identities, and verify before answering."

**Diff**: SYSTEM_PROMPT set; USER_INSTRUCTION_SUFFIX unchanged from 0062.

**Result**: accuracy **0.866**, cadence `0:439, 1:38, 2:12, 3:4, 4:2, 5:1, 6:2, 7:1, 8:1`. 87.8% 0-call.

| Run | accuracy | 0-call % |
|-----|----------|----------|
| 0062 / 0070 mean | 0.867 | ~77% |
| 0072 (sys-rubric) | 0.866 | 87.8% |

**Status**: `discard` (parity on accuracy, weaker cadence). Accuracy matches mean exactly but cadence shifted toward less tool use (87.8% 0-call vs ~77%). The rubric in the system channel doesn't add accuracy and *competes* with the tool-availability signal in the user message — fewer tool calls without any compensating gain.

**Take-away**: the recipe is multi-channel-saturated. Adding a system-prompt rubric is neutral on accuracy but harms cadence health.

**Action**: revert SYSTEM_PROMPT to empty.

**Best remains 0062 (2-sample mean 0.867).**

**Pattern after 11 post-0062 experiments**: prompt-engineering thoroughly explored. Every variation either regresses accuracy or weakens cadence. The optimum is *robust* under variance check.

**Next ideas (running thin)**:
1. **Try a small SFT** — but PURE_MATH_COUNT=0 is the current SFT-data setting; explored before with negative results. Could revisit at higher LR or fewer steps.
2. **Try renaming the tool** — already explored (0029 note_to_self lost 1.8pp). But maybe a different name like "scratchpad" would tilt differently.
3. **Try modifying the agent loop tool ack** — currently "ok"; could try richer feedback like "noted" or include the summary back.
4. **Punctuation/formatting only** — cheap, low-info.

Picking #2 (rename to "scratchpad"): cheap and structurally orthogonal to 0029's "note_to_self" rename. Different connotation (working area vs internal monolog).
