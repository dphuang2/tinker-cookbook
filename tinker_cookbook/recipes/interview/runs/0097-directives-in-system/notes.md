# 0097 directives-in-system

**Hypothesis**: move ALL directives (CoT, anti-rumination, cadence anchor) to SYSTEM_PROMPT. Empty USER_INSTRUCTION_SUFFIX. Tests channel choice.

**Result**: accuracy **0.880**, cadence `0:431, 1:36, 2:11, 3:10, 4:5, 5:2, 6:2, 7:2, 8:1`. **86.2% 0-call**.

**Status**: `discard` (cadence near-degenerate; tool-use collapsed from ~48% to ~14%).

| Channel for directives | accuracy | tool use % |
|------------------------|----------|-----------|
| User msg (0095) | 0.877 (n=2) | 43% |
| **System msg (0097)** | **0.880** | **14%** |

**Critical finding**: the user-message channel is **load-bearing for cadence** specifically. Putting directives in SYSTEM_PROMPT keeps accuracy but the model treats the tool as a less-mandatory option, suppressing use.

**Take-away**: per priority (1)>(2)>(3) for accuracy>records>cadence, this is technically a tie on accuracy (within noise) but a major cadence regression. Per the autoresearch goal of "interleave progress updates while thinking", the tool-use rate matters — 86% 0-call is approaching the "stealth no-tool" regime.

**Action**: revert. Keep directives in USER_INSTRUCTION_SUFFIX.

**Best remains 0095/0080-recipe with directive in user message.**
