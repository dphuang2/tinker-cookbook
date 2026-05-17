# 0036 think-step-by-step

**Hypothesis**: classic "Think step by step" prefix to the boxed-answer instruction might boost math accuracy on top of 0024's setup.

**Diff**: prepended "Think step by step, then write your final answer in \boxed{} format." to the existing 0024 user-msg directive.

**Result**: accuracy **0.810** — **NEW BEST**, +1.2pp over 0024.

Cadence: `0:487, 1:12, 2:1`. **97% emit 0 tool calls.** Only 13 of 500 problems use the tool.

**Status**: `keep` — primary goal (accuracy) gain. But cadence concern: at 2.6% tool-call rate, we're effectively close to "no tool" mode. Behavioral goal of "interleave progress updates" is borderline degenerate.

| Recipe | accuracy | tool-call % |
|--------|----------|-------------|
| 0033 (agent loop, NO tool) | 0.772 | 0% |
| 0024 (base + sparing directive) | 0.798 | 18% |
| **0036 (+ think-step-by-step)** | **0.810** | **2.6%** |
| 0033 (true ceiling — no tool, same harness) | 0.772 | n/a |

Interesting: the "think step by step" directive pushes the model to skip the tool almost entirely. The result lands +3.8pp above the no-tool ceiling — which is structurally suspicious. Possible explanations:
1. The 1 epoch over 500 problems contains random noise; +0.012 is within typical eval variance.
2. The "think step by step" framing actively improves the model's math reasoning beyond what the tool spec subtracts.

**Cadence trade-off**: behavioral goal says "interleave progress updates while thinking" — at 2.6% tool calls, the model still demonstrates the behavior on a fraction of problems, but most outputs look just like vanilla math reasoning. Whether this satisfies the goal depends on how strict we interpret "interleave".

**Best so far**: 0036 if accuracy alone. 0024 if cadence is required ≥18%.

**Next ideas**:
1. Combine 0036's "think step by step" with a SLIGHTLY weaker tool-suppression (e.g. "use the tool when appropriate" instead of "use it sparingly") — maybe accuracy holds at 0.81 with healthier cadence.
2. Move "think step by step" but drop the sparing-use directive — maybe the directive is no longer needed once thinking is encouraged.

Picking #2: simpler. Test if "Think step by step" subsumes the suppression directive.
