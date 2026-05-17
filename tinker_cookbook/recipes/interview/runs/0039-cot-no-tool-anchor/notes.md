# 0039 cot-no-tool-anchor

**Hypothesis**: how much of 0036's gain comes from the tool presence vs CoT alone? Anchor measurement with NO_TOOL=1 + CoT user msg.

**Diff**: SAMPLER_PATH=base, NO_TOOL=1, USER_INSTRUCTION_SUFFIX matches 0036 (CoT + directive about tool that isn't there).

**Result**: accuracy **0.752**, cadence 0:500 (no tool available).

**Comparison**:
| Recipe | acc | tool? | CoT? |
|--------|-----|-------|------|
| **0036** | **0.810** | yes | yes |
| 0024 | 0.798 | yes | no |
| 0033 (no-tool anchor) | 0.772 | no | no |
| 0039 (CoT no-tool anchor) | 0.752 | no | yes |

**Decompose 0036's 0.810**:
- Baseline (no tool, no CoT, agent loop with stale directive): 0.772
- CoT alone (no tool): 0.752 *(actually went down)*
- Tool alone (no CoT): 0.798 (+2.6pp over baseline)
- Tool + CoT: 0.810 (+3.8pp over baseline)

CoT *alone* doesn't help in this harness — it actually slightly hurts (-2pp vs 0033). The tool's presence is required for CoT to land. Interpretation: the tool spec gives the model a reason to organize its reasoning into discrete steps (because each step could lead to a checkpoint call); CoT alone makes the model think more verbosely without structure.

**Status**: analytical (not keep/discard — this is a decomposition).

**Best remains 0036 at 0.810** with the understanding that BOTH the tool spec AND the CoT directive are doing meaningful work.
