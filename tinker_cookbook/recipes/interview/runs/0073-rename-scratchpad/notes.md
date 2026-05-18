# 0073 rename-scratchpad

**Hypothesis**: rename `checkpoint` tool → `scratchpad`. Different connotation from 0029's "note_to_self" (working area vs internal monolog). Test whether name semantic affects accuracy/cadence.

**Diff**: tool name + all references in description and USER_INSTRUCTION_SUFFIX renamed.

**Result**: accuracy **0.858**, cadence `0:373, 1:87, 2:26, 3:5, 4:3, 5:2, 8:4`. 74.6% 0-call.

| Run | tool name | accuracy |
|-----|-----------|----------|
| 0011/0062 | checkpoint | 0.867 (mean) |
| 0029 | note_to_self | 0.780 (-9pp) |
| 0073 | scratchpad | 0.858 (-0.9pp) |

**Status**: `discard`. -0.9pp vs 0062 mean (borderline within noise but slightly worse). Cadence shape healthy and similar to 0062.

**Interpretation**: "scratchpad" lands ~1pp below "checkpoint" — much milder than 0029's "note_to_self" disaster. Naming the tool with a working-area connotation doesn't actively suppress reasoning the way "note_to_self" did, but doesn't help either. "Checkpoint" remains the optimal name; possibly because checkpointing is a specific structured-reasoning concept the model encountered in pre-training.

**Take-away**: tool naming is a sub-component of the optimum; "checkpoint" carries useful associations that other names lack.

**Action**: revert tool name to "checkpoint".

**Best remains 0062 at 0.867 (2-sample mean).**

**Pattern**: prompt-only exhaustion at 12 post-0062 variations. All regress or are within noise.

**Next ideas**:
1. **Try richer tool ack** — "noted: <summary echoed back>" instead of plain "ok". Tests whether echoing the checkpoint helps the model integrate it.
2. **Try tool name "verify"** — different semantic again (action-verb vs noun).
3. **Try tool name "note"** — shortest, most neutral.

Picking #1 (richer tool ack): structurally orthogonal to all prior name/description changes. Modifies the agent loop itself.
