# 0034 tool-ack-continue

**Hypothesis**: a more useful tool ack ("Acknowledged. Continue your reasoning." vs "ok") might reduce context-switch cost on tool calls and improve resumption.

**Diff**: eval_deepmath_agent.py tool-response content "ok" → "Acknowledged. Continue your reasoning."

**Result**: accuracy **0.798**, cadence `0:404, 1:80, 2:10, 3:2, ...`. Identical to 0024 within noise.

**Status**: `discard`. No effect. Adds complexity without benefit — revert to "ok".

**Take-away**: the tool ack content doesn't matter at this scale. The model handles either equivalently.

**Best remains 0024 at 0.798.** Reverting the tool ack change.
