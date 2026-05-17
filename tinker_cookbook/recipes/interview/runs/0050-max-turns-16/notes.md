# 0050 max-turns-16

Bumped MAX_TURNS 8 → 16.

Acc 0.796, cadence 415:48:22:8:0:4:2:0:1. No problem hit the 16-cap. Cadence distribution similar to 0024 baseline.

Within noise of 0024 (~0.797). Discard, revert MAX_TURNS to 8.

The 8-turn cap is not binding for prompt-only with sparing directive. Raising it provides no benefit.

Best remains 0024.
