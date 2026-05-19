# 0171 throttle-at-3 (discard)

**Hypothesis**: dropping the throttle threshold from 4 to 3 would
force more rollouts to finalize at the natural cadence anchor (3).
Bet: equal or better primary_score with cleaner distribution.

**Diff**: `eval_deepmath_agent.py` — `len(progress_updates) >= 4`
→ `>= 3`.

**Result vs 0170 (best)**:
- accuracy:          0.874   (was 0.882, −0.8 pp ~flat)
- in_think_rate:     0.002   (was 0.008, ~0)
- turn_split_rate:   **0.046**  (was 0.310, **−26.4 pp collapse**)
- interleaving_rate: 0.048   (was 0.310, −26.2 pp)
- primary_score:     **0.4580** (was 0.5777, **−0.120 → discard**)

**Status**: `discard`. Reverted threshold to 4.

**Why this killed interleaving**:
1. The natural cadence mode is 3 batched-in-one-turn calls.
2. With threshold=3: turn 0 emits 3 batched tool_calls → loop runs
   ack `for tc in tool_calls`. After the 3rd ack, `len(progress_updates)
   == 3 >= 3`, so the throttle message fires.
3. But all three acks went into the same turn's history; the model
   has only seen one turn. Next turn → throttle says "finalize" →
   model writes answer. **No cross-turn cadence develops**.
4. turn_split_rate drops to 0.046 (the few cases where the model
   emitted 3+1 or used a different shape naturally).

**Lesson**: state-aware ack thresholds must be > natural mode to
allow cross-turn cadence. With threshold = mode, the model
batches → gets throttled → finalizes, all in 2 turns. With
threshold > mode, the model gets at least one "continue" ack that
lets it think more in a later turn.

**Implication for RL**: this is exactly the kind of curriculum
question RL needs to answer empirically. A hand-tuned threshold
is brittle; a placement-reward will let the model find its own
sweet spot.

**Next idea**: try threshold = 5 (above 0170's mode at 3–4). This
should preserve all the room 0170 had for cross-turn cadence
while still capping the long tail. Quick comparison to 0170.
