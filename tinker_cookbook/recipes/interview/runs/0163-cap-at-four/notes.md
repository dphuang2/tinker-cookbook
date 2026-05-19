# 0163 cap-at-four (anti-spam attempt)

**Hypothesis**: "aim for three, four is the practical max, don't keep
checkpointing after you've finished" will tame the cadence runaway
from 0162 (40-call rollouts!) and recover accuracy.

**Diff**: `USER_INSTRUCTION_SUFFIX` adds explicit "four is the practical
maximum" + anti-tail-spam wording.

**Result vs 0162**:
- accuracy:          0.736  (was 0.794, **−5.8 pp**)
- in_think_rate:     0.004  (was 0.018, −1.4 pp)
- turn_split_rate:   0.396  (was 0.268, **+12.8 pp**)
- interleaving_rate: 0.398  (was 0.274, **+12.4 pp**)
- primary_score:     0.5145 (was 0.5058, **+0.009 → keep**)

**Status**: `keep` — marginal primary_score gain, but trend is wrong.

**Why this didn't work as intended**:
1. The "four is the practical maximum" wording did NOT cap the
   distribution. Tail is still heavy: 47 rollouts at exactly 24 calls,
   plus 32/35/40-call outliers.
2. `(0, 20)` → only 20 rollouts skipped the tool. The "between steps"
   language pushed nearly everyone into "use the tool a lot" mode.
3. Hypothesis: those 24-call rollouts are hitting `max_turns=8` with
   ~3 batched tool calls per turn. The model is alternating between
   "emit 3 tool calls" turns and... possibly never getting to the
   final answer because every assistant turn ends with tool_calls
   instead of text. That'd explain the accuracy collapse.
4. `in_think_rate` actually *fell* (0.018 → 0.004). The placement
   directive is reorganizing turn structure but not breaking the
   chat-template prior on `<tool_call>` outside `<think>`.

**Next idea**: stop adding more words. Pivot in a different direction.
The prompt-only signal is saturating ~0.50 primary_score. To break the
in_think wall, need either:
(a) a custom renderer / format change, or
(b) RL with placement reward.

For the next experiment, try a HARDER cap with explicit numbers: "Call
the tool *exactly* three times: once after you've understood the
problem, once after the main derivation, once before writing the
answer." If that doesn't pin the cadence to 3, prompt-only is dead and
we move to RL.
