# 0014 filter-8k

**Hypothesis**: 0006's filter at 4000 chars was a no-op (dropped only 3.5%). At 8000 chars the filter drops ~22% of records (the shortest-thinking ones). Those records may be teaching the model bad cadence on confident problems where the teacher artificially forced 2-3 tool calls.

**Diff**: sft_train.py `MIN_TOTAL_THINKING_CHARS = 0 → 8000`. Reverted `LORA_RANK` to 32.

**Filter stats**: kept 1878/2402 (78%), dropped 524 (22%). Real filter this time.

**Training**: 110 steps × batch 16 on 1778 train records. Final NLL 0.234.

**Result**: accuracy **0.710**, cadence `0:134, 1:279, 2:79, 3:8`. Worse than 0011 (0.740) by 3pp.

| Run | accuracy | training_records | filter |
|-----|----------|------------------|--------|
| **0011** | **0.740** | 2302 | none |
| 0014 | 0.710 | 1778 | min thinking 8000 chars |

**Status**: `discard`. Dropping the 22% shortest-thinking records hurt — even though they intuitively shouldn't have tool calls, they were load-bearing for accuracy.

**Lesson**: filters on thinking length aren't a good way to improve quality. Either the short-thinking records contribute useful signal we don't understand, or removing them just makes the dataset smaller (which we now know hurts: see 0007).

**Pattern**: training records aren't fungible. Subsampling to <2000 records always hurts. 2300 is the sweet spot.

**Next ideas (untried, getting more creative)**:
1. **Smaller, more carefully selected dataset** — instead of filtering by thinking length, filter by *teacher cadence* (drop the 0/1-call records, keep the 2-3-call records to bias more strongly toward tool use).
2. **Larger SFT dataset** — generate more raw traces (extend sample_deepmath_train.py range) and rewrite, get to ~4000 records. May break the 2300-record plateau.
3. **Tighter `max_length`** — force truncation of overly long traces. Could either help (less noise) or hurt (less context).
4. **Stronger "self-bookkeeping" framing on the tool description.

Picking #2 (larger dataset to 4000 records): structurally different from filter knobs (which are dead), tests whether we're data-bound. Requires sampling 1500 more raw traces + rewriting them. ~30 min for sampling + ~10 min for teacher + 15 min train + 10 min eval = ~65 min total. Longer than usual cycle but valuable test.

Actually, given the time cost, let me instead pick:
1'. **Filter by cadence (keep only 2-3-call records)** — fast, tests the bias-toward-tool-use hypothesis from a different angle than 0010.

Going with 1' (filter-keep-multi-call): fast, directly tests whether the dataset's 0/1-call records are pulling down eval cadence.
