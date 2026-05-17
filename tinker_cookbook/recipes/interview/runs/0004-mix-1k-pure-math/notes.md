# 0004 mix-1k-pure-math

**Hypothesis**: 0002 (0.736) is still 14pp below the no-SFT baseline (0.880). The likely cause is that SFT on tool-call-heavy data erodes the "just answer cleanly when confident" mode. Mixing in 1000 plain Qwen3 traces (no tool calls) — just `<think>` + boxed answer — should preserve that baseline behavior on no-tool problems.

**Diff**: sft_train.py
- Added `PURE_MATH_PATH` constant + `PURE_MATH_COUNT = 1000`.
- Added `pure_math_record_to_datum()` helper that builds a single Datum from a Qwen3 trace as [tool-spec system + user(question) + assistant(`<think>thinking</think>` + response)].
- Extended `InterviewSFTBuilder` to load N pure-math records (filtered to clean termination) and interleave with the 2401 tool-call records. Total mixed training: 2401 + 1000 = 3401 records (3301 train + 100 test).

**Training**: ~205 steps × batch 16 (longer than 0002's 142 due to more records). Final NLL 0.231.

**Result**: accuracy **0.702** — *worse* than 0002 (0.736) and same as v3 baseline.
Cadence collapsed: `0:432, 1:45, 2:19, 3:4`. 86% of problems now emit 0 tool calls (vs 0002's 28%).

| Run | accuracy | cadence (0:1:2:3+) | training_records |
|-----|----------|--------------------|------------------|
| 0002 | 0.736 | 138:250:102:10 | 2301 |
| **0004** | **0.702** | **432:45:19:4** | **3301** |

**Diagnosis**: mixing pure-math data biased the model strongly toward 0-call behavior. Tool-using subset shrunk from 72% of problems to 14%. The cadence is too degenerate now — even though baseline reasoning may be slightly more preserved, the model is barely using the tool, which violates the behavioral goal. And the no-tool problems aren't getting much better either.

**Status**: `discard`. Pure-math mixing biased cadence too far the other way.

**Lessons**:
- Mixing ratio matters. 1000/2401 ≈ 30% pure-math was too much.
- The right intervention might not be "show no-tool examples" but "show no-tool examples *for confident/short problems specifically*".

**Next ideas**:
1. **Filter teacher records by length** — drop records where the original thinking trace was short (< 4000 chars), since those probably shouldn't have tool calls in the first place. This is a cleaner way to bias cadence without losing the tool-using bias on hard problems.
2. **Smaller `lora_rank`** (8 from 32) — still untried. Less capacity may reduce baseline erosion without changing data distribution.
3. **Subsample training data** — e.g. 800 records from 2301 (data-efficiency goal). Cheap test.

Picking #2 (smaller LoRA rank) — it's the next "structural" knob I haven't tried, complementary to data composition, and orthogonal to LR (which is dead).
