# 0006 filter-short-thinking

**Hypothesis**: SFT records where the teacher rewrote a short Qwen3 trace (<4000 chars total thinking) probably encode bad cadence: confident short problems were artificially given progress updates by the teacher. Dropping them should improve cadence on easy problems.

**Diff**: sft_train.py
- Added `MIN_TOTAL_THINKING_CHARS = 4000` constant.
- `InterviewSFTBuilder.__call__` filters records whose `sum(len(turn.thinking))` is below the threshold.
- Reverted `LORA_RANK` to 32.

**Filter stats**: kept 2316 / 2401 records (3.5% dropped). Median thinking is much longer than 4000 chars — most teacher records survived. The filter was too lenient to matter.

**Training**: 136 steps × batch 16 on 2216 train records. Final NLL 0.260.

**Result**: accuracy **0.728**, cadence `138:258:97:7`. Almost identical to 0002 (0.736, 138:250:102:10). Noise-level difference.

| Run | accuracy | cadence (0:1:2:3+) | filter | training_records |
|-----|----------|--------------------|--------|------------------|
| 0002 | **0.736** | 138:250:102:10 | none | 2301 |
| 0006 | 0.728 | 138:258:97:7 | min thinking ≥ 4000 chars | 2216 |

**Status**: `discard`. The filter was too gentle (only 3.5% dropped) to test the hypothesis properly. Accuracy moved within noise.

**Lessons**: at 4000 chars, the filter is a near no-op for this dataset. Should either go much higher (8000+ chars to drop ~25% of records) or filter by a stronger signal (e.g. drop records where teacher emitted 3 calls on a short trace).

**Next ideas**:
1. **Higher threshold (8000+ chars)** — actual filter that drops meaningful chunk of records.
2. **Subsample to 800 records** — directly addresses data-efficiency secondary goal. We're at 0.736 with 2300. Can we get the same with 800?
3. **System prompt tweak** — bias toward 0 calls via the tool description.

Picking #2 (subsample to 800). It directly tests the data-efficiency goal and is structurally different from the filter knob. If we hit 0.73+ with 800 records, that's a big win for the secondary goal.
