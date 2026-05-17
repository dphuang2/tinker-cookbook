# 0005 lora-rank-8

**Hypothesis**: lora_rank=32 has too much capacity, allowing the adapter to overwrite base reasoning behavior. Smaller rank (8) means less weight perturbation → less corruption of base capability while still being enough to learn the format.

**Diff**: sft_train.py `LORA_RANK = 32` → `8`. Also reverted 0004 PURE_MATH_COUNT to 0.

**Training**: 142 steps, final NLL 0.271 (≈ same as 0002's 0.270 — rank doesn't affect train loss much for this dataset size).

**Result**: accuracy **0.728**, cadence `0:169, 1:222, 2:100, 3:8, 4:1`. Slightly worse than 0002 (0.736), better than 0004 (0.702). Cadence shape is similar to 0002.

| Run | accuracy | cadence (0:1:2:3+) | lora_rank | training_records |
|-----|----------|--------------------|-----------|------------------|
| 0002 | **0.736** | 138:250:102:10 | 32 | 2301 |
| 0005 | 0.728 | 169:222:100:9 | **8** | 2301 |

**Status**: `discard`. Smaller rank doesn't help — slightly worse. The hypothesis was wrong: capacity wasn't the issue. The damage is in the training data / format, not in the adapter's representational power.

**Lessons so far**:
- LR: dead knob (0001, 0003 both discard).
- Mixing pure-math: too crude (0004 discard, cadence collapse).
- LoRA rank: doesn't help (0005 discard).
- The thing that mattered: **dropping the duplicated reasoning arg** (0002, +2.8pp).

**Pattern**: format-level changes have moved the needle; hyperparameter knobs have not. Next idea should be another format change.

**Next ideas (more aggressive)**:
1. **Filter teacher records by length** — drop short-thinking records (< 4000 chars) from the SFT dataset. These probably shouldn't have tool calls and may be teaching the model bad cadence on confident problems.
2. **Subsample to 800 records** — data efficiency goal, see if we can hit 0.73+ with much less data.
3. **System prompt tweak** — currently says "between major reasoning steps". Try "only when uncertain" to bias cadence toward fewer calls on easy problems.

Picking #1 (filter short-thinking records). It directly addresses the diagnosis from 0004 (the model is bad on no-tool problems because the training data has tool calls forced on confident problems where they don't belong). Filtering keeps the tool-using bias but removes the bad signal.
