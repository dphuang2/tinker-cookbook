# 0007 subsample-800

**Hypothesis**: secondary goal is data efficiency. Test whether ~800 records (1/3 of the 2301 baseline) is enough to learn the tool-using format while preserving accuracy. If yes, big win on the data-efficiency axis.

**Diff**: sft_train.py
- Added `MAX_TOOL_RECORDS = 800` and deterministic shuffle-then-take subsample in `InterviewSFTBuilder`.
- Reverted `MIN_TOTAL_THINKING_CHARS` to 0 (filter disabled).

**Training**: ~46 steps × batch 16 (much faster than 0002's 142). Final NLL 0.288.

**Result**: accuracy **0.710**, cadence `0:481, 1:17, 2:2`. Catastrophic cadence collapse: 96% of problems emit 0 tool calls.

| Run | accuracy | cadence (0:1:2:3+) | training_records |
|-----|----------|--------------------|------------------|
| 0002 | **0.736** | 138:250:102:10 | 2301 |
| 0007 | 0.710 | 481:17:2:0 | **700 (after test split)** |

**Diagnosis**: with only 800 records (post-flatmap ~2800 datums), the model didn't see the tool-using format enough times to commit to it. It mostly defaulted back to no-tool behavior.

The model essentially "knows there's a tool but barely uses it". Accuracy ends up close to no-SFT baseline on no-tool problems, but it's still slightly worse than 0002 because:
1. A handful of tool-call attempts that go wrong (the 19 problems that did call tools).
2. The 481 no-tool problems use the format the model learned — which is *slightly* different from raw Qwen3 because the system prompt with tools still distorts it.

**Status**: `discard`. Subsampling too aggressively kills cadence — same failure mode as 0004 (pure-math mix). Data efficiency requires data quality, not just less data.

**Lessons**:
- 0002 (2301 records) is currently the best at 0.736.
- Aggressive data reduction collapses cadence.
- Need to find a sweet spot in the 1000-2000 range, OR change the *training signal* per record (e.g. weight loss differently) to make 800 records denser.

**Next ideas**:
1. **Subsample to 1500** — gentler reduction. Maybe 800 was too aggressive.
2. **System prompt tweak** — explicitly bias the model to use tools "between major reasoning steps" with stronger language. Compensates for less data.
3. **Increase epochs to 2** on the 0002 baseline — see more gradient steps without changing data, see if accuracy improves.

Picking #3 (2 epochs on 0002 baseline). Cheapest test of whether undertraining is part of the gap. Independent of data composition.
