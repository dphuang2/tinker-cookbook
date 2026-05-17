# 0008 epochs-2

**Hypothesis**: 0002 at 1 epoch hits 0.736 but final NLL is still 0.27 — maybe undertrained. A second epoch gives 2x gradient steps over the same data; might find a better optimum and recover more accuracy.

**Diff**: `NUM_EPOCHS = 1` → `2`. Reverted 0007's MAX_TOOL_RECORDS to 0.

**Training**: 283 steps × batch 16. Final NLL 0.203 (lower than 0002's 0.270, as expected from a 2nd epoch).

**Result**: accuracy **0.704**, cadence `0:178, 1:209, 2:93, 3:20`. Worse than 0002 (0.736).

| Run | accuracy | nll | epochs |
|-----|----------|------|--------|
| 0002 | **0.736** | 0.270 | 1 |
| 0008 | 0.704 | 0.203 | 2 |

**Status**: `discard`. 2 epochs overfits — lower training NLL but worse generalization. The 1-epoch sweet spot at 0002 is real.

**Pattern recognition** (after 8 experiments):

| Knob | Direction | Result |
|------|-----------|--------|
| LR up/down | both | dead |
| `reasoning` arg | drop | **+2.8pp** (only winner) |
| Pure-math mix | add | cadence collapse |
| LoRA rank | down | slightly worse |
| Filter short traces | mild | no effect |
| Subsample records | down | cadence collapse |
| Epochs | up | overfit |

The only winning move was a *data/format* change (drop reasoning arg, keeping the message-only format). Hyperparameter knobs are dead.

**Next ideas (untried)**:
1. **Smaller-still LoRA rank** (4) — to confirm 0005's finding more strongly.
2. **System prompt tweak** — change the tool description to bias cadence.
3. **Regenerate teacher data with stricter cadence** — re-run teacher_rewrite.py with lower max-calls and stricter "fewer rather than more" framing. This is a real data change.
4. **DPO** — preference pairs vs. SFT.

Picking #3 (regenerate teacher with stricter cadence): the entire dataset has been the same since v3, and 4 of 8 experiments hit cadence problems. The teacher's tendency to emit 3 splits on most records may be over-training the tool-use bias. Lower-cadence teacher data + same SFT setup may help. Cost: an extra ~5 min for the regenerate step.
