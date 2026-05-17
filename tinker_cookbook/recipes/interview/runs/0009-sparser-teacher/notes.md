# 0009 sparser-teacher

**Hypothesis**: the teacher's 64% rate of 3-call records was biasing the model toward verbose tool use. A sparser teacher (most records get 0-2 calls) should let the model use tools more naturally — fewer false splits on confident problems.

**Diff**:
- teacher_rewrite.py: rewrote the cadence instructions to "VERY SPARSE", cap 2, with explicit guidelines by trace length.
- Regenerated /tmp/tinker-examples/interview/sft_dataset.json via `teacher_rewrite.py`.
- No sft_train.py changes (reverted 0008 epochs to 1).

**New training-data cadence**:
| 0 calls | 1 call | 2 calls | 3 calls |
|---------|--------|---------|---------|
| 751 (31%) | 758 (32%) | 891 (37%) | 1 |

Old training data cadence had 64% with 3 calls and only 3% with 0.

**Training**: 142 steps. Final NLL 0.199.

**Result**: accuracy **0.716**, cadence `0:479, 1:20, 2:1`. CADENCE COLLAPSED: 96% emit 0 calls.

| Run | acc | training cadence (mode) | eval cadence (0-call %) |
|-----|-----|-------------------------|-------------------------|
| 0002 | **0.736** | 3 calls (teacher) | 28% |
| 0004 | 0.702 | mix in 30% no-tool | 86% |
| 0009 | 0.716 | 31% 0-call (teacher) | **96%** |

**Status**: `discard`. Same failure mode as 0004 — once ~30% of training records have 0 calls, the model never wants to call the tool.

**Key insight**: the model has a strong **default toward 0 tool calls**. To overcome it, training data must aggressively bias toward tool use. The teacher's old "force ~3 calls per record" distribution was actually load-bearing — it gave the model permission to use tools.

Concretely: training data needs to *bias the prior* toward calling. Otherwise the model reverts to its native "just answer" mode.

**Next ideas**:
- The "sparse teacher" framing is wrong; revert teacher_rewrite.py to the original prompt, regenerate, see if we reproduce 0002.
- Or try a *biased-mid* teacher: target distribution ~1 call per record (more uniform 1-2 instead of dominant-3). This is a smaller intervention than 0009.

Picking the **mid-cadence teacher**: rewrite the prompt to target exactly 1-2 calls (rarely 0, rarely 3+). This avoids both extremes: not too sparse (which collapses cadence) and not always-3 (which is the 0002 baseline). If this beats 0002 it's the new best. Note: requires another teacher regen.
