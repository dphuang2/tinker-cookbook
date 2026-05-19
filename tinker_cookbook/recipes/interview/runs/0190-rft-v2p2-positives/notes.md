# 0190 phaseB-rft-v2p2-positives (RFT iter on v2.2-filtered positives — discard)

**Hypothesis**: with the v2.2 score in rft_sample.py (which gives 0
to any skip-tool rollout), filtered positives all have is_interleaved=True.
SFT on these should produce a model that interleaves on held-out
problems too.

**Pipeline**:
1. Sample 800 rollouts from RFT 0187 step_32_final
2. Score with v2.2 (correct × interleaved × split_balance × efficiency)
3. Filter threshold 0.2 → 141 positives kept (all interleaved)
4. From-scratch SFT, 2 epochs, batch=8, 36 steps

**Eval result vs 0187 (best)**:
| metric              | 0187   | 0190   |
|---------------------|-------:|-------:|
| accuracy            | 0.834  | 0.794  |
| in_think_rate       | 0.004  | 0.000  |
| turn_split_rate     | 0.638  | **0.022 (collapsed)** |
| mean_split_balance  | 0.515  | 0.503  |
| mean_total_tokens   | 8187   | 3959   |
| efficiency          | 0.672  | 1.000  |
| **primary_score (v2.2)** | **0.184** | **0.009** |

Cadence: 477/500 skip the tool, 11 use it once, 11 twice, 1 four times.

**Status**: `discard`. SFT on v2.2-filtered (all interleaved)
positives STILL produced a tool-avoiding model. The 0190 model
behavior is qualitatively the same as 0188's Goodhart, but here the
filter cleanly required interleaving.

**Surprising failure mode**: even when every training example
interleaves, the resulting policy abandons the tool. Hypotheses
(any one could explain, or in combination):

1. **Cross-entropy loss with mask-on-assistant ignores tool-call
   tokens** if the renderer's `train_on_what` setting excludes them.
   Need to verify build_supervised_example's behavior.
2. **Small dataset overfitting** — 141 positives × 2 epochs × batch
   8 = 36 steps. The model may be overfitting on the final boxed-
   answer pattern that's common to all positives, and discarding
   the variable in-between tool_call text.
3. **From-scratch LoRA forgets the OPSD-acquired interleaving
   prior**. The 0187 model was already an SFT/RFT product, but
   this 0190 trains FROM SCRATCH on positives derived from it.
   The fresh LoRA doesn't see the inductive bias that made 0187
   work.

**Best fix**: warm-start RFT from 0187 state (not weights).
Requires saving state during 0187 training, which it didn't do
(we only have sampler weights). The 42cce7b commit added state
saving to OPSD; need to do the same for RFT.

**Next experiment**: 0191 — add state-saving to rft_train.py too,
then re-run 0187-equivalent (RFT from-scratch on OPSD positives)
WITH state saving. Then 0192 RFT-warm-start from 0191 state.

**Locked-in v3 best so far**: 0187 step_32_final at v2.2 score
**0.1841** (best behavior-preserving result).
