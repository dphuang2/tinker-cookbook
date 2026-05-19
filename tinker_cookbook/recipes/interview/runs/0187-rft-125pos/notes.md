# 0187 phaseB-rft-125pos — NEW V3 BEST

**Hypothesis**: SFT (cross-entropy) on 125 RFT-filtered positives
(score>0.4) from the OPSD 0181 step_2 sampler will beat both OPSD
and v2.1 prompt-only baselines on primary_score, because the SFT
data is already filtered to the v2.1 metric's components.

**Pipeline**:
1. Sample 800 rollouts from OPSD step_2 on 200 train problems
   (`rft_sample.py`)
2. Filter to top-1 per problem with score>0.4 → 125 positives
3. From-scratch LoRA SFT, 2 epochs, batch_size=8, 32 steps total
   (`rft_train.py`)

**Result vs all prior bests**:
| metric              | v2.1 base (0172) | OPSD best (0181) | **RFT (0187)** |
|---------------------|-----------------:|-----------------:|---------------:|
| accuracy            | 0.862            | 0.842            | **0.834**      |
| in_think_rate       | 0.012            | 0.000            | 0.004          |
| turn_split_rate     | 0.280            | 0.976            | 0.638          |
| interleaving_rate   | 0.280            | 0.976            | 0.638          |
| **mean_split_balance** | **0.029**     | 0.443            | **0.515**      |
| **mean_total_tokens** | 6939          | 12078            | **8187**       |
| efficiency          | 0.793            | 0.455            | 0.672          |
| **primary_score**   | **0.3444**       | 0.2746           | **0.3722**     |

**Status**: `keep` — **+0.098 vs OPSD, +0.028 vs v2.1 baseline.**
First v3 result to beat v2.1 prompt-only.

**What RFT learned**:
- Cadence collapsed to lower mode: was 4-call dominant under OPSD,
  now 2-call dominant (216/500). 1-call also common (137/500).
- Even though `turn_split_rate` dropped from 0.976 → 0.638, the
  rollouts that DO split have **better balance** (split_balance
  0.443 → 0.515) — i.e. when the model uses the tool, it really
  splits the CoT.
- Token usage dropped 32%, recovering most of the efficiency the
  OPSD step had given up.
- This is exactly the RFT story: the filter removed the
  pathological long-trace rollouts that OPSD was learning from.

**Why this beats v2.1 baseline (0172) at 0.3722 vs 0.3444**:
- v2.1 baseline had near-zero split_balance (0.029); the
  "interleaving" was decorative batched calls.
- RFT 0187 has real split_balance 0.515 and slightly slower
  rollouts (8187 vs 6939 tokens), so the joint metric is up.

**Next experiments to try**:
- **0188 scale up**: 500 problems × 8 samples → ~600 positives,
  3 epochs. Expect another bump if the data scaling helps.
- **0189 RFT iterate** (proper expert iteration): sample new
  rollouts from RFT 0187 step_32, filter again, retrain. Each
  iteration should improve since the sampler is now better.
- **0190 RFT with OPSD warm-start**: rerun OPSD with state-saving
  (the 42cce7b commit), then RFT loads state and continues. This
  combines OPSD's strong placement priors with RFT's efficiency
  refinement.
