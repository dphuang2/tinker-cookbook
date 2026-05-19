# 0196 phaseC-rl-rloo — discard (RL drifted policy from 0193 warmstart)

**Hypothesis**: RL with RLOO baseline starting from 0193's sampler
would push primary_score above 0.2025 by directly optimizing the
v2.2 score per rollout (negative gradient on bad ones, positive on
good ones, with leave-one-out variance reduction).

**Pipeline** (`rl_train.py`):
1. Warmstart sampling client from 0193 step_68_final
2. 6 RL steps × 16 problems × group_size 4 = 64 rollouts per step
3. Score each rollout v2.2; RLOO advantage = score_i − mean(group−i)
4. importance_sampling loss + Adam at lr=1e-5
5. Save sampler every 2 steps

**On-policy mean_score trajectory**:
| step | mean_score | n_datums |
|-----:|----------:|---------:|
| 1 | 0.327 | 60 |
| 2 | 0.356 | 63 |
| 3 | 0.267 | 57 |
| 4 | 0.210 | 58 |
| 5 | 0.347 | 58 |
| 6 | 0.229 | 53 |

Volatile, no improvement trend. Best was step 2 (0.356), so eval
that.

**Held-out eval (step_2 sampler)**:
| metric              | 0193 (warmstart) | **0196 step_2** | Δ |
|---------------------|----------------:|----------------:|---:|
| accuracy            | 0.854           | 0.820           | −3.4 pp |
| turn_split_rate     | 0.938           | 0.980           | +4.2 pp |
| mean_split_balance  | 0.478           | 0.424           | −0.054 |
| mean_total_tokens   | 10411           | 12610           | +21% |
| efficiency          | 0.528           | 0.436           | −0.092 |
| **primary_score**   | **0.2025**      | **0.1488**      | **−0.054** |

**Status**: `discard`. Two RL gradient updates moved the policy
*away* from the SFT optimum: tokens grew 21%, split_balance fell,
accuracy fell. Held-out primary_score dropped by 0.054.

**Why RL didn't help**:
1. **No KL anchor to reference policy.** Tinker's
   `importance_sampling` loss is pure policy-gradient with IS
   correction; no explicit penalty against drifting away from a
   reference. The policy is free to wander, and noisy reward
   gradients pushed it wrong.
2. **Small batch (60 datums per step).** RLOO variance scales as
   1/group_size = 1/4; with only 16 problems × 4 = 64 trajectories
   per step, gradient estimates are very noisy.
3. **On-policy score is biased upward vs held-out.** Mean_score
   on training prompts was 0.327-0.356 during steps 1-2, but
   held-out came in at 0.149. The sampling pool of 2000 train
   problems lets the model find pockets of easy ones; held-out
   doesn't.

**Lessons**:
- Pure RL (without KL anchor) at this batch size is *not* stable
  enough to push past the SFT plateau. Either (a) much bigger
  batches (200+ datums/step) to reduce variance, OR (b) a KL
  penalty against the reference (warmstart) policy.
- The cookbook's `incorporate_kl_penalty` (used in OPSD) computes
  per-token KL via a separate sampling client at the reference;
  rl_train could be modified to add that as a regularizer term in
  the advantage. Estimated lift over current 0196: meaningful, but
  ~2x more compute per step.

**Locked-in v3 best stays at 0193: primary_score 0.2025.**
