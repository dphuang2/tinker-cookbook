# 0188 phaseB-rft-expert-iter — METRIC GOODHART EXPOSED

**Hypothesis**: expert iteration on top of 0187 (sample from 0187,
filter, retrain) further sharpens the model toward high-score
behavior since the sampler is already better than OPSD.

**Pipeline**:
1. Sample 800 rollouts from RFT 0187 step_32_final on 200 train
   problems (threshold 0.5 — higher quality bar)
2. 120 positives kept, score_max=0.993 (almost ideal rollouts!)
3. From-scratch LoRA SFT, 2 epochs, batch=8, 30 steps

**Sample stats vs OPSD-sourced 0187**:
| metric            | 0187 sample (OPSD source) | 0188 sample (RFT source) |
|-------------------|---:|---:|
| n_kept (threshold) | 125 (0.4) | 120 (0.5) |
| score_mean        | 0.418 | **0.466** |
| score_max         | 0.940 | **0.993** |
| frac_correct      | 0.876 | 0.865 |
| frac_interleaved  | 0.989 | **0.659** |

The RFT-sourced sample has higher score_mean AND has *fewer*
interleaved rollouts. This is the first signal: at the RFT stage,
the model already learned that LESS interleaving improves the
metric (because efficiency dominates).

**Eval result (vs prior bests)**:
| metric              | 0187 (1st RFT) | **0188 (expert iter)** |
|---------------------|---------------:|-----------------------:|
| accuracy            | 0.834          | **0.770**              |
| in_think_rate       | 0.004          | 0.002                  |
| **turn_split_rate** | 0.638          | **0.022 (collapsed)**  |
| mean_split_balance  | 0.515          | 0.302                  |
| mean_total_tokens   | 8187           | **3841**               |
| efficiency_factor   | 0.672          | **1.000 (capped)**     |
| **primary_score**   | **0.3722**     | **0.3876**             |

Cadence histogram: 465/500 skip tool, 23 use once, 8 twice, 4 thrice.
**The model abandoned the tool.**

**The Goodhart**: v2.1's `primary_score` formula is:
```
score = accuracy × (0.5 + 0.5 × interleaving × split_balance) × efficiency
```
The `0.5` floor means even at zero interleaving, the term is 0.5.
So accuracy × 0.5 × 1.0 = 0.385 beats accuracy × ~0.65 × 0.67 = 0.36
under the OLD efficient-but-interleaved configuration.

**Status**: **discard from a behavioral standpoint.** The model is
no longer interleaving — it's the v2.1 baseline pathology again,
just with smaller traces. This was a metric loophole RFT exploited.

**Proposed metric fix (request user input before committing)**:
Replace the additive 0.5 floor with multiplicative interleaving:
```
score = accuracy × interleaving_rate × split_balance × efficiency
```
or, to preserve the floor convention but make it less generous:
```
score = accuracy × (0.2 + 0.8 × interleaving × split_balance) × efficiency
```

Recomputing prior rows under pure-multiplicative:
- v2.1 base (0172):  0.862×0.280×0.029×0.793 = **0.0056**
- OPSD 0181:         0.842×0.976×0.443×0.455 = **0.166**
- RFT 0187 (1st):    0.834×0.638×0.515×0.672 = **0.184**
- RFT 0188 (iter):   0.770×0.022×0.302×1.000 = **0.005**

Under the stricter metric:
- 0187 leads decisively
- 0188 looks as bad as v2.1 base
- OPSD is competitive with RFT — Phase A wasn't a regression

This is a much truer ranking. **The current metric let RFT escape
into "skip the tool" Goodhart.**

**Status update on overall v3**: 0187 is still our best
behaviorally-correct result. We should:
1. Tighten the primary_score formula (drop the 0.5 floor)
2. Re-evaluate v3 progress under the new formula
3. Decide whether to continue from 0187 step_32_final or try a
   different RFT data composition (e.g. only keep rollouts with
   interleaving_rate × split_balance > 0.5)
