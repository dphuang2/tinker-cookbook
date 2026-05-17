# 0025 system-user-reinforce

**Hypothesis**: 0024's user-message directive worked best. Adding a brief reinforcing system prompt ("checkpoint is optional; use it only when it genuinely aids your reasoning") might compound the effect.

**Diff**: sft_train.py
- Kept 0024's user-message directive.
- Added a short SYSTEM_PROMPT reinforcing the same message.

**Result**: accuracy **0.792**, cadence `0:447, 1:33, 2:13, 3:5, 5:1, 6:1`. 89% 0-call.

Worse than 0024 (0.798) by 0.6pp. Cadence further compressed (89% vs 82%). Same pattern as 0023 — more aggressive suppression → less tool use → slight accuracy drop.

| Recipe | accuracy | 0-call % |
|--------|----------|----------|
| **0024 (user-msg only)** | **0.798** | 82% |
| 0025 (user + system) | 0.792 | 89% |

**Status**: `discard`. Reinforcement hurts; one position is enough.

**Take-away**: too much suppression dampens tool use beyond the optimal level. The model's natural decision-making (with one directive) is better than two.

**Best remains 0024 at 0.798.**

**Next ideas**:
1. **Tighten 0024's directive phrasing** — fewer words, see if information density matters.
2. **Try opposite framing** — "Use the checkpoint tool aggressively to track your reasoning; it does not hurt accuracy" — counterfactual test.
3. **Add a positive example** in the description — show one good example of when to call. May ground the model.

Picking #3 (positive example in description): cheap, structurally distinct, tests whether concrete examples help.
