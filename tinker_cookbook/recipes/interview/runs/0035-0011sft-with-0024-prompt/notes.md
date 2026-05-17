# 0035 0011sft-with-0024-prompt

**Hypothesis**: 0011's SFT adapter might benefit from the same sparing-use directive that helped the base model in 0024. If the directive contributes ~+1.8pp on top of the base, maybe it also helps the SFT.

**Diff**: same code as 0024. Just pointed eval at 0011's SFT sampler (`tinker://dbbf713d.../sampler_weights/final`) instead of `base`.

**Result**: accuracy **0.744**, cadence `0:148, 1:242, 2:102, 3:8`. Vs 0011's original 0.740 (without directive) — marginal +0.4pp.

**Status**: `discard`. SFT + directive gets 0.744; base + directive gets 0.798. SFT is definitively hurting. The directive doesn't rescue the SFT damage.

| Setup | Accuracy |
|-------|----------|
| **0024 (base + directive)** | **0.798** |
| 0011 SFT + 0024 directive (this) | 0.744 |
| 0011 SFT (no directive) | 0.740 |

**Conclusion**: SFT is a strict downgrade for our recipe. Don't combine. The best path is purely prompt engineering on the base model.

**Best remains 0024 at 0.798.**
