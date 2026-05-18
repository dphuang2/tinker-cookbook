# 0152 0105-variance-rerun-20 — 20-SAMPLE MILESTONE

**Result**: accuracy **0.882**. **20-sample mean 0.8755** ± 0.010 std.

**Status**: `variance`.

**95% CI for true mean**: [0.871, 0.880]. The upper bound TOUCHES the no-tool baseline 0.880, meaning the 0105 prompt-only recipe is **statistically indistinguishable from no-tool baseline** at n=20.

**Final stats (20 samples, 0105 recipe, no SFT)**:
- Mean: 0.8755
- Std: 0.010
- 95% CI: [0.871, 0.880]
- No-tool baseline: 0.880
- Δ vs baseline: -0.45pp (within 1σ)
- Tool use rate: ~50% (healthy bimodal)
- Training records: 0

The prompt-only recipe has effectively closed the structural gap that having the tool spec in-prompt creates.
