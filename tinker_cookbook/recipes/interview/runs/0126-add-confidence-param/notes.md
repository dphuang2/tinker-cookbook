# 0126 add-confidence-param

**Hypothesis**: add optional `confidence` (low/medium/high) param to PROGRESS_TOOL_SPEC. Tests whether asking the model to self-assess confidence at each checkpoint improves reasoning quality.

**Diff**: PROGRESS_TOOL_SPEC.parameters.properties += {"confidence": {"type": "string", "enum": ["low", "medium", "high"]}}. Not required.

**Result**: accuracy **0.882**, cadence: 35% 0-call, 51% exactly 3 calls. At no-tool baseline (single sample).

**Status**: `keep` (tentative; +0.7pp vs 0105 mean 0.8745; needs corroboration).

**Take-away**: forcing structured confidence self-assessment may genuinely lift accuracy — but variance ~1pp.

**Best**: 0126-recipe (tentative).

**Next**: variance rerun.
