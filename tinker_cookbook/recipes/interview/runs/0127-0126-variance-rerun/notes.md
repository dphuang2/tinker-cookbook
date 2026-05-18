# 0127 0126-variance-rerun

**Result**: accuracy **0.874**, cadence: 33% 0-call, 52% exactly 3 calls.

**Status**: `variance`. 0126 2-sample mean **0.878** = 0105 mean (within noise). Confidence param doesn't compound accuracy gain.

**Take-away**: structured self-assessment (confidence enum) doesn't lift accuracy beyond the recipe ceiling. Adds complexity for no gain.

**Action**: revert confidence param to keep canonical simpler PROGRESS_TOOL_SPEC.
