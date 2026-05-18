# 0118 tool-ack-continue

**Diff**: tool ack content "ok" → "continue".

**Result**: accuracy **0.878**, cadence `0:272, 1:3, 2:1, 3:189, 4:12, 5:5, 6:9, 7:4, 8:1, 9:2, 12:1, 15:1`. 54.4% 0-call.

**Status**: `variance`. Parity with 0105/0107 mean (0.878). Tool ack content is fungible.

**Take-away**: ack content variations (ok / noted / continue / echoed summary) are all equivalent. The agent loop's signal for continuation is the existence of the tool response, not its content.

**Action**: revert to "ok" for canonical/minimal recipe.

**Best remains 0105-recipe at 5-sample mean 0.8744 (with "ok" ack).**
