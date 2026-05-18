# 0084 0080-plus-verify

**Hypothesis**: 0069 added a verify sentence to 0062 (null). Test under 0080 recipe — might compound.

**Diff**: appended "Verify your answer briefly before writing it in the boxed format." to USER_INSTRUCTION_SUFFIX.

**Result**: accuracy **0.868**, cadence `0:294, 1:10, 2:22, 3:121, 4:34, 5:7, 6:5, 7:1, 8:1, 9:4, 10:1`. 58.8% 0-call.

**Status**: `discard` (within noise, no improvement; slight cadence weakening to 59% 0-call from 0080's 51%).

The verify directive doesn't compound with 0080's cadence anchor. Same null result as 0069.

**Action**: revert verify sentence.

**Best remains 0080-recipe at 3-sample mean 0.876.**
