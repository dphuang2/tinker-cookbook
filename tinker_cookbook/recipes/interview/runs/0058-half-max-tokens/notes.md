# 0058 half-max-tokens

MAX_TOKENS_PER_TURN 8192 → 4096.

Acc 0.596 — catastrophic. Most problems truncated mid-reasoning.

Discard. Reverted MAX_TOKENS_PER_TURN to 8192.

Lesson: 8192 is at or near the lower bound for the thinking traces this dataset requires.

Best remains 0024.
