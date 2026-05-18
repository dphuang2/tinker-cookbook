# 0063 near-max-tokens

MAX_TOKENS_PER_TURN = 28672 (close to Qwen3's 32k context window cap).

Acc 0.866, cadence 395:68:23:6:3:3:0:0:2.

Curve:
| max_tokens | acc |
|------------|-----|
| 4096 | 0.596 |
| 8192 | ~0.80 |
| 12288 | ~0.84 |
| 16384 | 0.860 |
| 24576 | **0.870** |
| 28672 | 0.866 |

Looks saturated around 24576-28672. The gains from raising the cap stop after ~24576. The actual ceiling for prompt-only with this eval is around 0.87.

Best remains 0062 (max_tokens=24576, 0.870) by point estimate.
