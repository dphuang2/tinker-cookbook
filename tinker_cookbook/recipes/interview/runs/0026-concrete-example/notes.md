# 0026 concrete-example

**Hypothesis**: ground the tool description in a concrete worked example (showing what kind of moment merits a checkpoint). The model can imitate the pattern.

**Diff**: PROGRESS_TOOL_SPEC description expanded to include a multi-sentence "Example: after spending a paragraph testing whether the integral factors and finding it does not, you might call checkpoint(...)" worked example.

**Result**: accuracy **0.780**, cadence `0:416, 1:48, 2:19, 3:9, 5:4, 6:2, 7:1, 8:1`. Worse than 0024 (0.798) by 1.8pp.

**Status**: `discard`. Concrete example hurt. Hypotheses why:
1. Verbose description eats prompt budget / attention.
2. The contour-integral example primes the model toward a specific framing that isn't appropriate for most DeepMath problems.
3. Examples make the tool feel more "specialized" — model uses it less freely and on the "wrong" problems.

**Best remains 0024 at 0.798.**

**Pattern summary** (prompt-only experiments 0020-0026):

| Run | Modification | Accuracy |
|-----|--------------|----------|
| 0020 | base + 0011 description | 0.774 |
| 0021 | minimal description | 0.764 |
| 0022 | + sparing system prompt | 0.792 |
| 0023 | + stronger optional system prompt | 0.788 |
| **0024** | **directive in user message instead** | **0.798** |
| 0025 | user + system reinforce | 0.792 |
| 0026 | concrete example in description | 0.780 |

The optimum sits at 0024: medium-length description + user-message sparing directive + no system prompt. Variations in any direction regress.

**Next ideas**:
1. **Try opposite framing** — "Use the tool aggressively; the user wants to see your progress" — counterfactual. If this beats 0024, we'd learn the model defaults to under-use.
2. **Shorter user-message directive** — distill 0024's instruction into fewer words.
3. **Acknowledge the loop is exhausted** — declare 0024 the final recipe, stop exploring. Document and exit.

Picking #2 (shorter directive): keeps exploring the local-maximum hypothesis. If shorter wording matches or beats, we have a more efficient recipe.
