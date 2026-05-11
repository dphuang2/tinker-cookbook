"""Countdown numbers-game RL environment.

Given a target integer T and N source integers s_1..s_N, the model must emit
an arithmetic expression using each source at most once that evaluates to T.

Reward:
  1.0  -- valid expression that uses only allowed sources (each ≤ once) and
          evaluates to the target.
  0.0  -- otherwise. A small format bonus is applied via ProblemEnv.format_coef
          when the answer is wrapped in <answer>...</answer>.

The verifier is a strict AST walker over +, -, *, / and integer literals from
the source pool. Division is rational (we accept expressions whose value
equals the target exactly, e.g. (6/2) is fine; (5/2) is not because non-int).
"""
from __future__ import annotations

import ast
import operator
import re
from collections import Counter
from collections.abc import Sequence
from functools import partial

import chz
import numpy as np

from tinker_cookbook import renderers
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer


_BIN_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)


def _safe_eval(expr: str) -> tuple[float | None, list[int]]:
    """Evaluate `expr` allowing only +,-,*,/ over int literals.

    Returns (value_or_None_on_error, used_int_literals).
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None, []
    used: list[int] = []

    def walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            l = walk(node.left)
            r = walk(node.right)
            if isinstance(node.op, ast.Div) and r == 0:
                raise ValueError("div by zero")
            return _BIN_OPS[type(node.op)](l, r)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -walk(node.operand)
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            used.append(node.value)
            return float(node.value)
        raise ValueError(f"disallowed node {type(node).__name__}")

    try:
        val = walk(tree)
    except (ValueError, ZeroDivisionError):
        return None, []
    return val, used


class CountdownEnv(ProblemEnv):
    def __init__(
        self,
        target: int,
        sources: tuple[int, ...],
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        format_coef: float = 0.1,
        require_stop_sequence_for_format: bool = True,
    ):
        super().__init__(
            renderer,
            convo_prefix,
            format_coef=format_coef,
            require_stop_sequence_for_format=require_stop_sequence_for_format,
        )
        self.target = target
        self.sources = sources

    def get_question(self) -> str:
        return (
            f"Using each of the numbers {list(self.sources)} at most once and the "
            f"operators +, -, *, /, write an arithmetic expression that equals "
            f"{self.target}. Put your final expression inside <answer>...</answer>."
        )

    def _extract(self, sample_str: str) -> str | None:
        m = _ANSWER_RE.search(sample_str)
        if m:
            return m.group(1).strip()
        # Fallback: last non-empty line, common for base models that ignore tags.
        for line in reversed(sample_str.strip().splitlines()):
            line = line.strip()
            if line:
                return line
        return None

    def check_answer(self, sample_str: str) -> bool:
        expr = self._extract(sample_str)
        if expr is None:
            return False
        val, used = _safe_eval(expr)
        if val is None:
            return False
        if abs(val - self.target) > 1e-9:
            return False
        # No source used more than its multiplicity in the pool.
        pool = Counter(self.sources)
        for n in used:
            pool[n] -= 1
            if pool[n] < 0:
                return False
        return True

    def check_format(self, sample_str: str) -> bool:
        return _ANSWER_RE.search(sample_str) is not None

    def get_reference_answer(self) -> str:
        # ProblemEnv uses this only for logging; the verifier is what counts.
        return f"<answer>(expression equal to {self.target} using {list(self.sources)})</answer>"


def _sample_problem(rng: np.random.RandomState, n_sources: int = 4, max_source: int = 25, max_target: int = 100) -> tuple[int, tuple[int, ...]]:
    """Sample a problem guaranteed solvable by construction.

    Picks `n_sources` ints in [1, max_source], builds a random left-associative
    expression over +,-,*,/, and uses its value as the target if it is a
    positive integer in [1, max_target]. Retries up to 50 times.
    """
    for _ in range(50):
        srcs = tuple(int(rng.randint(1, max_source + 1)) for _ in range(n_sources))
        ops = [rng.choice(["+", "-", "*"]) for _ in range(n_sources - 1)]
        expr = str(srcs[0])
        for op, s in zip(ops, srcs[1:]):
            expr = f"({expr}{op}{s})"
        val, _ = _safe_eval(expr)
        if val is None:
            continue
        if val == int(val) and 1 <= int(val) <= max_target:
            return int(val), srcs
    # Fallback: trivial.
    return sum(srcs), srcs


class CountdownDataset(RLDataset):
    def __init__(
        self,
        batch_size: int,
        renderer: renderers.Renderer,
        group_size: int,
        n_batches: int = 200,
        n_sources: int = 4,
        seed: int = 0,
    ):
        self._rng = np.random.RandomState(None)
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.n_batches = n_batches
        self.n_sources = n_sources
        self.seed = seed

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        self._rng.seed(self.seed * 1_000_003 + index)
        return [self._make(self._rng) for _ in range(self.batch_size)]

    def _make(self, rng: np.random.RandomState) -> ProblemGroupBuilder:
        target, sources = _sample_problem(rng, n_sources=self.n_sources)
        return ProblemGroupBuilder(
            env_thunk=partial(CountdownEnv, target, sources, renderer=self.renderer),
            num_envs=self.group_size,
        )

    def __len__(self) -> int:
        return self.n_batches


@chz.chz
class CountdownDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    n_batches: int = 200
    group_size: int = 8
    n_sources: int = 4
    seed: int = 0

    async def __call__(self) -> tuple[CountdownDataset, None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        return CountdownDataset(
            batch_size=self.batch_size,
            renderer=renderers.get_renderer(self.renderer_name, tokenizer=tokenizer),
            n_batches=self.n_batches,
            group_size=self.group_size,
            n_sources=self.n_sources,
            seed=self.seed,
        ), None
