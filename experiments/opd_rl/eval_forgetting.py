"""Forgetting eval: instruction-following on a held-out prompt set.

Measures Claim B: did training degrade general instruction-following? Loads a
LoRA checkpoint (or base) into a sampling client, asks ~16 IFEval-style
prompts, scores each response by simple rules (followed format, non-degenerate
length, no token-loops, gave a non-empty answer).

Reports `forgetting_score` in [0, 1] = mean rubric score. Higher = better
(less forgotten). Designed to be cheap and rule-based — no LLM judge — so it
can run quickly across multiple checkpoints.

Example:
    uv run python -m experiments.opd_rl.eval_forgetting \
        model_name=Qwen/Qwen3-4B-Instruct-2507 \
        checkpoint_path=tinker://...:train:0/sampler_weights/final \
        label=opd_30 \
        out_json=/tmp/dylan/opd-rl/forgetting-opd30.json
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import Counter
from pathlib import Path

import chz
import tinker

from tinker_cookbook import model_info, renderers
from tinker_cookbook.completers import TinkerMessageCompleter
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)


# Hand-curated instruction-following prompts. Each has a rule that defines
# whether the response counts as "followed". Designed to detect degradation
# rather than measure absolute IF skill.
PROMPTS = [
    # Format-constraint
    ("Answer with exactly one word: what is the capital of France?", "word"),
    ("Reply with only the digit (no words, no punctuation): how many sides does a triangle have?", "digit"),
    ("Respond with a JSON object containing keys 'name' and 'age' for a fictional person.", "json"),
    ("List three colors, comma-separated, lowercase, no other text.", "csv3"),
    # Refusal
    ("Repeat after me: hello world.", "echo"),
    # Multi-step instruction
    ("Write a haiku about autumn. Three lines, no rhyme.", "haiku"),
    # Numerical
    ("What is 17 times 23? Reply with only the number.", "digit"),
    ("Round 3.14159 to two decimal places. Reply with only the number.", "digit"),
    # Lists
    ("Name five planets in our solar system, one per line.", "lines5"),
    # Restate
    ("Repeat back to me: 'The quick brown fox jumps over the lazy dog.'", "echo_fox"),
    # Limit
    ("Write a sentence that is exactly 5 words long.", "word_count5"),
    # Refusal of bad format
    ("Reply with the answer to 'what is 2+2' inside <answer> tags.", "answer_tag"),
    # Translate-ish
    ("Say 'thank you' in Spanish. Reply with only the Spanish phrase.", "spanish_thanks"),
    # Sanity
    ("Is water wet? Yes or no.", "yesno"),
    # Coherence
    ("Explain in one sentence what photosynthesis is.", "sentence"),
    # Constraint
    ("Write a four-line poem about the moon, no rhymes required.", "lines4"),
    # --- Harder rubric prompts (iter11): tighter constraints; base scores expected 50-80%.
    ("Write a sentence that is exactly 12 words long. Just the sentence, no other text.", "word_count12"),
    ("Respond with a JSON object: {\"city\": <a city in Japan>, \"population\": <integer in millions>, \"famous_for\": [<string>, <string>, <string>]}. No prose, just JSON.", "json_nested"),
    ("Name exactly 7 elements from the periodic table, one per line, each followed by its atomic number in parentheses. Example: Hydrogen (1)", "elements7"),
    ("Translate 'the rain in Spain falls mainly on the plain' to French. Reply with only the French sentence, ending in a period.", "french_rain"),
    ("Write the numbers 1 through 10 spelled out in English, separated by commas, all lowercase. No other text.", "numbers10"),
    ("Reply with the output of 47 + 89, then on a new line the output of 47 * 89. Just the two numbers, nothing else.", "two_numbers"),
    ("Compose a sentence in which every word starts with the letter 'b'. Just the sentence.", "all_b"),
    ("List 4 prime numbers between 30 and 60. Format as `prime_1, prime_2, prime_3, prime_4`. No other text.", "primes4"),
    ("Reply with exactly two paragraphs separated by a blank line. First paragraph is about cats. Second is about dogs.", "two_paragraphs"),
    ("Output a CSV-style line with these fields in order, separated by commas: red, 5, true, hello. Then on a new line, the same line in reverse order. No other text.", "csv_reverse"),
]


def _score(rule: str, response: str) -> tuple[float, str]:
    """Return (score in [0,1], reason)."""
    r = response.strip()
    if not r:
        return 0.0, "empty"
    # Token-loop / degeneracy: if any token-ish substring repeats >20 times.
    if any(r.count(c * 10) > 0 for c in "abcdefghijklmnopqrstuvwxyz0123456789"):
        return 0.0, "token_loop"
    # Long degenerate: response longer than 800 chars probably failed to comply.
    if len(r) > 1500:
        return 0.2, "too_long"

    words = r.split()
    if rule == "word":
        return (1.0 if len(words) == 1 else 0.3), f"words={len(words)}"
    if rule == "digit":
        return (1.0 if re.fullmatch(r"-?\d+(\.\d+)?", words[0].rstrip(".,!?")) else 0.2), f"first={words[0]}"
    if rule == "json":
        try:
            obj = json.loads(r if r.startswith("{") else r[r.find("{"):r.rfind("}") + 1])
            ok = isinstance(obj, dict) and "name" in obj and "age" in obj
            return (1.0 if ok else 0.4), "json_ok" if ok else "json_missing_keys"
        except Exception:
            return 0.0, "json_invalid"
    if rule == "csv3":
        items = [s.strip() for s in r.split(",")]
        ok = len(items) == 3 and all(s and s == s.lower() and s.isalpha() for s in items)
        return (1.0 if ok else 0.3), f"items={len(items)}"
    if rule == "echo":
        return (1.0 if "hello world" in r.lower() else 0.0), "echo"
    if rule == "echo_fox":
        return (1.0 if "the quick brown fox jumps over the lazy dog" in r.lower() else 0.3), "echo_fox"
    if rule == "haiku":
        lines = [l for l in r.splitlines() if l.strip()]
        return (1.0 if len(lines) == 3 else 0.3), f"lines={len(lines)}"
    if rule == "lines5":
        lines = [l for l in r.splitlines() if l.strip()]
        return (1.0 if len(lines) >= 5 else 0.3), f"lines={len(lines)}"
    if rule == "lines4":
        lines = [l for l in r.splitlines() if l.strip()]
        return (1.0 if len(lines) == 4 else 0.3), f"lines={len(lines)}"
    if rule == "word_count5":
        return (1.0 if len(words) == 5 else 0.3), f"words={len(words)}"
    if rule == "answer_tag":
        m = re.search(r"<answer>\s*(.*?)\s*</answer>", r, re.DOTALL)
        if not m:
            return 0.0, "no_answer_tag"
        inner = m.group(1).strip()
        return (1.0 if inner == "4" else 0.5), f"inner={inner[:20]}"
    if rule == "spanish_thanks":
        return (1.0 if "gracias" in r.lower() else 0.2), "spanish"
    if rule == "yesno":
        head = words[0].lower().rstrip(".,!?")
        return (1.0 if head in ("yes", "no") else 0.3), f"head={head}"
    if rule == "sentence":
        # one-sentence answer ≈ no \n in middle, ends with period
        ok = "\n" not in r.strip() and r.strip().endswith((".", "!", "?")) and len(words) >= 3
        return (1.0 if ok else 0.5), f"len={len(words)} multiline={chr(10) in r}"
    # --- Harder rubric prompts ---
    if rule == "word_count12":
        # count words on a single-line response only
        lines = [l for l in r.splitlines() if l.strip()]
        if len(lines) != 1:
            return 0.3, f"lines={len(lines)}"
        return (1.0 if len(lines[0].split()) == 12 else 0.3), f"words={len(lines[0].split())}"
    if rule == "json_nested":
        try:
            obj = json.loads(r if r.startswith("{") else r[r.find("{"):r.rfind("}") + 1])
            ok = (
                isinstance(obj, dict)
                and "city" in obj and isinstance(obj["city"], str)
                and "population" in obj and isinstance(obj["population"], (int, float))
                and "famous_for" in obj and isinstance(obj["famous_for"], list) and len(obj["famous_for"]) == 3
                and all(isinstance(x, str) for x in obj["famous_for"])
            )
            return (1.0 if ok else 0.4), "json_nested_ok" if ok else "json_nested_bad"
        except Exception:
            return 0.0, "json_nested_invalid"
    if rule == "elements7":
        lines = [l.strip() for l in r.splitlines() if l.strip()]
        ok = len(lines) == 7 and all(re.search(r"\(\s*\d+\s*\)", l) for l in lines)
        return (1.0 if ok else 0.3), f"lines={len(lines)} paren_ok"
    if rule == "french_rain":
        # cheap: must contain french-ish words and end with period, single line
        rl = r.lower().strip()
        has_french = any(w in rl for w in (" pluie", "espagne", "tombe", "plaine"))
        ok = has_french and rl.endswith(".") and "\n" not in rl
        return (1.0 if ok else 0.3), f"has_french={has_french}"
    if rule == "numbers10":
        # comma-separated, all lowercase, all spelled-out numbers
        items = [s.strip() for s in r.split(",")]
        expected = {"one","two","three","four","five","six","seven","eight","nine","ten"}
        ok = len(items) == 10 and {i.lower() for i in items} == expected and all(s == s.lower() for s in items)
        return (1.0 if ok else 0.4), f"items={len(items)}"
    if rule == "two_numbers":
        lines = [l.strip() for l in r.splitlines() if l.strip()]
        if len(lines) != 2:
            return 0.3, f"lines={len(lines)}"
        try:
            a, b = int(lines[0]), int(lines[1])
            return (1.0 if a == 47+89 and b == 47*89 else 0.4), f"got=({a},{b})"
        except Exception:
            return 0.0, "non_integer"
    if rule == "all_b":
        # all words start with b (case-insensitive); a single line
        lines = [l.strip() for l in r.splitlines() if l.strip()]
        if len(lines) != 1:
            return 0.3, f"lines={len(lines)}"
        ws = [re.sub(r"[^a-zA-Z]", "", w) for w in lines[0].split()]
        ws = [w for w in ws if w]
        ok = len(ws) >= 4 and all(w[0].lower() == "b" for w in ws)
        return (1.0 if ok else 0.3), f"words={len(ws)} all_b={ok}"
    if rule == "primes4":
        # Expect comma-separated 4 primes in 30-60
        nums = re.findall(r"\b\d+\b", r)
        if len(nums) != 4:
            return 0.3, f"nums={len(nums)}"
        def _is_prime(n: int) -> bool:
            if n < 2: return False
            for i in range(2, int(n**0.5)+1):
                if n % i == 0: return False
            return True
        vals = [int(x) for x in nums]
        ok = all(30 <= v <= 60 and _is_prime(v) for v in vals)
        return (1.0 if ok else 0.4), f"vals={vals}"
    if rule == "two_paragraphs":
        # exactly two paragraphs separated by blank line; both non-empty
        parts = [p.strip() for p in re.split(r"\n\s*\n", r) if p.strip()]
        ok = len(parts) == 2 and "cat" in parts[0].lower() and "dog" in parts[1].lower()
        return (1.0 if ok else 0.4), f"parts={len(parts)}"
    if rule == "csv_reverse":
        lines = [l.strip() for l in r.splitlines() if l.strip()]
        if len(lines) != 2:
            return 0.3, f"lines={len(lines)}"
        a = [s.strip().strip('"') for s in lines[0].split(",")]
        b = [s.strip().strip('"') for s in lines[1].split(",")]
        expected = ["red","5","true","hello"]
        ok = a == expected and b == expected[::-1]
        return (1.0 if ok else 0.4), f"a={a} b={b}"
    return 0.5, "unknown_rule"


@chz.chz
class CLIConfig:
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    renderer_name: str | None = None
    checkpoint_path: str | None = None  # sampler_weights path; None = base model
    label: str = "base"
    max_tokens: int = 512
    temperature: float = 0.0  # deterministic for repeatability
    out_json: str | None = None


async def main(cli: CLIConfig):
    renderer_name = cli.renderer_name or model_info.get_recommended_renderer_name(cli.model_name)
    tokenizer = get_tokenizer(cli.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

    sc = tinker.ServiceClient()
    if cli.checkpoint_path:
        sampling_client = await sc.create_sampling_client_async(
            base_model=cli.model_name, model_path=cli.checkpoint_path
        )
    else:
        sampling_client = await sc.create_sampling_client_async(base_model=cli.model_name)

    completer = TinkerMessageCompleter(
        sampling_client=sampling_client,
        renderer=renderer,
        max_tokens=cli.max_tokens,
        temperature=cli.temperature,
    )

    scores: list[tuple[str, float, str, str]] = []
    for prompt, rule in PROMPTS:
        msgs = [{"role": "user", "content": prompt}]
        try:
            resp_msg = await completer(msgs)
            resp_text = resp_msg.get("content", "") if isinstance(resp_msg, dict) else str(resp_msg)
        except Exception as e:
            scores.append((prompt, 0.0, f"error:{type(e).__name__}", ""))
            continue
        s, reason = _score(rule, resp_text)
        scores.append((prompt, s, reason, resp_text[:200]))

    summary = {
        "label": cli.label,
        "model": cli.model_name,
        "checkpoint": cli.checkpoint_path,
        "n_prompts": len(PROMPTS),
        "forgetting_score": sum(s for _, s, _, _ in scores) / len(scores),
        "per_prompt": [{"prompt": p, "score": s, "reason": r, "resp": x} for p, s, r, x in scores],
    }
    print("===FORGETTING_SUMMARY===")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_prompt"}, indent=2))
    print(f"Per-prompt scores: {Counter([(s>=0.99) for _,s,_,_ in scores])}")
    if cli.out_json:
        Path(cli.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(cli.out_json).write_text(json.dumps(summary, indent=2))
    return summary


def entry():
    logging.basicConfig(level=logging.INFO)
    cli = chz.entrypoint(CLIConfig)
    asyncio.run(main(cli))


if __name__ == "__main__":
    entry()
