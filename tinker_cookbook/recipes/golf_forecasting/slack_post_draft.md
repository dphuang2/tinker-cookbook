# Slack Post Draft for #when-in-doubt

## Context for reviewers

I want to post a thread on X about an autoresearch experiment I ran using Claude Code on the web. Wanted to get feedback on whether this is interesting enough and appropriate to share publicly. The vibe I'm going for is similar to Karpathy's autonomous research posts where he shows the full trajectory of experiments (including failures). The kicker is backtesting the model against the 2026 Masters (which just finished today) and showing where it disagreed with Kalshi prediction markets. Looking for thumbs up / concerns.

---

## Proposed X post (thread format)

**Post 1 (hook):**

I let Claude Code run autonomously for 49 hours to build a golf forecasting model from scratch. 100 experiments, zero human intervention. It discovered data, trained models, and iterated on hypotheses while the Masters was being played.

Then I backtested it against the 2026 Masters and compared it to Kalshi odds. [attach: experiment_progress.png]

**Post 2 (setup):**

The task: given a mid-tournament golf leaderboard, predict who wins. The agent had full freedom over models, training, prompts, data, and eval design. One constraint: maintain a frozen benchmark so progress is trackable.

It started with a heuristic baseline (log-loss 2.81) and a broken zero-shot LLM (7.36).

**Post 3 (the journey):**

108 experiments. Only 48% were kept -- the rest regressed or didn't help. That's the real story of research.

The agent tried 9+ model families (1B to 235B params), RL vs SFT, prompt engineering, data augmentation, multi-teacher distillation, ensembles, chain-of-thought, and round-dependent routing.

[attach: experiment chart showing all dots including failures, Karpathy-style]

**Post 4 (key discoveries):**

The most interesting findings, all discovered autonomously:

1. Teacher quality >> student capacity. A 1B model distilled from DeepSeek-V3.1 matches 8B and 70B. The bottleneck is labels, not parameters.

2. RL consistently destroyed SFT calibration. Tried 4x across configs. Proper scoring rewards weren't enough.

3. Chain-of-thought made predictions WORSE (1.72 vs 1.23). Overthinking hurts calibration.

4. Best system: Kimi-K2.5 (thinking model, 8k budget) for early rounds + DeepSeek for final round. The agent invented a multi-model router.

**Post 5 (results):**

Final: log-loss 0.73 -- 74% better than the heuristic baseline.

The 1B distilled student achieves 1.10, matching the DeepSeek teacher. A frontier model's golf forecasting compressed to run on a phone.

**Post 6 (the Masters backtest -- this is the good part):**

The 2026 Masters just finished. Rory McIlroy won his second straight green jacket. I backtested the model on the actual round-by-round leaderboards and compared to Kalshi:

After R1 (McIlroy co-leading at -5):
- Model: 12% McIlroy
- Kalshi: 25% McIlroy
- Model more conservative -- historically, R1 leaders win ~15-20%. Arguably better calibrated.

After R2 (McIlroy with historic 6-shot lead at -12):
- Model: 33% McIlroy
- Sportsbooks: 74% McIlroy (-280)
- Model was WAY more skeptical of the massive lead than the public market.

What happened next? McIlroy shot 73 in R3 and nearly blew the entire tournament. The model's skepticism was validated.

After R3 (McIlroy tied with Cameron Young at -11):
- Model: 32% McIlroy, 28% Young, 22% field
- Kalshi: 36% McIlroy, 29% Young
- Remarkably close to prediction market pricing. Model correctly picked McIlroy.

**Post 7 (the edge):**

The model's biggest disagreement with the market was after R2: 33% vs 74%. A 41 percentage point gap.

McIlroy did win, so this specific "trade" loses. But the CALIBRATION question matters: should a 6-shot lead after 36 holes really be priced at 74%? The model says no. And the near-collapse in R3 suggests the model understood something the market didn't.

Over many tournaments, better calibration = consistent edge.

**Post 8 (what this means):**

This isn't about golf. It's about what happens when you give an AI agent a well-scoped research problem and let it run.

It formed hypotheses, tested them, learned from failures, and made genuinely surprising discoveries -- all documented in 100 git commits over 2 days. And the model it built has opinions that disagree with prediction markets in interesting ways.

Built with @ThinkingMachinesLab's Tinker + Claude Code on the web.

---

## Attachments to include

1. `experiment_progress.png` -- the hero chart showing all 108 experiments with the best-so-far frontier line (Karpathy-style)
2. Backtest comparison table: Model vs Kalshi vs Sportsbook at each round
3. Optionally: `change_type_breakdown.png`, git log screenshot

## Risk assessment for reviewers

**Positive:**
- Showcases Claude Code + Tinker capabilities in an engaging, technical way
- Golf + the Masters is timely and non-controversial
- Karpathy-style progress chart is proven to be engaging
- The Kalshi comparison adds a concrete "so what" -- it's not just an academic exercise
- Shows failures honestly (48% reverted, model not always right)
- The "model was more skeptical than the market" angle is genuinely interesting

**Potential concerns:**
- Could be read as "AI can beat prediction markets" -- mitigated by explicitly noting McIlroy DID win, so the specific trade lost. The claim is about calibration, not outcome.
- Could be seen as encouraging sports betting -- mitigated by framing it as a research/calibration question, not a betting strategy.
- "Autonomous AI" could feel overhyped -- mitigated by showing the 48% failure rate and noting it followed a structured program.md.
- Uses competitor model names (DeepSeek, Kimi, Qwen) -- these are just the models the system evaluated, showing breadth.

**Tone:** Technical but accessible. Not claiming the model beats Vegas. Claiming the research process and calibration findings are interesting.
