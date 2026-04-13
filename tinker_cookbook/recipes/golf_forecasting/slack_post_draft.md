# Slack Post Draft for #when-in-doubt

## Context for reviewers

I want to post a thread on X about an autoresearch experiment I ran using Claude Code on the web. Wanted to get feedback on whether this is interesting enough and appropriate to share publicly. The vibe I'm going for is similar to Karpathy's autonomous research posts where he shows the full trajectory of experiments (including failures). The kicker: backtesting the model against the 2026 Masters that just finished and showing it identified contrarian edges against Kalshi in both directions. Looking for thumbs up / concerns.

---

## Proposed X post (thread format)

**Post 1 (hook):**

I let Claude Code run autonomously for 49 hours to build a golf forecasting model from scratch. 100 experiments, zero human intervention.

Then I backtested it against this weekend's Masters and compared it to Kalshi. The model correctly faded the crowd in both directions. [attach: trading_timeline.png]

**Post 2 (setup):**

The task: given a mid-tournament golf leaderboard, predict who wins. The agent had full freedom -- models, training, prompts, data, eval design. One constraint: maintain a frozen benchmark.

It started with a heuristic baseline (log-loss 2.81) and a broken zero-shot LLM (7.36). 108 experiments later, it built a system at 0.73 -- a 74% improvement.

[attach: experiment_progress.png]

**Post 3 (the journey):**

Only 48% of experiments were kept. The rest regressed or didn't help. That's the real story.

The agent tried 9+ model families (1B to 235B), RL vs SFT, prompt engineering, multi-teacher distillation, chain-of-thought, and round-dependent routing. It invented a multi-model router on its own.

**Post 4 (key discoveries):**

All discovered autonomously:

1. Teacher quality >> student capacity. A 1B model matches 8B and 70B when distilled from the same teacher. Labels matter, not parameters.

2. RL consistently destroyed SFT calibration. 4 attempts, all worse.

3. Chain-of-thought made predictions WORSE. Overthinking hurts calibration.

4. Best system: Kimi-K2.5 (thinking model) for early rounds + DeepSeek for the final round.

**Post 5 (the Masters backtest -- this is the good part):**

McIlroy just won his second straight Masters. I ran the model on the actual 2026 leaderboard after each round and compared to Kalshi/sportsbook odds:

After R2 (McIlroy with historic 6-shot lead):
- Sportsbooks: 74% McIlroy
- Kalshi: ~65%
- Our model: 34%

The model was *dramatically* more skeptical of the massive lead. What happened next? McIlroy shot 73 in R3 and nearly blew the entire tournament.

**Post 6 (the flip):**

After R3 (McIlroy tied with Cameron Young, lead gone):
- Kalshi: 36% McIlroy (market panicked)
- Our model: 55% McIlroy

The model FLIPPED. When everyone panicked about the collapse, the model said: he's still the best player on the board.

McIlroy held on to win by one shot.

**Post 7 (the trade):**

If you followed the model on Kalshi:

After R2: SELL McIlroy YES at 74c (model says overpriced)
After R3: BUY BACK McIlroy YES at 36c (model flips, says underpriced)

Round-trip: 38c profit per contract. No outcome risk.

Then hold the YES through Sunday. McIlroy wins. Pays $1.

Total: $1.02 profit on 74c of capital. 138% return in 2 days.

**Post 8 (the real takeaway):**

This isn't a "beat the market" claim from one tournament. It's one data point.

But what's interesting is HOW the model disagreed. It correctly identified that the crowd:
- Overpriced certainty after the 6-shot lead
- Overreacted to the subsequent stumble

That's the pattern well-calibrated models exploit: they fade emotional extremes.

**Post 9 (what this means for AI research):**

49 hours. 100 experiments. Zero human intervention.

The agent formed hypotheses, tested them, learned from failures, and made genuinely surprising discoveries. Then the model it built had opinions that disagreed with prediction markets in interesting, testable ways.

Built with @ThinkingMachinesLab's Tinker + Claude Code on the web.

---

## Attachments

1. `masters_2026_trading_timeline.png` -- the hero chart (model vs market lines + P&L waterfall)
2. `experiment_progress.png` -- Karpathy-style 108-experiment progress chart
3. Optional: git log screenshot, `change_type_breakdown.png`

## Risk assessment for reviewers

**Strong positives:**
- Timely: Masters literally just finished today
- Engaging: prediction markets + golf + AI research is a fun combo
- Honest: shows 48% failure rate, explicitly says "one data point, not a beat-the-market claim"
- Technical depth: the discoveries (1B matching 8B, RL hurting calibration) are genuinely interesting
- The Kalshi comparison has a concrete "so what" that non-ML people can understand

**Potential concerns:**
- "AI can beat prediction markets" reading -- **mitigated** by explicitly calling it one data point and framing it as calibration, not a trading strategy
- Encouraging sports betting -- **mitigated** by framing around calibration quality, not "here's how to make money"
- Competitor model names (DeepSeek, Kimi, Qwen) -- these are just the models the agent evaluated, showing breadth of search
- "Autonomous AI" could feel overhyped -- **mitigated** by 48% failure rate and noting it followed a structured program.md

**Tone:** Technical but accessible. The claim is narrow: "the model's calibration was interesting in ways that disagreed with the market." Not: "we cracked sports betting."
