# Slack Post Draft for #when-in-doubt

## Slack message for #when-in-doubt

:bufo-gives-an-idea: I want to post a thread on X about a fun autoresearch experiment for training a golf forecasting model I ran over the weekend. Wanted to get feedback on whether this is interesting enough and appropriate to share publicly. The vibe I'm going for is similar to Karpathy's autonomous research posts where he shows the full trajectory of experiments (including failures).

Result: backtesting the model against the 2026 Masters that just finished and showing it identified contrarian edges against Kalshi in both directions.

Full session link (internal): https://claude.ai/code/session_01VzCzqLRWq3ttDiWQRqpPtV
Code branch: https://github.com/dphuang2/tinker-cookbook/tree/claude/golf-forecasting-setup-VIpRZ/tinker_cookbook/recipes/golf_forecasting
Proposed X posts in the thread here

---

## Proposed X post (thread format)

**Post 1:**

I pointed a cloud-hosted coding agent at an open-ended research task -- build a golf forecasting system -- and let it run for 49 hours on Tinker. No human in the loop.

It ran 108 experiments. Here's the full trajectory, including the ones that made things worse. [attach: experiment_progress.png]

**Post 2:**

The setup: given a mid-tournament leaderboard, produce calibrated win probabilities. The agent chose models, training methods, prompts, data sources, and eval design. The only fixed constraint was a frozen benchmark so I could measure real progress.

Starting point: a heuristic baseline at log-loss 2.81. The best system got it down to 0.54.

**Post 3:**

52% of experiments were reverted. Some findings along the way:

- A 1B student distilled from DeepSeek-V3.1 matched 8B and 70B models. The teacher's labels were the bottleneck, not the student's capacity.
- RL degraded calibration every time it was tried (4 attempts, different configs).
- Chain-of-thought made predictions worse. Calibration suffered when the model overthought.

**Post 4:**

I backtested the system on this weekend's Masters.

After R2, McIlroy held a historic 6-shot lead. Kalshi priced him at ~65%. The model gave him 34%.

McIlroy then shot 73 in R3 and nearly lost the tournament. [attach: trading_timeline.png]

**Post 5:**

After R3, McIlroy was tied with Cameron Young. Kalshi dropped him to 36%. The model gave him 55%.

McIlroy held on to win by one.

On Kalshi, the R2 call alone was actionable. McIlroy NO was ~35c after R2, ~64c after R3. One tournament, not a system -- but the model identified the mispricing.

**Post 6:**

I also ran raw Claude Opus 4.6 on the same prompts (no fine-tuning, no knowledge of the outcome):

After R2: Claude said 42%. Closer to the model's 34% than Kalshi's 65%.
After R3: Claude said 32%.

Base Opus is already more skeptical than prediction markets. Fine-tuning sharpened that further -- especially at R2, where the trained model was more disciplined about the 6-shot lead (34% vs Claude's 42%).

**Post 7:**

This worked because of Tinker. The agent needed to train, distill, and evaluate dozens of models across 108 experiments -- switching between model families, RL and SFT, different prediction formats -- without managing GPUs or distributed training. Tinker handled all of that. Claude Code on the web provided the agent loop.

Code and results are open:

https://github.com/dphuang2/tinker-cookbook/tree/claude/golf-forecasting-setup-VIpRZ/tinker_cookbook/recipes/golf_forecasting

108 experiments, 100 git commits, every hypothesis logged.

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

1. **Competitive intelligence / signaling what we're building** -- Some of our customers use Tinker for forecasting workloads. Posting a polished forecasting demo could signal to competitors what our platform is good at and what kinds of use cases we're investing in. Even though golf is a toy domain, the underlying pipeline (calibrated probability distributions, proper scoring rules, SFT distillation for forecasting, multi-model routing) maps directly to real customer workflows. Possible mitigations:
   - Reframe the post to lead with "autonomous AI research agent" rather than "forecasting." The hero story is Claude Code running 100 experiments unattended, not Tinker's forecasting stack.
   - Downplay the Tinker-specific details (reward functions, training infra) and emphasize the agent loop + discoveries.
   - Or: decide we're comfortable with this signal because the recipe is already open-source in tinker-cookbook anyway.
   - **This is the concern I'm most uncertain about -- would appreciate team input.**

2. "AI can beat prediction markets" reading -- **mitigated** by explicitly calling it one data point and framing it as calibration, not a trading strategy

3. Encouraging sports betting -- **mitigated** by framing around calibration quality, not "here's how to make money"

4. Competitor model names (DeepSeek, Kimi, Qwen) -- these are just the models the agent evaluated, showing breadth of search

5. "Autonomous AI" could feel overhyped -- **mitigated** by 48% failure rate and noting it followed a structured program.md

**Tone:** Technical but accessible. The claim is narrow: "the model's calibration was interesting in ways that disagreed with the market." Not: "we cracked sports betting."

## Alternative framing (if concern #1 is serious)

If the team thinks the forecasting angle is too revealing, the post could be reframed to focus purely on the autonomous research loop:

- Lead: "I let Claude Code run 100 experiments in 49 hours with zero intervention"
- Hero visual: the Karpathy-style experiment progress chart
- Findings: 1B matching 8B, RL hurting calibration, agent inventing multi-model routing
- Skip or minimize: the Kalshi backtest, the forecasting-specific details
- Mention Tinker only as "our training platform" without detailing the forecasting pipeline

This version is less spicy (no prediction market angle) but also less revealing about what the platform does well. It's more of a "look at what coding agents can do" post than a "look at what our forecasting stack can do" post.

The tradeoff: the Kalshi angle is what makes this post stand out from generic "I let an AI agent run" posts. Without it, we're competing with every other autonomous agent demo. With it, we have a concrete, testable, timely hook that nobody else has.
