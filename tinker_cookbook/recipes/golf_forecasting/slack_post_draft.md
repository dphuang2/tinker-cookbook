# Slack Post Draft for #when-in-doubt

## Slack message for #when-in-doubt

:bufo-gives-an-idea: I want to post a thread on X about a fun autoresearch experiment I ran over the weekend. I pointed a cloud-hosted coding agent (Claude Code on the web) at Tinker and told it to build a golf forecasting system from scratch. It ran 108 experiments over 49 hours with no human in the loop, driving the benchmark from 2.81 to 0.54.

I backtested the best system on the Masters that just finished. The model was more confident than Kalshi in McIlroy's 6-shot lead (85% vs 65%) and was right -- McIlroy won. The vibe I'm going for is similar to Karpathy's autonomous research posts (full experiment trajectory including failures).

Wanted to get a read on whether this is good to share publicly. Proposed X posts in the thread.

Full session link (internal): https://claude.ai/code/session_01VzCzqLRWq3ttDiWQRqpPtV
Code branch: https://github.com/dphuang2/tinker-cookbook/tree/claude/golf-forecasting-setup-VIpRZ/tinker_cookbook/recipes/golf_forecasting

---

## Proposed X post (thread format)

**Post 1:**

I pointed a cloud-hosted coding agent at an open-ended research task -- build a golf forecasting system -- and let it run for 49 hours on Tinker. No human in the loop.

It ran 108 experiments. Here's the full trajectory, including the ones that made things worse. [attach: experiment_progress.png]

**Post 2:**

The setup: given a mid-tournament leaderboard, predict who wins. The agent chose models, training methods, prompts, data sources, and eval design. The only fixed constraint was a frozen benchmark so I could measure real progress.

Starting point: a heuristic baseline at log-loss 2.81. The best system got it down to 0.54.

**Post 3:**

52% of experiments were reverted. Some findings along the way:

- A 1B student distilled from DeepSeek-V3.1 matched 8B and 70B models. The teacher's labels were the bottleneck, not the student's capacity.
- RL degraded calibration every time it was tried (4 attempts, different configs).
- Chain-of-thought made predictions worse. Calibration suffered when the model overthought.

**Post 4:**

I backtested the best system on the Masters that just finished, using the fine-tuned 1B model the agent trained.

After R2, McIlroy held a historic 6-shot lead. Kalshi had him at ~65%. The model gave him 85%.

McIlroy won. The model was more confident than the market -- and was right. [attach: trading_timeline.png]

**Post 5:**

After R3, McIlroy's lead had collapsed. He shot 73 and was tied with Cameron Young. Kalshi had him at 36%. The model: 35%.

The model and market agreed when the situation was ambiguous. The edge was R2 -- correctly reading a dominant lead that the market underpriced.

McIlroy held on to win by one.

**Post 6:**

This worked because of Tinker. The agent trained, distilled, and evaluated dozens of models across 108 experiments -- switching between model families, RL and SFT, different prediction formats -- all without managing GPUs. Tinker handled the training infrastructure. Claude Code on the web ran the agent loop.

Code and results are open:

https://github.com/dphuang2/tinker-cookbook/tree/claude/golf-forecasting-setup-VIpRZ/tinker_cookbook/recipes/golf_forecasting

108 experiments, 100 git commits, every hypothesis logged.

---

## Attachments

1. `experiment_progress.png` -- Karpathy-style 108-experiment progress chart (hero image for Post 1)
2. `masters_2026_trading_timeline.png` -- model vs Kalshi at each round (Post 4)

## Risk assessment for reviewers

**Strong positives:**
- Timely: Masters just finished today
- Engaging: prediction markets + golf + AI research is a fun combo
- Honest: shows 52% failure rate, one tournament backtest framed as one data point
- Technical depth: the discoveries (1B matching 8B, RL hurting calibration) are genuinely interesting
- The Kalshi comparison gives a concrete "so what" that non-ML people understand
- Code is fully open, every experiment is in git

**Potential concerns:**

1. **Competitive intelligence / signaling what we're building** -- Some customers use Tinker for forecasting workloads. A polished forecasting demo could signal what the platform is good at. Mitigations:
   - Golf is a toy domain, and the recipe is already open-source in tinker-cookbook
   - The post leads with "autonomous agent" not "forecasting platform"
   - **Would appreciate team input on this one**

2. **"AI can beat prediction markets" reading** -- The model was right on one tournament. The post frames it as "the model was more confident and happened to be right" not "we found an edge." One data point, explicitly stated.

3. **Encouraging sports betting** -- Framed around calibration quality, not trading strategy. No P&L calculations in the post.

4. **Competitor model names (DeepSeek, Kimi)** -- These are just models the agent evaluated during its search. Shows breadth.

5. **"Autonomous AI" overhype** -- Mitigated by the 52% failure rate and linking the full code so people can see it followed a structured program.

**Tone:** Technical, understated. The claim is: "an autonomous agent drove real benchmark improvement, and the resulting model had an interesting opinion about the Masters." Not: "we cracked prediction markets."
