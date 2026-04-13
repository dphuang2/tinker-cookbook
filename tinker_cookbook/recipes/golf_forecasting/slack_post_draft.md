# Slack Post Draft for #when-in-doubt

## Slack message for #when-in-doubt

:bufo-gives-an-idea: I want to post a thread on X about a fun autoresearch experiment I ran over the weekend. I pointed a cloud-hosted coding agent (Claude Code on the web) at Tinker and told it to build a golf forecasting system from scratch. It ran 108 experiments over 49 hours with no human in the loop, driving a frozen benchmark from 2.81 to 0.54 log-loss across 19 held-out tournaments.

For fun, I also ran the model on the Masters that just finished to see what it thought. The vibe I'm going for is similar to Karpathy's autonomous research posts (full experiment trajectory including failures).

Wanted to get a read on whether this is good to share publicly. Proposed X posts in the thread.

Full session link (internal): https://claude.ai/code/session_01VzCzqLRWq3ttDiWQRqpPtV
Code branch: https://github.com/dphuang2/tinker-cookbook/tree/claude/golf-forecasting-setup-VIpRZ/tinker_cookbook/recipes/golf_forecasting

---

## Proposed X post (thread format)

**Post 1:**

I pointed a cloud-hosted coding agent at a research task (build a golf forecasting system) and let it run for 49 hours on Tinker. No human in the loop.

It ran 108 experiments. Here's the full trajectory, including the ones that made things worse. [attach: experiment_progress.png]

**Post 2:**

The setup: given a mid-tournament leaderboard, predict who wins. The agent chose models, training methods, prompts, data sources, and eval design. Only fixed constraint was a frozen benchmark so I could measure real progress.

52% of experiments were reverted. The rest drove the benchmark from 2.81 to 0.54 log-loss, measured across 19 held-out tournaments.

**Post 3:**

Some findings along the way:

- A 1B student distilled from DeepSeek-V3.1 matched 8B and 70B models. Teacher labels were the bottleneck, not student capacity.
- RL degraded calibration every time it was tried (4 attempts, different configs).
- Chain-of-thought made predictions worse. Overthinking hurt calibration.

The winning system: DeepSeek-V3.1 as teacher with a prompt the agent iterated into (binary prediction with explicit lead margin context), distilled into a 1B model via SFT. The 1B matches the teacher and runs on a phone.

**Post 4:**

For fun I ran the best system on the Masters that just finished.

After R2, McIlroy held a historic 6-shot lead. Kalshi had him at ~65%. The model said 85%. After R3, his lead had collapsed. Tied with Cameron Young. Kalshi: 36%. Model: 35%.

McIlroy held on to win by one. [attach: trading_timeline.png]

**Post 5:**

One tournament doesn't prove anything about calibration. The frozen benchmark across 19 tournaments does. But it's fun to see what the model thinks about a live event and whether it disagrees with the market.

**Post 6:**

This worked because of Tinker. The agent trained, distilled, and evaluated dozens of models across 108 experiments, switching between model families, RL and SFT, different prediction formats, all without managing GPUs. Tinker handled the training infra. Claude Code on the web ran the agent loop.

Code and results:

https://github.com/dphuang2/tinker-cookbook/tree/claude/golf-forecasting-setup-VIpRZ/tinker_cookbook/recipes/golf_forecasting

108 experiments, 100 git commits, every hypothesis logged.

---

## Attachments

1. `experiment_progress.png` - Karpathy-style 108-experiment progress chart (hero image for Post 1)
2. `masters_2026_trading_timeline.png` - model vs Kalshi at each round (Post 4)

## Risk assessment for reviewers

**Strong positives:**
- Timely: Masters just finished today
- Engaging: prediction markets + golf + AI research is a fun combo
- Honest: shows 52% failure rate, benchmark is the punchline not one tournament, explicitly says "one tournament doesn't prove anything"
- Technical depth: the discoveries (1B matching 8B, RL hurting calibration) are genuinely interesting
- Code is fully open, every experiment is in git

**Potential concerns:**

1. **Competitive intelligence / signaling what we're building** - Some customers use Tinker for forecasting workloads. A polished forecasting demo could signal what the platform is good at. Mitigations:
   - Golf is a toy domain, and the recipe is already open-source in tinker-cookbook
   - The post leads with "autonomous agent" not "forecasting platform"
   - **Would appreciate team input on this one**

2. **"AI can beat prediction markets" reading** - Heavily mitigated. The post explicitly says "one tournament doesn't prove anything about calibration" and frames the Masters as "for fun" color, not evidence.

3. **Competitor model names (DeepSeek)** - Just a model the agent evaluated. Shows breadth.

4. **"Autonomous AI" overhype** - Mitigated by the 52% failure rate and linking the full code.

**Tone:** Technical, understated. The punchline is the benchmark improvement driven by autonomous research, not the Masters prediction.
