# Cortex RL-sandbox design: validation against current research (Oct 2026)

Verification legend: **[V]** means I checked the arXiv/proceedings page this session. **[V-s]** means I confirmed it only through search-result metadata or summaries. **[K]** means I am citing from prior knowledge and did not re-check it this session. **[U]** means unverified or speculative.

---

## Q1. Is "RL expert in sim, then distil into ACT via targeted demos" sound?

**Answer: yes.** It is a well-established pattern: train a privileged teacher, generate data with it, then distil into the image student. It is the cheapest and most robust option for an SO-101 pick-and-place on CPU. It is not the frontier of *policy fine-tuning*, though. Since 2024 the frontier has been RL applied to the BC policy itself: residual RL, latent steering, or direct policy-gradient fine-tuning. Cortex should keep distillation as the default and offer residual RL as an advanced mode.

Evidence:
- **Teacher-to-student distillation with a scripted or privileged expert is standard.** ACT itself was trained in sim on demos from *scripted* policies with privileged state: Zhao et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware", RSS 2023, arXiv 2304.13705 [K]. ResiP v1 also combined residual RL with teacher-student distillation and visual domain randomization to produce RGB policies (see below) [V-s].
- **Residual RL on frozen BC policies:**
  - *ResiP*: Ankile, Simeonov, Shenfeld, Torne, Agrawal, "From Imitation to Refinement – Residual RL for Precise Assembly", arXiv 2407.16677 (2024). It puts an on-policy (PPO) residual on a frozen chunked BC policy. Key finding: BC on precise tasks saturates with more data, and the residual adds closed-loop correction. https://arxiv.org/abs/2407.16677 [V-s]
  - *ResFiT*: Ankile, Jiang, Duan, Shi, Abbeel, Nagabandi, "Residual Off-Policy RL for Finetuning Behavior Cloning Policies", arXiv 2509.19301, ICLR 2026. Off-policy, sparse binary reward, treats the BC base as a black box (a_t = a_base + a_res). On a 29-DoF humanoid it raised success from about 14–23% to about 64% in tens of minutes of real data. https://arxiv.org/abs/2509.19301 [V-s]
  - *Policy Decorator*: Yuan, Mu, Tao, Fang, Zhang, Su, ICLR 2025, arXiv 2412.13630. A model-agnostic residual over a frozen Behavior Transformer or Diffusion Policy, with bounded exploration, tested on ManiSkill. https://arxiv.org/abs/2412.13630 [V-s]
- **Steering and policy-agnostic methods:**
  - *DSRL*: Wagenmaker et al. (Levine lab), "Steering Your Diffusion Policy with Latent Space Reinforcement Learning", CoRL 2025, arXiv 2506.15799. It runs RL over the *noise* input of a diffusion policy, needs only black-box access, and leaves the weights unchanged. https://arxiv.org/abs/2506.15799 [V-s]. It does **not** apply directly to ACT. ACT's CVAE latent z is set to its prior mean at inference, so an analogous "steer z" variant is conceivable but unvalidated [U].
  - *PA-RL*: Sobol Mark, Gao, Sampaio, Srirama, Sharma, Finn, Kumar, "Policy Agnostic RL", ICLR 2025, arXiv 2412.06685. It samples candidate actions from the base policy, improves them with a critic, then *distils with a supervised loss*. Structurally this is the closest published analogue to Cortex's "improved rollouts → BC retrain". https://arxiv.org/abs/2412.06685 [V-s]
- **Direct policy-gradient fine-tuning:** *DPPO*: Ren et al., "Diffusion Policy Policy Optimization", ICLR 2025, arXiv 2409.00588 [V-s]. *RL-100*: Lei, Li, … Xu, arXiv 2510.14830 (2025–26). It does IL, then OPE-gated offline RL, then online PPO-style RL on diffusion policies, and explicitly supports action chunking [V-s]. *PAC-ACT*: Pang & Li, "Post-training Actor-Critic for Action Chunking Transformers", arXiv 2607.09590 (Jul 2026). This is the first paper I found that does chunk-level actor-critic RL directly on a pretrained ACT, using a behaviour-prior constraint. It is a two-author preprint, not yet peer-reviewed. https://arxiv.org/abs/2607.09590 [V]. *Q-chunking*: Li, Zhou, Levine, "Reinforcement Learning with Action Chunking", NeurIPS 2025, arXiv 2507.07969. It shows TD-RL over action chunks with n-step backups, which makes it the natural critic design for chunked policies [V-s].
- **Real-world human-in-the-loop RL:** *HIL-SERL*: Luo, Xu, Wu, Levine, arXiv 2410.21845 (2024) [V-s]. *ConRFT*: Chen et al., RSS 2025, arXiv 2502.05450. It fine-tunes only the VLA action head with a consistency policy [V-s]. **LeRobot** ships HIL-SERL with a distributed SAC actor/learner, a reward classifier and `gym_hil` MuJoCo environments (Franka). The commands have changed across versions, so check them against the repo. https://huggingface.co/docs/lerobot/hilserl_sim [V-s]
- **VLA RL** (not relevant at Cortex's scale): VLA-RL (Lu et al., arXiv 2505.18719) [V-s]; SimpleVLA-RL (ICLR 2026, arXiv 2509.09674) [V-s]; RIPT-VLA (Tan et al., arXiv 2505.17016) [V-s]. All need multi-GPU rollouts of 7B models.

**What fits ACT and small compute.** On CPU, nothing that back-propagates through an image ACT is practical. The best fit for ACT is **residual RL** (ResiP/ResFiT/Policy Decorator are architecture-agnostic, and ResiP was built on chunked BC). PAC-ACT is ACT-specific but immature. DSRL and DPPO assume diffusion policies.

**Recommendation.** Keep CEM skill-tuning plus distillation as the default. Add residual RL on ACT as a GPU-cluster "advanced" sandbox method (detail in Q2). Note in the docs that PA-RL is the closest published precedent for "improve with RL, distil by supervised loss".

---

## Q2. Residual RL on ACT vs skill-parameter RL vs full state-based PPO

**Sample efficiency.** Searching a low-dimensional parameter space over a structured primitive is the most sample-efficient of the three by a wide margin. RAPS showed that switching the action interface to parameterised primitives improves learning efficiency regardless of the RL algorithm used: Dalal, Pathak, Salakhutdinov, NeurIPS 2021, arXiv 2110.15360 [V-s]. MAPLE adds primitives alongside atomic actions: Nasiriany, Liu, Zhu, arXiv 2110.03655 (ICRA 2022 per my recollection [K]; venue not re-verified). With 5–9 parameters, CEM needs on the order of 10–20 iterations × 16–64 rollouts. That is a few hundred to a few thousand episodes, which is fine on CPU MuJoCo. CEM is also an accepted optimiser in robot learning: QT-Opt uses CEM to maximise its Q-function over actions (Kalashnikov et al., CoRL 2018, arXiv 1806.10293) [V-s for the paper; the CEM detail is from the paper's method, K]. Strictly, QT-Opt's CEM is an inner-loop argmax, not episodic policy search. The better analogy for Cortex is episodic black-box policy search over primitive parameters, in the spirit of RAPS.

Full state-based PPO from scratch is the least efficient option (typically 10^6–10^8 steps). On CPU it is only realistic with MuJoCo Playground/MJX-style batching on a GPU: Zakka et al., "MuJoCo Playground", arXiv 2502.08844 [V-s]. Residual RL sits between the two. It starts at the BC success rate, so its sample needs depend on the size of the gap. Policy Decorator and ResFiT report sample-efficient off-policy learning, but with GPU training.

**Transfer back to the image policy.** This is the weak point of skill-RL. Improvements reach ACT only through BC on new demos, so the uplift is bounded by (a) how well ACT imitates the tuned skill from pixels, and (b) whether the open-loop, near-perfect scripted demos contain any *recovery* behaviour. Classic remedies:
- DART noise injection (Laskey et al., CoRL 2017, arXiv 1703.09327 [K]): perturb the expert during collection so the demos cover recovery states.
- DAgger-style relabelling (Ross, Gordon, Bagnell, AISTATS 2011 [K]): roll out ACT, then have the privileged skill label corrective actions from ACT-visited states.

Residual RL changes the *deployed* policy (ACT + residual), so it needs no lossy distillation step. But the residual must then run at deployment. If the residual uses privileged state, it is sim-only and still needs distillation. Asymmetric actor-critic, with a privileged critic and an image or proprio actor, is the standard compromise: Pinto et al., arXiv 1710.06542 [K].

**Recommendation for a 5-DoF SO-101 pick-and-place on CPU.**
1. Default: CEM over skill parameters with privileged state, plus DART noise and DAgger relabels when generating targeted demos.
2. Advanced (Slurm GPU): residual SAC over frozen ACT, a chunk-level residual on the first k actions of each chunk, sparse reward plus the shaped term, and a privileged critic.
3. Do not offer full PPO from scratch except as a research toggle.

---

## Q3. Weakness discovery and automatic environment/curriculum generation

**Domain randomisation.**
- Uniform DR can give high-variance, suboptimal policies. Active DR learns which parameter regions are informative: Mehta, Diaz, Golemo, Pal, Paull, CoRL 2019, arXiv 1904.04762 [V-s].
- OpenAI's Automatic DR widens ranges once performance crosses a threshold: OpenAI et al., "Solving Rubik's Cube with a Robot Hand", arXiv 1910.07113 [K].
- DrEureka uses an LLM to generate rewards and DR ranges for sim-to-real: Ma et al., RSS 2024, arXiv 2406.01967 [V-s].

Cortex's "DR centred on the weak condition" is a reasonable discrete form of Active DR. ADR-style *range expansion* is better for robustness, because it prevents over-fitting the new demos to a single point.

**LLM-generated environments and rewards.**
- Eureka (Ma et al., ICLR 2024, arXiv 2310.12931) [V-s] evolves reward code.
- Eurekaverse (Liang et al., CoRL 2024, arXiv 2411.01775) [V-s] evolves an environment curriculum.
- GenSim (Wang et al., ICLR 2024, arXiv 2310.01361) [V-s] and GenSim2 (Hua et al., CoRL 2024, arXiv 2410.03645) [V-s] generate tasks and demos.
- RoboGen (Wang et al., ICML 2024, arXiv 2311.01455) [V-s] runs a propose-generate-learn loop.
- **GenEx** (Lu, Shu et al., arXiv 2412.09624) [V-s] generates explorable 3D *video worlds* from one image for embodied planning. It is **not relevant** to MuJoCo manipulation sandboxes and should be dropped from the design rationale.

For Cortex, the calling agent already plays the "LLM proposer" role. The research supports letting it propose sandbox *parameters* and *reward terms* (Eureka-style), but these must be validated by measured pilots, because LLM proposals hallucinate (the GenSim2 authors note this).

**Failure-driven targeted data.** This is the most relevant line of work for Cortex, and it validates `improve_policy`.
- *Mulligan*: Ankile, Dong, …, Song, Finn, arXiv 2610.05882 (Oct 2026) [V]. Failures cluster in a small set of initial states. Steering collection toward observed failures and untried states beats uniform sampling at equal budget, giving +10–34 pp on real tasks.
- *AdaDemo*: Mu, Guo, Xu, Goyal, Su, Fox, Garg, arXiv 2404.07428 (ICLR 2024 workshop) [V-s]. Collects demos targeted at the policy's weaknesses.
- *DISEIL*: Khanal, Kumar, Rana, arXiv 2609.08123 (Sep 2026) [V]. Clusters failures into modes and uses a VLM/LLM to write the next demo request, checked against constraints. Its design is very close to Cortex's VLM-hypothesis plus targeted-demo loop. Sim-only, with scripted experts.
- *Sirius*: Liu, Nasiriany, Zhang, Bao, Zhu, RSS 2023, arXiv 2211.08416 [V-s]. Uses intervention-weighted BC.

**Demo curation.** Score the new demos before they enter the dataset:
- DemInf: Hejna et al., "Robot Data Curation with Mutual Information Estimators", arXiv 2502.08623 [V-s].
- CUPID: Agia et al., CoRL 2025, arXiv 2506.19121 [V-s]. Influence of each demo on closed-loop return, needing about 25–100 rollouts.
- DataMIL: Dass et al., ICLR 2026, arXiv 2505.09603 [V-s].

**MimicGen-style synthesis.**
- MimicGen: Mandlekar et al., CoRL 2023, arXiv 2310.17596 [V-s].
- SkillMimicGen: Garrett, Mandlekar, Wen, Fox, CoRL 2024, arXiv 2410.18907 [V-s].
- DexMimicGen: Jiang et al., arXiv 2410.24185 [V-s].

These generate thousands of demos from a few by re-targeting object-centric segments to new poses.

**Recommendation: use it alongside RL, not instead of it.** Cortex's scripted oracle already *is* a demo generator for new poses, so MimicGen adds most value for **agent-teleop or human demos**, which carry human-like strategies the IK script lacks. Pipeline: tuned skill (CEM) for conditions the script can solve, and MimicGen re-targeting of agent/human demos for conditions it cannot. Both pass a success filter and a curation score (CUPID-lite) before versioning.

---

## Q4. Predicting improvement before running

**Answer: the field has no validated predictor of uplift from a single round.** The closest work:
- *Data Scaling Laws in IL*: Lin, Hu, Sheng, Wen, You, Gao, ICLR 2025, arXiv 2410.18647 [V-s]. Generalisation follows a power law in the number of environments and objects, but **not** in demo count with those held fixed. Demo count shows fast early gains, then a plateau. The authors themselves caution against fitting 3-parameter curves to about 6 points.
- *Factored Scaling Curves*: Zha, Badithela, …, Majumdar, arXiv 2505.07728 (2025) [V]. Fits per-factor performance-vs-data curves (camera, table height, distractors…) and allocates a fixed budget across factors, giving up to +26% real success. **This is the principled version of what Cortex is attempting**, with Cortex's "conditions" as the factors.

**Is the heuristic defensible?** Only as a labelled prior.
- `sandbox_after = pilot + (1−pilot)·k` is not needed as a prediction. A short CEM run *measures* the tuned-skill success directly. Report a Beta posterior on it instead of a guess.
- `t ∈ [0.3,0.6]` (distillation transfer) has no published support for these specific numbers [U]. It also ignores two things: interference with other conditions (a new targeted data mix can lower success elsewhere), and saturation (Lin et al.).

**More principled replacement:**
1. **Sandbox ceiling, measured.** Run CEM on a small budget. The elite set's success gives a Beta(1+s, 1+f) posterior on `sandbox_after`. Present the 90% credible interval.
2. **Transfer, learned per project.** For each past round r, store (condition, demos added n_r, success before p_r, success after p'_r, tuned-skill success q_r). Fit a per-condition saturating curve, e.g. logit p(n) = a + b·log(1+n) capped by q, or a monotone fit like the factored scaling curves. Use hierarchical pooling across conditions with weakly-informative priors; until there are ≥3 rounds, use the current heuristic range as the prior. Get prediction intervals by bootstrapping over episodes and rounds.
3. **Report a cross-condition regression term.** Predict the change on the *other* conditions as well (starting prior ≈ 0 ± a few pp, updated from history).
4. **Cost**: measured seconds per episode × episodes (CEM iterations × population + demo generation + held-out eval) + training time from past `lerobot-train` runs on the same hardware tier. Report wall-clock and GPU-hours separately.
5. **Rank sandboxes by expected uplift per hour.** Optionally rank by probability of improvement ≥ δ.

---

## Q5. Evaluation rigour

- **STEP**: Snyder, Hancock, Badithela, Dixon, Miller, Ambrus, Majumdar, Itkina, Nishimura, "Is Your Imitation Learning Policy Better than Mine? Policy Comparison with Near-Optimal Stopping", RSS 2025, arXiv 2503.10966 [V]. A sequential test with valid early stopping, up to 32% fewer trials than baselines. Follow-up: "Beyond Binary Success", arXiv 2603.13616 (RSS 2026 per proceedings listing) [V-s]. Partial-credit metrics, anytime-valid.
- **Kress-Gazit et al.**, "Robot Learning as an Empirical Science: Best Practices for Policy Evaluation", arXiv 2409.09491 (2024) [V-s]. Report conditions, multiple metrics, statistics and failure modes. Cortex already does most of this.
- **SureSim**: Badithela et al., arXiv 2510.04354 (2025) [V]. Prediction-powered inference combining many sim and few real evals. Relevant later for sim-to-real.

**Practical rules.**
- Keep Wilson CIs for single rates; Clopper-Pearson is conservative.
- For **base vs tuned and old vs new ACT, use matched seeds and McNemar** (exact binomial on the discordant pairs). Matched seeds remove initial-state variance and usually cut the trials needed substantially when outcomes are correlated.
- Sample sizes for an unpaired two-proportion test (α=0.05 two-sided, power 0.8), computed with the standard normal-approximation formula: 0.50→0.80 needs about 39 per arm; 0.60→0.75 about 152; 0.80→0.90 about 199. Wilson half-width at n=50, p=0.5 is about ±0.13.
- Implication: per-condition claims of <10 pp uplift are not detectable at typical Cortex budgets. Use STEP-style sequential stopping, with a cap of about 200 episodes per condition.

---

## Q6. Sim-to-real (brief)

- *Squint*: Almuzairee & Christensen, RA-L 2026, arXiv 2602.21203 [V]. Visual SAC on an **SO-101** 8-task ManiSkill3 suite with heavy DR. It trains in about 15 minutes on one RTX 3090 and transfers zero-shot to a real SO-101. This is the strongest direct precedent.
- ManiSkill3 ships an SO-100 grasp-cube digital twin with a DR config [V-s].
- MuJoCo Playground covers MuJoCo DR of robot parameters, mass and friction [V-s].

**Recommendation.** Record DR ranges (camera pose/FOV, lighting, textures, friction, mass, actuator gains/latency) in each sandbox's manifest so they can be reused for real transfer later. Plan a SureSim-style real-plus-sim evaluation once hardware exists.

---

## Q7. UI for live RL runs (brief)

- **Common patterns:**
  - Isaac Lab / Isaac Gym: a tiled grid of parallel envs rendered in one viewport.
  - TensorBoard / W&B: learning curves (return, success, episode length) against env steps and wall-clock, with media panels of rollout videos at checkpoints.
  - Rerun: LeRobot's dataset visualiser uses it [K] for time-synced camera and joint streams.
- **For Cortex**, each sandbox card shows:
  - a live thumbnail grid of N concurrent episodes, green/red bordered by outcome;
  - CEM progress (elite mean ± std per parameter, success of best/elite vs iteration);
  - a Wilson band on current success vs the pilot;
  - the failure-mode stacked bar;
  - budget burned vs estimate.
- The summary view puts predicted vs realised uplift side by side per round. That is what calibrates the Q4 model.

---

## Recommended design changes

1. **Keep CEM skill-parameter RL as the default sandbox method.** It is cheap, CPU-friendly and interpretable (RAPS/MAPLE precedent; PA-RL is the closest analogue of "improve, then distil"). Use CEM with a population of 32, 20% elites and ≤15 iterations. Add a small smoothing/noise floor so the distribution does not collapse early.
2. **Make the generated demos recovery-rich.** Add DART-style noise injection during tuned-skill demo generation (e.g. 10–30% of demos with perturbed waypoints, kept only if successful). Add an optional **DAgger-relabel pass**: roll out current ACT in the weak condition and label those states with the tuned skill's action. Only possible if the skill can re-plan from arbitrary states; mark this as a skill requirement.
3. **Add "residual RL on ACT" as an advanced, GPU-only method**:
   - frozen ACT;
   - a small Gaussian residual on the executed chunk (ResiP/ResFiT/Policy Decorator recipe), with an SAC-style off-policy learner;
   - a privileged-state critic (asymmetric);
   - sparse reward plus the failure-mode shaping term;
   - a residual magnitude clamp (e.g. ≤2 cm, ≤10° per step).

   Output either (a) a deployable ACT+residual checkpoint, evaluated like any other, or (b) distillation demos. List PAC-ACT (2607.09590) as experimental only. Do not offer DSRL/DPPO, which are diffusion-specific.
4. **Add MimicGen-style re-targeting** (`synthesize_demos`) for conditions where the scripted skill's pilot is below a floor (e.g. <30%). It re-targets successful agent/human teleop demos to the weak condition's object poses. Every synthesised demo must pass a sim success check.
5. **Upgrade weakness discovery.**
   - Keep gap × (1 − CI width), but compute it on the **lower Wilson bound of the gap** so noisy conditions are not over- or under-ranked.
   - Cluster failures by initial-state region (Mulligan) and by failure mode (DISEIL).
   - Attach the VLM hypothesis to each cluster as an *unverified* label.
6. **Sandbox DR.** Centre on the weak condition, then use **ADR-style widening**: expand a range when tuned-skill success exceeds 80% inside it, until the condition's neighbourhood is covered. Store ranges in the sandbox manifest.
7. **Replace the uplift heuristic with measured plus learned estimates** (see Q4):
   - `sandbox_after` is a Beta posterior from a short CEM pilot (the base-skill pilot stays as the baseline).
   - `policy_after` comes from a per-project, per-condition saturating transfer curve fitted on past rounds, with a bootstrap interval. The current k/t ranges are the explicit prior until ≥3 rounds exist, labelled "heuristic prior" in the MCP response.
   - Always also return the predicted change on other conditions and P(uplift ≥ δ).
8. **MCP `propose_sandboxes` response fields:**
   - `weakness_id`, `method` (cem_skill | residual_rl | mimicgen);
   - `pilot {n, successes, wilson_95}`;
   - `predicted_sandbox_success {mean, ci90, source: measured|prior}`;
   - `predicted_policy_uplift {mean, ci90, source}`;
   - `predicted_regression_elsewhere`;
   - `cost {episodes, cpu_hours, gpu_hours, wall_clock_est}`;
   - `uplift_per_hour`, `p_improve_ge_delta`;
   - `assumptions[]`.

   `accept_sandboxes(ids, budget_cap)` spawns the selected sandboxes.
9. **Stopping rules (per sandbox):**
   - CEM stops when the elite success lower Wilson bound ≥ target, OR there is <2 pp improvement over 3 iterations, OR the budget is exhausted.
   - Demo generation stops at N targeted demos (default 20–50 per condition, capped so targeted data is ≤30–40% of the dataset to limit interference), or when the curation score drops.
   - The round stops when the re-evaluated ACT has failed to improve the weak condition over 2 consecutive rounds (paired test), which flags it for human/teleop demos instead.
10. **Evaluation protocol:**
    - Base vs tuned skill, and old vs new ACT, on **matched held-out seeds** with McNemar.
    - STEP-style sequential stopping, capped at about 200 episodes per condition.
    - Evaluate **all** conditions, not just the targeted one, so regressions are caught.
    - Only the re-evaluated ACT counts as "measured improvement". Keep this rule.
11. **Close the loop explicitly:**
    1. evaluate;
    2. weaknesses;
    3. propose_sandboxes (with predictions);
    4. agent accepts;
    5. sandboxes run (live UI);
    6. targeted demos plus curation filter;
    7. new dataset version (diff and provenance: which sandbox, params, seeds);
    8. **human-approved** retrain;
    9. paired re-evaluation;
    10. log predicted vs realised uplift, which refits the transfer model.
12. **UI:** a per-sandbox card with an episode thumbnail grid, a CEM parameter/elite chart, a success-vs-iteration line with a Wilson band, a failure-mode stacked bar and a budget meter. The round view shows predicted vs realised uplift.

## Unverified / caveats
- Exact k and t values have no literature basis.
- PAC-ACT, Mulligan and DISEIL are recent preprints without peer review (as of Oct 2026).
- MAPLE's venue and QT-Opt's CEM detail are from memory.
- OpenAI ADR, DART, DAgger, the ACT paper and asymmetric actor-critic were cited from memory and not re-fetched this session.
- The VLA-RL, RIPT-VLA and Beyond-Binary-Success details come from search summaries only.
- LeRobot RL command syntax changes between versions.
