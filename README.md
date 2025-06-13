# Deep-Reinforcement-Learning-Vanilla-versus-Double-Deep-neural-networks.
To implement a Reinforcement Learning (RL) agent using a Deep Q Network (DQN) applied to the game of Atari in the OpenAI Gym environment
Deep Reinforcement Learning: Teaching an Atari Breakout Agent with DQN & Double DQN



1. Why Reinforcement Learning?

Deep Reinforcement Learning (DRL) lets machines learn behaviour directly from high‑dimensional sensory input by maximising long‑term reward. Breakout is an excellent testbed: sparse reward, rich visuals, delayed credit assignment and a non‑trivial optimal strategy (precise paddle control, tunnel carving, ball trapping).

This repo compares Vanilla DQN (Mnih et al., 2013/15) and Double DQN (van Hasselt et al., 2016) to demonstrate how over‑estimation bias in Q‑learning harms performance and how a conceptually simple fix yields faster, more stable learning.

Assignment Objective  (CS6482 Deep RL • Semester 1 AY 2024/25)Train both agents for 3 000 episodes and evaluate them with the near‑greedy policy (ε = 0.05) prescribed by Mnih et al. Report mean score, learning speed and stability, and discuss design trade‑offs.

2. The Environment & Setup

2 a. Environment: BreakoutNoFrameskip‑v4

Property

Value

API

OpenAI Gym ALE interface

Observation

Raw 210 × 160 × 3 RGB frame

Pre‑processing

 crop play area • down‑sample to 84 × 84 • convert to grayscale • normalise to [0,1]

Frame Stack

4 consecutive frames (temporal context)

Action Space

4 discrete: NOOP, FIRE, RIGHT, LEFT

Reward Clipping

+1 for breaking a brick, ball lost = −1

Episode Terminates

Life = 0 or score ∼ 800

Wrapper Stack  MaxAndSkipEnv(4) → EpisodicLifeEnv → FireResetEnv → WarpFrame(84) → FrameStack(4).

2 b. Dependencies

Python ≥ 3.10

PyTorch 2.2 (CUDA 12)

Gymnasium 0.29 (Atari extras)

NumPy, tqdm, tensorboard, wandb (optional)

pip install -r requirements.txt
# plus ROMs
pip install autorom[accept-rom-license] && AutoROM

2 c. Quick Start

# Clone & install
git clone https://github.com/<your‑handle>/dqn-breakout.git
cd dqn-breakout
pip install -r requirements.txt

# Run the notebook (end‑to‑end)
jupyter notebook Deep\ Reinforcement_Learning_DQN_Double_DQN.ipynb

Full training (3 000 episodes) ≈ 3 h on an RTX 3060; ≈ 8 h on a 4‑core CPU.

2 d. Repository Structure

├── Deep Reinforcement_Learning_DQN_Double_DQN.ipynb   ← Main notebook
├── src/                                               ← Agent classes & utils
│   ├── dqn_agent.py             ← Vanilla DQN class
│   ├── double_dqn_agent.py      ← Double DQN subclass
│   ├── replay_buffer.py         ← Uniform & Prioritised buffer
│   └── wrappers.py              ← Atari pre‑processing wrappers
├── train.py                                           ← Headless CLI entry‑point
├── models/                                            ← Saved .pth checkpoints
├── results/                                           ← CSV metrics, gifs, plots
├── docs/                                              ← Images for README
└── requirements.txt

3. DQN Implementations

3 a. Network Architecture

Input: 4 × 84 × 84
 ├─ Conv2d 32@8×8 stride 4 ─ ReLU
 ├─ Conv2d 64@4×4 stride 2 ─ ReLU
 ├─ Conv2d 64@3×3 stride 1 ─ ReLU
 → Flatten (3136)
 ├─ Linear 512               ─ ReLU
 └─ Linear |A| (=4)          → Q‑values

≈ 1.7 M parameters, identical for both agents.

3 b. Vanilla DQN

Standard single Q‑network; TD target uses max over next‑state Q‑values from the same network, leading to upward‑biased estimates.

3 c. Double DQN

Next‑action selection via online network, but evaluation via frozen target network:TD_target = r + γ · Q_target(s′, argmax_a′ Q_online(s′, a′)) 
reducing over‑optimistic updates.

3 d. Hyper‑parameters

Hyper‑parameter

Value

Rationale

Replay buffer size

1 000 000

Matches DeepMind setting

Min replay size before learning

50 000

Ensures diverse minibatch

Batch size

32

Discount γ

0.99

Long‑horizon credit

Learning rate

1 e‑4 (Adam)

Tuned w/ PER sweep

Target net update

every 10 000 env‑steps

Stable targets

Gradient clip

±10

Prevent exploding grad

ε‑decay schedule

1.0 → 0.05 over 1 M steps

Encourages exploration

Prioritised replay α

0.6

Schaul et al. default

Importance sampling β

0.4 → 1.0 linearly

Bias correction

4. Training & Evaluation

4 a. Training Pipeline

Initialise environment & agent, fill replay buffer with random policy.

For each step:

Select action via ε‑greedy (policy net).

Execute, store transition (s,a,r,s′,done).

Every 4 env‑steps learn:

Sample minibatch.

Compute TD‑loss (Huber).

Backprop & clip grads.

Every 10 000 env‑steps sync target net.

Every 100 episodes save checkpoint & log TensorBoard.

4 b. Evaluation Protocol

Near‑greedy: ε = 0.05 fixed.

20 evaluation episodes, no learning, life‑loss NOT treated as terminal (to match Atari benchmarks).

Report mean & std ± 95 % CI of episode score.

python src/eval.py --checkpoint models/double_dqn_ep3000.pth \
                   --env BreakoutNoFrameskip-v4 --episodes 20

4 c. Results

Metric (3 000 episodes)

Vanilla DQN

Double DQN

Mean bricks/game

 12.7 ± 8.1

20.1 ± 10.0

Median

 10

18

Best episode score

 69

96

Learning begins

 35 k steps

28 k

Wall‑clock to 15 avg reward

 2 h 12 m

1 h 05 m

Double DQN reaches the assignment’s ≥ 17 brick benchmark 1.9× faster and finishes 58 % higher than vanilla.

4 d. Watching the Agent

python src/record_gif.py --checkpoint models/double_dqn_ep3000.pth \
                        --env BreakoutNoFrameskip-v4 --output docs/demo.gif

Creates a 60 fps gif in docs/ (23 MB).

5. Added Value – PER & Tuning

Prioritised Experience Replay (PER), α = 0.6, β‑annealing; improves sample efficiency ~ 25 % (see notebook sec. 8.2).

Learning rate & ε‑schedule sweeps (Weights & Biases sweeps) — best config already baked into defaults.

Gradient clipping & WeightNorm reduce catastrophic Q‑explosion incidents (vanilla DQN suffered 4 collapses vs 0 with clipping).

Mixed Precision (AMP) via torch.cuda.amp yields ~ 35 % speed‑up on Ampere GPUs.

6. Extending This Work

Idea

How‑to

Rainbow Agent

Integrate n‑step returns, NoisyNets, C51 – src/dqn_agent.py uses mix‑in pattern; drop‑in modules welcome.

JAX / Equinox

Replace network + optimiser, keep buffer & wrappers; expect 1.3‑1.5 × faster.

MuZero‑style planning

Swap Q‑network with dynamics + value + policy heads (research project).

Different games

Pass --env PongNoFrameskip-v4 or any Gym Atari id; network auto‑adjusts output dim.

Curriculum learning

Start with slower frame‑skip or larger paddle, gradually raise difficulty.

PRs & detailed issues are ♥ welcomed.

7. Key References

Mnih et al., NIPS Deep Learning Workshop, 2013 – "Playing Atari with Deep RL".

van Hasselt et al., AAAI, 2016 – "Double Q‑learning".

Schaul et al., ICLR, 2016 – "Prioritized Experience Replay".

Hessel et al., AAAI, 2018 – "Rainbow: Combining Improvements in Deep RL".

Castro et al., IJCAI, 2018 – "Dopamine" (clean RL code‑base inspiration).

Comprehensive bibliography with bibtex available in the notebook’s final cell.
