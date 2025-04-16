# Traffic Light Optimization using Reinforcement Learning (PPO)

This project uses Reinforcement Learning (specifically Proximal Policy Optimization, PPO) to train an agent to control a traffic light at a four-way intersection. The goal is to reduce vehicle queue lengths and prevent overflow by learning efficient switching strategies—without relying on external traffic simulation software like SUMO.

---

## Project Objectives

- Simulate basic urban traffic at a single intersection using a custom Gym-compatible environment.
- Train an RL agent to dynamically control the traffic light based on current lane queue sizes.
- Design a reward system that encourages queue reduction, discourages overflow, and promotes smooth transitions.
- Compare the trained agent's performance against a simple rule-based baseline.

---

## Environment Overview

- **Lanes**: 4 directions (North, South, East, West).
- **Actions**:
  - `0`: NS green, EW red
  - `1`: EW green, NS red
- **Inflow**: Cars enter each lane based on a Poisson distribution.
- **Queues**: Each lane has a capped maximum queue length of 10 vehicles.
- **Reward Function**:
  - + for reducing queue size (delta-based)
  - − for maxed-out lanes (overflow penalty)
  - − for frequent switching (cooldown penalty)

---

## PPO Algorithm Details

- **Policy**: Multi-layer perceptron (MlpPolicy)
- **Framework**: [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- **Training steps**: 1,000,000
- **Entropy coefficient**: Tuned to encourage action exploration

---

## Improvements Made

Over several iterations, the environment and training loop were refined based on agent behavior and reward trends:

- **Reward shaping** using queue deltas to provide feedback for small improvements.
- **Overflow penalties** to strongly discourage letting lanes stay full.
- **Light switching cooldown** to model real-world light transition constraints.
- **Asymmetric car inflow** for more realistic lane behavior.
- **Baseline agent** added for comparison: fixed light switching every 5 steps.
- **Adjusted reward display** for interpretability (shifted to positive scale for visualization).

---

## PPO vs Baseline Performance

Average adjusted reward over 5 test episodes (higher is better):

| Agent      | Avg Adjusted Reward |
|------------|---------------------|
| PPO Agent  | 317.3               |
| Baseline   | 155.8               |

PPO consistently achieved better queue management and overflow reduction than the naive baseline.

---


---

## How to Run

1. Install dependencies:

pip install -r requirements.txt

2. Train the PPO agent

python train_agent.py

3. Test the trained agent

python simulate.py

4. Run a baseline agent

python baseline_agent.py

5. Compare the PPO and baseline performances

python compare_agents.py


Developed by Adnan Jasim Sudheesh as part of a portfolio project in Reinforcement Learning and Applied AI.
Open to feedback, collaborations, or internship opportunities in ML/AI/RL-related roles.