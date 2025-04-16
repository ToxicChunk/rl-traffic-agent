from traffic_env import TrafficEnv
from stable_baselines3 import PPO
import numpy as np

REWARD_SHIFT = 1100  #makes rewards positive for human-friendly comparison

def evaluate_agent(agent, name, episodes=5):
    rewards = []

    for ep in range(episodes):
        env = TrafficEnv()
        obs, _ = env.reset()
        total_reward = 0

        for step in range(100):
            if name == "baseline":
                action = (step // 5) % 2 
            else:
                action, _ = agent.predict(obs)

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        rewards.append(total_reward)
        adjusted = total_reward + REWARD_SHIFT
        print(f"{name} - Episode {ep + 1}: Adjusted Reward = {adjusted:.2f}")

    avg = np.mean(rewards)
    adjusted_avg = avg + REWARD_SHIFT
    print(f"✅ Average Adjusted Reward for {name}: {adjusted_avg:.2f}")
    return adjusted_avg

ppo_agent = PPO.load("ppo_traffic_light")

evaluate_agent(ppo_agent, "PPO")
evaluate_agent(None, "baseline")
