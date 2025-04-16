from traffic_env import TrafficEnv
from stable_baselines3 import PPO

env = TrafficEnv()
model = PPO.load("ppo_traffic_light")

obs, _ = env.reset()
total_reward = 0

for step in range(100):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    env.render()
    print(f"Step {step + 1}: Action = {action}, Queues = {obs}")
    if terminated or truncated:
        break

print(f"Total Reward: {total_reward}")
