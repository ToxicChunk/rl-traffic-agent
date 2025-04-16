from traffic_env import TrafficEnv
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

env = TrafficEnv()
model = PPO.load("ppo_traffic_light")

obs = env.reset()
queue_history = []

for _ in range(100):
    action, _ = model.predict(obs)
    obs, _, done, _ = env.step(action)
    queue_history.append(sum(obs))
    if done:
        break

plt.plot(queue_history)
plt.xlabel("Step")
plt.ylabel("Total Queue Length")
plt.title("Traffic Queue Over Time (PPO Agent)")
plt.grid(True)
plt.show()
