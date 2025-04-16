from traffic_env import TrafficEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

env = Monitor(TrafficEnv())

model = PPO("MlpPolicy", env, verbose=1, ent_coef=0.01)
model.learn(total_timesteps=1_000_000)

model.save("ppo_traffic_light")
print("✅ Model saved as ppo_traffic_light.zip")
