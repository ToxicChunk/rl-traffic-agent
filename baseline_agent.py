from traffic_env import TrafficEnv

env = TrafficEnv()
obs, _ = env.reset()
total_reward = 0
steps_per_action = 5
current_action = 0

for step in range(100):
    if step % steps_per_action == 0:
        current_action= 1 - current_action 

    obs, reward, terminated, truncated, _ = env.step(current_action)
    total_reward += reward
    env.render()
    print(f"Step {step + 1}: Action = {current_action}, Queues = {obs}")

    if terminated or truncated:
        break

print(f"🧠 Baseline Agent Total Reward: {total_reward}")
