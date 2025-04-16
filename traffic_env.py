import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TrafficEnv(gym.Env):
    def __init__(self):
        super(TrafficEnv, self).__init__()

        self.num_lanes = 4  # N, S, E, W
        self.max_queue = 10
        self.max_steps = 100
        self.current_step = 0

        self.cooldown_period = 2  #steps before allowing action switch
        self.cooldown_counter = 0

        self.prev_action = None
        self.last_action = None

        self.observation_space = spaces.Box(low=0, high=self.max_queue, shape=(self.num_lanes,), dtype=np.int32)
        self.action_space = spaces.Discrete(2)  #0 = NS green, 1 = EW green

        self.state = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.cooldown_counter = 0
        self.prev_action = None
        self.last_action = None

        #initialize state with slight randomization
        self.state = np.random.randint(low=2, high=6, size=(self.num_lanes,), dtype=np.int32)
        return self.state, {}

    def step(self, action):
        self.current_step += 1
        queue_before = np.sum(self.state)

        #light switching cooldown logic
        if self.last_action is not None and action != self.last_action and self.cooldown_counter < self.cooldown_period:
            penalty = 5  #discourages rapid light switching
        else:
            penalty = 0
            self.last_action = action
            self.cooldown_counter = 0

        self.cooldown_counter += 1

        #cars arrive at each lane
        new_cars = np.random.poisson([1.8, 2.0, 1.5, 1.7])  #balanced, but slightly asymmetric inflow
        cars_passed = np.zeros(self.num_lanes)

        if action == 0:  #NS green
            cars_passed[0] = min(2, self.state[0])
            cars_passed[1] = min(2, self.state[1])
        else:  #EW green
            cars_passed[2] = min(2, self.state[2])
            cars_passed[3] = min(2, self.state[3])

        self.state = np.maximum(0, self.state - cars_passed) + new_cars
        self.state = np.clip(self.state, 0, self.max_queue).astype(np.int32)

        #reward design
        queue_after = np.sum(self.state)
        queue_delta = queue_before - queue_after

        overflow_penalty = np.sum(self.state == self.max_queue) * 4
        reward = (queue_delta * 0.5) - overflow_penalty - penalty

        #clip reward to stabilize learning
        reward = float(np.clip(reward, -50, 50))

        terminated = False
        truncated = self.current_step >= self.max_steps
        return self.state, reward, terminated, truncated, {}

    def render(self):
        print(f"Step {self.current_step}: Queues = {self.state}")
