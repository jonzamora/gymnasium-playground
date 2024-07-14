import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np


class RelativePosition(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(2,))

    def observation(self, obs):
        return obs["target"] - obs["agent"]
