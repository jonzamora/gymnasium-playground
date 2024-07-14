import gymnasium as gym
from gymnasium import spaces
from enum import Enum
import numpy as np
import pygame


class Actions(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size  # The size of the Square Grid
        self.window_size = 512  # The size of the PyGame Window

        # Observations are Dictionaries with the Agent's and Target's Location
        # Each Location is encoded as an element of {0,...,`size`}^2, i.e. MultiDiscrete([size, size])
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=0, high=size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(low=0, high=size - 1, shape=(2,), dtype=int),
            }
        )

        self._agent_location = np.array([-1, -1], dtype=int)
        self._target_location = np.array([-1, -1], dtype=int)

        # We have 4 Actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(n=4)

        """
        The following dictionary maps Abstract Actions from `self.action_space` to 
        the Direction we will walk in if that Action is taken
        (i.e. 0 corresponds to "right", 1 to "up", 2 to "left", 3 to "down")
        """
        self._action_to_direction = {
            Actions.RIGHT.value: np.array([1, 0]),
            Actions.UP.value: np.array([0, 1]),
            Actions.LEFT.value: np.array([-1, 0]),
            Actions.DOWN.value: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If `"human"` rendering is used, `self.window` will be a reference to the window
        that we draw to. `self.clock` will be a clock that is used to ensure that the environment
        is rendered at the correct framerate in `"human"` mode. They will remain `None` until
        `"human"` mode is used for the first time
        """
        self.window = None
        self.clock = None
