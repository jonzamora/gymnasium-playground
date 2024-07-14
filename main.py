import gymnasium as gym
import gymnasium_env
from gymnasium.wrappers import FlattenObservation

env = gym.make("gymnasium_env/GridWorld-v0", size=10, render_mode="human")
env = FlattenObservation(env)
observation, info = env.reset(seed=42)

for step in range(300):
    action = env.action_space.sample()  # this is where you would insert your policy
    observation, reward, terminated, truncated, info = env.step(action)
    print(observation)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
