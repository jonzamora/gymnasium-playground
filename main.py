import gymnasium as gym
import gymnasium_env
from gymnasium.wrappers import FlattenObservation, RecordEpisodeStatistics, RecordVideo

num_eval_episodes = 4
env = gym.make("gymnasium_env/GridWorld-v0", size=10, render_mode="rgb_array")
env = FlattenObservation(env)
env = RecordVideo(env, video_folder="videos", episode_trigger=lambda x: True)
env = RecordEpisodeStatistics(env, deque_size=num_eval_episodes)

for episode_num, seed in zip(range(num_eval_episodes), range(42, 46)):
    observation, info = env.reset(seed=seed)
    episode_over = False

    while not episode_over:
        action = env.action_space.sample()  # this is where you would insert your policy
        observation, reward, terminated, truncated, info = env.step(action)
        episode_over = terminated or truncated

env.close()

print(f"Episode Total Rewards: {env.return_queue}")
print(f"Episode Lengths: {env.length_queue}")
