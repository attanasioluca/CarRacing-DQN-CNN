from DQN import SkipFrame
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation
from DQN import PreprocessFrame
from agent import Agent
import numpy as np
import time  # Import time module for timing

def make_env(render_mode=None):
    env = gym.make('CarRacing-v3', render_mode=render_mode, continuous=False)
    env = SkipFrame(env, skip=4)  # Skip 4 frames
    env = PreprocessFrame(env)  # Convert to grayscale and resize to 84x84
    env = FrameStackObservation(env, 4)  # Stack 4 frames together to capture temporal information
    return env

env = make_env()

state_shape = (4, 84, 84)
action_n = 5
num_episodes = 1000
max_steps_per_episode = 1000
log_interval = 10

agent = Agent(state_shape, action_n, epsilon_decay=((0.1) ** (1 / num_episodes)))

# Training loop
start_time = time.time()  # Start timing the training process
for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0
    for step in range(max_steps_per_episode):
        action = agent.take_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.store_experience(state, action, reward, next_state, done)
        agent.train()

        state = next_state
        total_reward += reward
        if done:
            break
    agent.update_epsilon()
    # Log timing and performance every `log_interval` episodes
    if (episode + 1) % log_interval == 0:
        eval_reward = 0
        old_epsilon = agent.epsilon
        agent.epsilon = 0.0  # Set epsilon to 0 for evaluation (no exploration)
        state, _ = env.reset()
        for step in range(max_steps_per_episode):        
            action = agent.take_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
            eval_reward += reward
            if done:
                break
        agent.epsilon = old_epsilon # Restore old epsilon
        elapsed_time = time.time() - start_time  # Calculate elapsed time
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward:{total_reward:.2f}, "
              f"Epsilon: {agent.epsilon:.2f}, Elapsed Time: {elapsed_time:.2f} seconds,", 
              f"Eval Reward: {eval_reward:.2f}"
        )
        start_time = time.time()  # Reset the timer for the next interval

# Evaluation loop
agent.reset_epsilon()
num_eval_episodes = 10

for episode in range(num_eval_episodes):
    state, _ = env.reset()
    total_reward = 0

    for step in range(max_steps_per_episode):
        action = agent.take_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        state = next_state
        total_reward += reward
        if done:
            break
    print(f"Eval Episode {episode + 1}/{num_eval_episodes}, Total Reward: {total_reward:.2f}")
