from DQN import SkipFrame, PreprocessFrame
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation
from agent import Agent
import numpy as np
import time
import torch

def make_env(render_mode=None):
    env = gym.make('CarRacing-v3', render_mode=render_mode, continuous=False)
    env = SkipFrame(env, skip=4)
    env = PreprocessFrame(env)
    env = FrameStackObservation(env, 4)
    return env

env = make_env()

state_shape = (4, 84, 84)
action_n = 5
num_episodes = 2000
max_steps_per_episode = 1000
log_interval = 10
 
agent = Agent(state_shape, action_n, epsilon_decay=((0.1) ** (1 / num_episodes)), lr=0.00001)

best_eval_reward = -float("inf")
start_time = time.time()

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

    # Logging and evaluation
    if (episode + 1) % log_interval == 0:
        old_epsilon = agent.epsilon
        agent.epsilon = 0.0  # No exploration during evaluation
        eval_rewards = []

        for _ in range(10):  # Evaluate over 10 episodes
            state, _ = env.reset()
            eval_total_reward = 0
            for step in range(max_steps_per_episode):
                action = agent.take_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                state = next_state
                eval_total_reward += reward
                if done:
                    break
            eval_rewards.append(eval_total_reward)

        eval_reward_mean = np.mean(eval_rewards)
        eval_reward_std = np.std(eval_rewards)

        # Save the best model
        if eval_reward_mean > best_eval_reward:
            best_eval_reward = eval_reward_mean
            torch.save(agent.q_net.state_dict(), "best_model.pth")
            print(f"New best model saved with Eval Reward: {best_eval_reward:.2f}")

        agent.epsilon = old_epsilon  # Restore epsilon
        elapsed_time = time.time() - start_time
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward:.2f}, "
              f"Epsilon: {agent.epsilon:.2f}, Elapsed Time: {elapsed_time:.2f}s, "
              f"Eval Reward: {eval_reward_mean:.2f} ± {eval_reward_std:.2f}")
        start_time = time.time()  # Reset timer for next interval

# Final evaluation using best model
agent.q_net.load_state_dict(torch.load("best_model.pth"))
agent.q_net.eval()
agent.epsilon = 0.0
num_eval_episodes = 100

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
