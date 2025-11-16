import time
import numpy as np
import torch
import gymnasium as gym
from FLAgent import FLAgent
from FLShaping import FrozenLakeRewardWrapper


def make_env(map_name="8x8", slippery=True, wrapped=True):
    env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=slippery)
    if wrapped: 
        env = FrozenLakeRewardWrapper(env)
    return env


env = make_env(map_name="8x8", slippery=True, wrapped=True)
unwrapped_env = make_env(map_name="8x8", slippery=True, wrapped=False)

n_states = env.observation_space.n
action_n = env.action_space.n
num_episodes = 10000
max_steps_per_episode = 200
log_interval = 10

agent = FLAgent(
    n_states=n_states,
    action_n=action_n,
    epsilon_decay=(0.1) ** (1 / num_episodes),
    lr=0.00001,
    target_update_freq=2000
)

best_eval_reward = -float("inf")
start_time = time.time()
rewards = np.zeros(num_episodes)

for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0.0

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
            eval_total_reward = 0.0
            for step in range(max_steps_per_episode):
                action = agent.take_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                state = next_state
                eval_total_reward += reward
                if done:
                    break
            eval_rewards.append(eval_total_reward)

        eval_reward_mean = float(np.mean(eval_rewards))
        eval_reward_std = float(np.std(eval_rewards))

        # Save the best model
        if eval_reward_mean > best_eval_reward:
            best_eval_reward = eval_reward_mean
            torch.save(agent.q_net.state_dict(), "best_fl_model.pth")
            print(f"New best model saved with Eval Reward: {best_eval_reward:.2f}")

        agent.epsilon = old_epsilon  # Restore epsilon
        elapsed_time = time.time() - start_time
        """
        print(
            f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward:.2f}, "
            f"Epsilon: {agent.epsilon:.2f}, Elapsed Time: {elapsed_time:.2f}s, "
            f"Eval Reward: {eval_reward_mean:.2f} ± {eval_reward_std:.2f}"
        )
        """
        rewards[int(episode/10)] = eval_reward_mean
        start_time = time.time()  # Reset timer for next interval

    if (episode + 1) % 10 == 0:
        old_epsilon = agent.epsilon
        agent.epsilon = 0.0  # No exploration during evaluation
        eval_rewards = []

        for _ in range(50):  # Evaluate over 10 episodes
            state, _ = unwrapped_env.reset()
            eval_total_reward = 0.0
            for step in range(max_steps_per_episode):
                action = agent.take_action(state)
                next_state, reward, terminated, truncated, info = unwrapped_env.step(action)
                done = terminated or truncated
                state = next_state
                eval_total_reward += reward
                if done:
                    break
            eval_rewards.append(eval_total_reward)

        eval_reward_mean = float(np.mean(eval_rewards))
        eval_reward_std = float(np.std(eval_rewards))

        agent.epsilon = old_epsilon  # Restore epsilon
        elapsed_time = time.time() - start_time
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward:.2f}, Non Wrapped Eval Reward: {eval_reward_mean:.2f} ± {eval_reward_std:.2f}")
        start_time = time.time()  # Reset timer for next interval

# Final evaluation using best model
env=make_env(map_name="8x8", slippery=True, wrapped=False)
agent.q_net.load_state_dict(torch.load("best_fl_model.pth", map_location=agent.device))
agent.q_net.eval()
agent.epsilon = 0.0
num_eval_episodes = 100
evals = []

for episode in range(num_eval_episodes):
    state, _ = env.reset()
    total_reward = 0.0
    for step in range(max_steps_per_episode):
        action = agent.take_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward
        if done:
            break
    evals.append(total_reward)
print(f"Avg Eval: {np.mean(evals)} ± {np.std(evals)}")
    