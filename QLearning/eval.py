import gymnasium as gym
from qlearn import qlearn
from policies import GreedyPolicy
from gymnasium.wrappers import TimeLimit
import time

def test(env, alpha, gamma, max_episode_steps=200, max_steps=1_000_000, eval_gamma=1, eval_episodes=10_000, verbose=False):
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    start = time.time()
    qtable = qlearn(env=env, alpha0=alpha, gamma=gamma, max_steps=max_steps)
    total_time = time.time() - start
    policy = GreedyPolicy(qtable)
    avg_return = eval(env=env, policy=policy, gamma=eval_gamma, n_episodes=eval_episodes, render=False)
    if(verbose): print(f"Alpha={alpha}, Gamma={gamma}, Max steps={max_steps} => Avg Return={avg_return}, Time taken={total_time} seconds")
    return avg_return

def eval(env: gym.Env, policy, gamma: float, n_episodes: int, render=False):
    unwrapped_env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True) # to keep score consistent with non wrapped baseline
    sum_returns = 0.0
    done = True
    discounting = 1
    ep = 0
    obs = unwrapped_env.reset()[0]
    if render:
        unwrapped_env.render()
    while ep <= n_episodes:
        if done:
            if render:
                print("New episode")
            obs = unwrapped_env.reset()[0]
            discounting = 1
            ep += 1

        action = policy(obs)
        obs, rew, terminated, truncated, info = unwrapped_env.step(action)
        done = terminated or truncated
        sum_returns += rew * discounting
        discounting *= gamma
        if render:
            unwrapped_env.render()

    return sum_returns / n_episodes