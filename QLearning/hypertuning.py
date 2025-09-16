import gymnasium as gym
from eval import test
from FLShaping import FrozenLakeRewardWrapper

def hypertune_frozenlake(size=8, slippery=True, steps=1_000_000):
    alphas = [0.05, 0.1, 0.2, 0.5, 0.9]
    gammas = [0.9, 0.99, 0.999, 0.9999]
    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=slippery) if size==8 else gym.make('FrozenLake-v1', map_name="4x4", is_slippery=slippery)
    env = FrozenLakeRewardWrapper(env)
    best_score = -float('inf')
    best_params = (None, None)
    for alpha in alphas:
        for gamma in gammas:
                avg_return = test(env, alpha, gamma, max_steps=steps, eval_episodes=1000, verbose=True)
                if avg_return > best_score:
                    best_score = avg_return
                    best_params = (alpha, gamma)
    return best_params