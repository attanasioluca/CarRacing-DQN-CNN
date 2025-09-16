import gymnasium as gym
from hypertuning import hypertune_frozenlake
from eval import test
from FLShaping import FrozenLakeRewardWrapper
import time
from gymnasium.wrappers import TimeLimit


def main():

    alpha, gamma = hypertune_frozenlake(size=8, slippery=True, steps=100_000_000)
    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True)
    score = test(env, alpha=alpha, gamma=gamma, max_steps=10_000_000, verbose=True)
    print(f"8x8 FrozenLake, slippery: Best values are: Alpha:{alpha}, Gamma:{gamma}. With score of {score}")

if __name__ == "__main__":
    main()
