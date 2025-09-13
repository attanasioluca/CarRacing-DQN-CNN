import gymnasium as gym
from carRacing import DiscretizedCarRacing
from hypertuning import hypertune_frozenlake
from hypertuning import test

def main():
    '''
    alpha, gamma = hypertune_frozenlake(size=4, slippery=True, steps=100000000)
    # Case 1: 4x4 FrozenLake, slippery
    env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True)
    score = test(env, alpha=alpha, gamma=gamma, max_episode_steps=100, max_steps=100000000, verbose=True)
    print(f"4x4 FrozenLake, slippery: Best values are: Alpha:{alpha}, Gamma:{gamma}. With score of {score}")

    # Case 2: 8x8 FrozenLake, slippery  
    alpha, gamma = hypertune_frozenlake(size=8, slippery=True, steps=100000000)
    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True)
    score = test(env, alpha=alpha, gamma=gamma, max_steps=100000000, verbose=True)
    print(f"8x8 FrozenLake, slippery: Best values are: Alpha:{alpha}, Gamma:{gamma}. With score of {score}")
    '''
    # Case 3: Discretized CarRacing
    env = gym.make("CarRacing-v3", lap_complete_percent=0.95, domain_randomize=False, continuous=False, render_mode="rgb_array")
    env = DiscretizedCarRacing(env)
    score = test(env, alpha=0.01, gamma=0.99, max_episode_steps=1000, max_steps=10000000, eval_episodes=100, verbose=True)


if __name__ == "__main__":
    main()
