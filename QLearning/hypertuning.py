import gymnasium as gym
from main import test

alphas = [0.05, 0.1]
gammas = [0.99]
max_steps_list = [100000000]

def hypertune_frozenlake(alpha, gamma, max_steps):

    # Case 2: 8x8 FrozenLake, slippery  
    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True)
    test(env, alpha=alpha, gamma=gamma, max_steps=max_steps)

    # Case 3: Car_Racing_v3, discretized observation/action spaces
    # env = DiscretizedCarRacing(continuous=False)
    # test(env, alpha=0.1, gamma=0.99, max_episode_steps=1000, max_steps=100000, eval_gamma=0.99, eval_episodes=100)

for alpha in alphas:
    for gamma in gammas:
        for max_steps in max_steps_list:
            print(f"Testing with alpha={alpha}, gamma={gamma}, max_steps={max_steps}")
            hypertune_frozenlake(alpha, gamma, max_steps)
