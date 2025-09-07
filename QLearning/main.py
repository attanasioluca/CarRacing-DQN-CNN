from gymnasium.wrappers import TimeLimit
import gymnasium as gym
from qlearn import qlearn
from policies import GreedyPolicy
from eval import eval
from carRacing import DiscretizedCarRacing

def test(env, alpha, gamma, max_episode_steps=200, max_steps=1000000, eval_gamma=1, eval_episodes=10000):
    #print("Training...")
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    qtable = qlearn(env=env, alpha0=alpha, gamma=gamma, max_steps=max_steps)
    policy = GreedyPolicy(qtable)
    #print("Evaluating...")
    avg_return = eval(env=env, policy=policy, gamma=eval_gamma, n_episodes=eval_episodes, render=False)
    print(f"Alpha={alpha}, Gamma={gamma}, Max steps={max_steps} => Avg Return={avg_return}")

def main():
    # Case 1: 4x4 FrozenLake, slippery
    env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True)
    test(env, alpha=0.1, gamma=0.999, max_steps=100000)
    
    # Case 2: 8x8 FrozenLake, slippery  
    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True)
    test(env, alpha=0.1, gamma=0.999)

    # Case 3: Car_Racing_v3, discretized observation/action spaces
    # env = DiscretizedCarRacing(continuous=False)
    # test(env, alpha=0.1, gamma=0.99, max_episode_steps=1000, max_steps=100000, eval_gamma=0.99, eval_episodes=100)

if __name__ == "__main__":
    main()
