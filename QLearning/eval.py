import gymnasium as gym

def eval(env: gym.Env, policy, gamma: float, n_episodes: int, render=False,):
    sum_returns = 0.0
    done = True
    discounting = 1
    ep = 0
    obs = env.reset()[0]
    if render:
        env.render()

    while ep <= n_episodes:
        if done:
            if render:
                print("New episode")
            obs = env.reset()[0]
            discounting = 1
            ep += 1

        action = policy(obs)
        obs, rew, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        sum_returns += rew * discounting
        discounting *= gamma
        if render:
            env.render()

    return sum_returns / n_episodes