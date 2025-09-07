import gymnasium as gym
import torch.nn as nn
import numpy as np
import cv2
import torch

class PreprocessFrame(gym.ObservationWrapper):
    """Convert frames to grayscale (84x84) and return HxW (no channel dim)."""
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84), dtype=np.uint8
        )

    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized  # shape: (84,84)
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            state, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated:
                break
        return state, total_reward, terminated, truncated, info




class DQN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        channel_n = in_dim[0]

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=channel_n, out_channels=16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute conv output size dynamically using a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, *in_dim)  # shape: (1, C, H, W)
            n_flatten = self.feature_extractor(dummy).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(n_flatten, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.fc(x)


