import gymnasium as gym
import torch.nn as nn
import numpy as np
import cv2

class PreprocessFrame(gym.ObservationWrapper): # Changes the observation to a grayscale 84x84 image
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84), dtype=np.uint8
        )

    def observation(self, obs): # obs is the original observation from the environment
        # Resize to 84x84 and convert to grayscale
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

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
        channel_n, height, width = in_dim         # Channel_n is the number of frames to consider

        if height != 84 or width != 84:
            raise ValueError(f"DQN model requires input of a (84, 84)-shape. Input of a ({height, width})-shape was passed.")

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=channel_n, out_channels=16, kernel_size=8, stride=4), # Extracts spatial features from the input image
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(), # Converts the 2D feature maps into a 1D feature vector
            nn.Linear(2592, 256), # Maps the flattened feature vector to a hidden layer of 256 neurons
            nn.ReLU(),
            nn.Linear(256, out_dim), # Outputs Q-values for each possible action
        )

    def forward(self, input):
        return self.net(input)


