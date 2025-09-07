import gymnasium as gym
import numpy as np

class DiscretizedCarRacing(gym.Env):
    def __init__(self, lap_complete_percent=0.95, domain_randomize=False, continuous=False, reward_params=None):
        """
        Wrapper for CarRacing-v3 with discretized observation space and reward shaping.

        Args:
            lap_complete_percent (float): Percentage of the lap required to complete an episode.
            domain_randomize (bool): Whether to randomize the environment.
            continuous (bool): Whether to use continuous action space.
            reward_params (dict): Parameters for reward shaping.
        """
        self.env = gym.make(
            "CarRacing-v3",
            lap_complete_percent=lap_complete_percent,
            domain_randomize=domain_randomize,
            continuous=continuous,  
            render_mode="rgb_array"
        )
        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Discrete(10_000)

        # Reward shaping parameters
        self.reward_params = reward_params or {
            "speed_coeff": 0.01,
            "angular_coeff": 0.005,
            "penalty": 0.001
        }

    def reset(self, **kwargs):
        """
        Resets the environment and returns the discretized initial observation.
        """
        obs, info = self.env.reset(**kwargs)
        return self._discretize_obs(), info

    def step(self, action):
        """
        Takes a step in the environment.

        Args:
            action (int): Discrete action to take.

        Returns:
            tuple: Discretized observation, shaped reward, termination flag, truncation flag, and info.
        """
        obs, rew, terminated, truncated, info = self.env.step(action)
        rew = self._shape_reward(rew)
        return self._discretize_obs(), rew, terminated, truncated, info

    def _discretize_obs(self):
        """
        Discretizes the observation space into a unique state index.

        Returns:
            int: Discretized state index.
        """
        unwrapped = self.env.unwrapped
        car = unwrapped.car

        # Extract relevant features
        tile_id = getattr(unwrapped, "tile_id", 0)
        vel = car.hull.linearVelocity
        speed = np.linalg.norm(vel)
        angle = car.hull.angle

        # Discretize features
        vel_bin = self._discretize_speed(speed)
        angle_bin = self._discretize_angle(angle)

        # Combine features into a unique state index
        state = (tile_id * 100) + (vel_bin * 10) + angle_bin
        return state % 10_000

    def _discretize_speed(self, speed):
        """
        Discretizes the speed into bins.

        Args:
            speed (float): The speed of the car.

        Returns:
            int: Discretized speed bin.
        """
        return int(np.digitize(speed, bins=np.linspace(0, 100, 5)))

    def _discretize_angle(self, angle):
        """
        Discretizes the angle into bins.

        Args:
            angle (float): The angle of the car.

        Returns:
            int: Discretized angle bin.
        """
        ang_norm = (angle + np.pi) % (2 * np.pi)  # Normalize angle to [0, 2π]
        return int(ang_norm / (2 * np.pi / 8))  # Divide into 8 bins

    def _shape_reward(self, reward):
        """
        Shapes the reward to encourage forward motion and penalize angular velocity.

        Args:
            reward (float): Original reward from the environment.

        Returns:
            float: Shaped reward.
        """
        car = self.env.unwrapped.car
        vel = car.hull.linearVelocity
        speed_forward = np.dot(
            vel, np.array([np.cos(car.hull.angle), np.sin(car.hull.angle)])
        )

        # Apply reward shaping
        params = self.reward_params
        shaped = reward
        shaped += params["speed_coeff"] * speed_forward
        shaped -= params["angular_coeff"] * abs(car.hull.angularVelocity)
        shaped -= params["penalty"]
        return shaped
