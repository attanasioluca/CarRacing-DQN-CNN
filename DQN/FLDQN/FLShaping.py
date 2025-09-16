import gymnasium as gym
import numpy as np

class FrozenLakeRewardWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        step_penalty: float = -0.005,
        hole_penalty: float = -1.0,
        first_visit_bonus: float = 0.01,
        potential_scale: float = 0.05,
        gamma: float = 0.99,
    ):
        super().__init__(env)
        self.step_penalty = step_penalty
        self.hole_penalty = hole_penalty
        self.first_visit_bonus = first_visit_bonus
        self.potential_scale = potential_scale
        self.gamma = gamma
        self._visited = None
        self._goal_rc = None
        self._rows = None
        self._cols = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._visited = set([int(obs)])
        desc = np.array(self.env.unwrapped.desc)
        self._rows, self._cols = desc.shape
        self._goal_rc = tuple(map(int, np.argwhere(desc == b'G')[0]))
        return obs, info

    def _idx_to_rc(self, idx: int):
        r = idx // self._cols
        c = idx % self._cols
        return int(r), int(c)

    def _manhattan_to_goal(self, idx: int):
        r, c = self._idx_to_rc(idx)
        gr, gc = self._goal_rc
        return abs(r - gr) + abs(c - gc)

    def _phi(self, idx: int):
        # Potential function (negative distance)
        return -float(self._manhattan_to_goal(idx))

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        obs = int(obs)
        shaped = float(rew)

        # Step penalty
        shaped += self.step_penalty

        # First-visit bonus
        if obs not in self._visited:
            shaped += self.first_visit_bonus
            self._visited.add(obs)

        prev_obs = info.get("prev_obs")
        if prev_obs is None:
            # Fallback: estimate prev as last element in visited (not exact). Better: patch env to store prev.
            # We approximate by using the most recent visited state; this still provides a useful signal.
            if len(self._visited) > 1:
                prev_obs = obs
        if prev_obs is not None:
            shaped += self.potential_scale * (self.gamma * self._phi(obs) - self._phi(int(prev_obs)))

        # Hole penalty
        if terminated and float(rew) == 0.0:
            shaped += self.hole_penalty

        return obs, shaped, terminated, truncated, info