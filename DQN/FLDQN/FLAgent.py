import torch
import numpy as np
from collections import deque
from FLDQN import FLDQN

class FLAgent:
    def __init__(self, n_states, action_n, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1, lr=0.000025, target_update_freq=10000):
        self.n_states = n_states
        self.action_n = action_n
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

        self.q_net = FLDQN(self.n_states, self.action_n).to(self.device)
        self.q_net_target = FLDQN(self.n_states, self.action_n).to(self.device)
        self.q_net_target.load_state_dict(self.q_net.state_dict())
        self.q_net_target.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = torch.nn.SmoothL1Loss() #Huber Loss

        self.buffer_size = 10000
        self.replay_buffer = deque(maxlen=self.buffer_size)
        self.batch_size = 128

        self.target_update_freq = target_update_freq  #update every 10k steps
        self.train_step = 0

    def store_experience(self, state, action, reward, next_state, done):
        reward = np.clip(reward, -1.0, 1.0)
        self.replay_buffer.append((state, action, reward, next_state, done))

    def sample_experiences(self):
        indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.long, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.long, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        return states, actions, rewards, next_states, dones

    def take_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_n)
        state_tensor = torch.tensor([state], dtype=torch.long, device=self.device)
        q_values = self.q_net(state_tensor)
        return torch.argmax(q_values, dim=1).item()

    def train(self):
        if len(self.replay_buffer) < max(self.batch_size * 10, 1000):
            return

        states, actions, rewards, next_states, dones = self.sample_experiences()

        # Q(s,a)
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.q_net_target(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.q_net_target.load_state_dict(self.q_net.state_dict())

    def update_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def reset_epsilon(self):
        self.epsilon = 1.0