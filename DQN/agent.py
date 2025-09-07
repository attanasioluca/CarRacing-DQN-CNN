import torch
import numpy as np
from DQN import DQN
from collections import deque

class Agent:
    def __init__(self, state_space_shape, action_n, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1, lr=0.001):
        """
        A simple DQN agent for environments with discrete action spaces.

        Args:
            state_space_shape (tuple): Shape of the state space (e.g., (4, 84, 84)).
            action_n (int): Number of actions (discrete).
            gamma (float): Discount factor for future rewards.
            epsilon (float): Initial exploration rate.
            epsilon_decay (float): Decay rate for epsilon.
            epsilon_min (float): Minimum exploration rate.
            lr (float): Learning rate for the optimizer.
        """
        self.state_shape = state_space_shape
        self.action_n = action_n  # Number of discrete actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Device setup
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

        # Q-network
        self.q_net = DQN(self.state_shape, self.action_n).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = torch.nn.MSELoss()  # Mean Squared Error loss

        # Replay buffer
        self.buffer_size = 10000  # Maximum size of the replay buffer
        self.replay_buffer = deque(maxlen=self.buffer_size)  # Simple list for storing experiences
        self.batch_size = 32  # Batch size for training

    def store_experience(self, state, action, reward, next_state, done):
        """
        Stores an experience in the replay buffer.
        """
        self.replay_buffer.append((state, action, reward, next_state, done))

    def sample_experiences(self):
        """
        Samples a batch of experiences from the replay buffer.
        """
        indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert lists of NumPy arrays to a single NumPy array
        states = np.array(states)
        next_states = np.array(next_states)

        # Convert actions to a 1D NumPy array
        actions = np.array(actions)  # Shape: [batch_size]

        return (
            torch.tensor(states, dtype=torch.float32, device=self.device),
            torch.tensor(actions, dtype=torch.long, device=self.device),  # Correctly formatted actions
            torch.tensor(rewards, dtype=torch.float32, device=self.device),
            torch.tensor(next_states, dtype=torch.float32, device=self.device),
            torch.tensor(dones, dtype=torch.float32, device=self.device),
        )

    def take_action(self, state):
        """
        Selects an action using an epsilon-greedy policy and maps it to a continuous action.
        """
        if np.random.rand() < self.epsilon:
            discrete_action = np.random.randint(self.action_n)  # Random action (exploration)
        else:
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_net(state)
            discrete_action = torch.argmax(q_values, dim=1).item()  # Best action (exploitation)

        # Map the discrete action to a continuous action
        return discrete_action

    def train(self):
        """
        Trains the Q-network using a batch of experiences from the replay buffer.
        """
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough experiences to sample a batch

        # Sample a batch of experiences
        states, actions, rewards, next_states, dones = self.sample_experiences()

        # Compute Q-values for the current states
        q_values = self.q_net(states)  # Shape: [batch_size, action_n]
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # Ensure actions is 2D

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.q_net(next_states).max(1)[0]  # Shape: [batch_size]
            targets = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute the loss
        loss = self.loss_fn(q_values, targets)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        """
        Decays epsilon according to the decay schedule.
        """
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def reset_epsilon(self):
        """
        Resets epsilon to its initial value (useful for evaluation).
        """
        self.epsilon = 1.0