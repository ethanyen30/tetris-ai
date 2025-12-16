from collections import deque
import random
import torch

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, action, reward, next_obs, done = zip(*batch)
        return (
            torch.tensor(obs, dtype=torch.float32),    # (batch, 40, 13)
            torch.tensor(action, dtype=torch.long),    # (batch,)
            torch.tensor(reward, dtype=torch.float32), # (batch,)
            torch.tensor(next_obs, dtype=torch.float32),# (batch, 40, 13)
            torch.tensor(done, dtype=torch.float32),   # (batch,)
        )

    def __len__(self):
        return len(self.buffer)
