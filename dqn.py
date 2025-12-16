import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # Q-value for one action

    def forward(self, obs_batch):
        # obs_batch: (batch_size, 40, 13)
        x = F.relu(self.fc1(obs_batch))  # shape: (batch_size, 40, hidden_dim)
        q_values = self.fc2(x).squeeze(-1)  # shape: (batch_size, 40)
        return q_values
