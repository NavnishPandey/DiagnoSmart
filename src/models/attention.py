import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, device):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to(device)

    def forward(self, x):
        x = x.unsqueeze(1)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention = torch.softmax(attention, dim=-1)
        x = torch.matmul(attention, V)
        return x.squeeze(1)
