import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, device):
        super(SelfAttention, self).__init__()
        
        # Linear layers to compute Query, Key, and Value from the input
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        
        # Scaling factor to normalize dot product scores (prevents exploding values)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to(device)

    def forward(self, x):
        # Add a sequence dimension (needed for attention mechanism)
        x = x.unsqueeze(1)  # Shape: (batch_size, 1, input_dim)

        # Compute Q, K, V projections
        Q = self.query(x)  # Shape: (batch_size, 1, hidden_dim)
        K = self.key(x)    # Shape: (batch_size, 1, hidden_dim)
        V = self.value(x)  # Shape: (batch_size, 1, hidden_dim)

        # Compute attention scores using scaled dot-product
        attention = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # Shape: (batch_size, 1, 1)

        # Apply softmax to get attention weights
        attention = torch.softmax(attention, dim=-1)  # Shape: (batch_size, 1, 1)

        # Compute the weighted sum of values
        x = torch.matmul(attention, V)  # Shape: (batch_size, 1, hidden_dim)

        # Remove the added sequence dimension
        return x.squeeze(1)  # Shape: (batch_size, hidden_dim)
