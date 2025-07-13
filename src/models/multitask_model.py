import torch.nn as nn
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.attention import SelfAttention
import torch 

# The input — consisting of the original embedding vectors concatenated with the output of the attention mechanism —
# is first passed through a linear layer that compresses each vector into a 512-dimensional representation.
# A ReLU activation is then applied to introduce non-linearity and allow the model to learn complex patterns.
# To improve training stability and avoid overfitting, Batch Normalization and Dropout are applied immediately after.
# This is followed by a second linear layer that reduces the representation to 256 features.
# Again, a ReLU activation is applied, followed by BatchNorm and Dropout for regularization.
# These shared layers learn a compact and general-purpose feature representation
# that is then fed into the three specialized output branches (specialty, severity, chronicity).

class MultiTargetAttentionModel(nn.Module):
    def __init__(self, input_dim, num_specialties, num_severities, device):
        super(MultiTargetAttentionModel, self).__init__()
        self.attention_heads = nn.ModuleList([
            SelfAttention(input_dim, input_dim // 4, device)
            for _ in range(4)
        ])

        self.shared_input = nn.Linear(input_dim * 2, 512)
        self.shared_bn1 = nn.BatchNorm1d(512)
        self.shared_dropout1 = nn.Dropout(0.5)
        self.shared_hidden = nn.Linear(512, 256)
        self.shared_bn2 = nn.BatchNorm1d(256)
        self.shared_dropout2 = nn.Dropout(0.4)

        self.specialty_branch = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_specialties)
        )

        self.severity_branch = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_severities)
        )

        self.chronic_branch = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    # A Linear layer applies a linear transformation to the input: output = xW^T + b
    # It learns weights W and bias b that best transform the input features into a new space.
    # This helps the model extract compact and meaningful representations for downstream tasks.

    # If the model were just a composition of linear functions (linear layer + linear layer + ...),
    # the overall result would still be a linear function (i.e., a matrix multiplied by the input).
    # This limits the model's ability to represent complex functions, because a combination
    #  of linear transformations is always a linear transformation.
    # By introducing a non-linear function like ReLU=max(0,x) after each linear layer,
    # the model becomes capable of learning much more complex relationships and patterns in the data.
    # ReLU "turns neurons on or off": if the input value is negative, it becomes zero → neuron "off"
    # If it is positive, it passes the value as is → neuron "on"
    # This allows the model to selectively use parts of the network depending on the input,
    # creating non-linear combinations.

    def forward(self, x):
        attention_outputs = [head(x) for head in self.attention_heads]
        attention = torch.cat(attention_outputs, dim=1)
        combined = torch.cat([x, attention], dim=1)

        shared = self.shared_input(combined)
        shared = nn.functional.relu(shared)
        shared = self.shared_bn1(shared)
        shared = self.shared_dropout1(shared)
        shared = self.shared_hidden(shared)
        shared = nn.functional.relu(shared)
        shared = self.shared_bn2(shared)
        shared = self.shared_dropout2(shared)

        return (
            self.specialty_branch(shared),
            self.severity_branch(shared),
            self.chronic_branch(shared)
        )
