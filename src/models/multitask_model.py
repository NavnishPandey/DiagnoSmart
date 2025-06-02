import torch.nn as nn
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.attention import SelfAttention
import torch 

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
