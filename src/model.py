
import torch
import torch.nn.functional as F

from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool


class GCNRegressor(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, dropout=0.2):
        super().__init__()

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.dropout = dropout
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, batch)
        x = F.dropout(
            x, p=self.dropout, training=self.training
        )

        return self.regressor(x).view(-1)
