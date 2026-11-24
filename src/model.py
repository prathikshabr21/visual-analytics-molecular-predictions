
import torch
import torch.nn.functional as F

from torch import nn
from torch_geometric.nn import (
    GCNConv,
    SAGEConv,
    GATConv,
    GINEConv,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
)


# ==========================================
# 1. Graph pooling methods
# ==========================================

POOLING_METHODS = (
    "mean", "sum", "max", "combined"
)


def apply_pooling(x, batch, pooling):

    if pooling == "mean":
        return global_mean_pool(x, batch)

    elif pooling == "sum":
        return global_add_pool(x, batch)

    elif pooling == "max":
        return global_max_pool(x, batch)

    elif pooling == "combined":
        return torch.cat(
            [
                global_mean_pool(x, batch),
                global_add_pool(x, batch),
                global_max_pool(x, batch),
            ],
            dim=1,
        )

    raise ValueError(
        f"Unknown pooling method: {pooling}"
    )


def get_pooling_dim(hidden_dim, pooling):

    if pooling not in POOLING_METHODS:
        raise ValueError(
            f"Unknown pooling method: {pooling}"
        )

    if pooling == "combined":
        return hidden_dim * 3

    return hidden_dim


# ==========================================
# 2. Graph Convolutional Network
# ==========================================

class GCNRegressor(nn.Module):

    def __init__(
        self,
        input_dim=6,
        hidden_dim=64,
        dropout=0.2,
        pooling="mean",
    ):
        super().__init__()

        self.conv1 = GCNConv(
            input_dim, hidden_dim
        )

        self.conv2 = GCNConv(
            hidden_dim, hidden_dim
        )

        self.dropout = dropout
        self.pooling = pooling

        self.regressor = nn.Linear(
            get_pooling_dim(hidden_dim, pooling),
            1
        )

    def forward(
        self, x, edge_index, batch, edge_attr=None
    ):

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        x = apply_pooling(x, batch, self.pooling)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training
        )

        return self.regressor(x).view(-1)


# ==========================================
# 3. GraphSAGE
# ==========================================

class GraphSAGERegressor(nn.Module):

    def __init__(
        self,
        input_dim=6,
        hidden_dim=64,
        dropout=0.2,
        pooling="mean",
    ):
        super().__init__()

        self.conv1 = SAGEConv(
            input_dim,
            hidden_dim,
            aggr="mean"
        )

        self.conv2 = SAGEConv(
            hidden_dim,
            hidden_dim,
            aggr="mean"
        )

        self.dropout = dropout
        self.pooling = pooling

        self.regressor = nn.Linear(
            get_pooling_dim(hidden_dim, pooling),
            1
        )

    def forward(
        self, x, edge_index, batch, edge_attr=None
    ):

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        x = apply_pooling(x, batch, self.pooling)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training
        )

        return self.regressor(x).view(-1)


# ==========================================
# 4. Graph Attention Network
# ==========================================

class GATRegressor(nn.Module):

    def __init__(
        self,
        input_dim=6,
        hidden_dim=64,
        dropout=0.2,
        heads=4,
        pooling="mean",
    ):
        super().__init__()

        if hidden_dim % heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by heads"
            )

        self.conv1 = GATConv(
            input_dim,
            hidden_dim // heads,
            heads=heads,
            concat=True,
        )

        self.conv2 = GATConv(
            hidden_dim,
            hidden_dim,
            heads=1,
            concat=False,
        )

        self.dropout = dropout
        self.pooling = pooling

        self.regressor = nn.Linear(
            get_pooling_dim(hidden_dim, pooling),
            1
        )

    def forward(
        self, x, edge_index, batch, edge_attr=None
    ):

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        x = apply_pooling(x, batch, self.pooling)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training
        )

        return self.regressor(x).view(-1)


# ==========================================
# 5. Edge-aware Graph Attention Network
# ==========================================

class EdgeGATRegressor(nn.Module):

    def __init__(
        self,
        input_dim=6,
        hidden_dim=64,
        dropout=0.2,
        heads=4,
        pooling="mean",
        edge_dim=14,
    ):
        super().__init__()

        if hidden_dim % heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by heads"
            )

        self.conv1 = GATConv(
            input_dim,
            hidden_dim // heads,
            heads=heads,
            concat=True,
            edge_dim=edge_dim,
            fill_value=0.0,
        )

        self.conv2 = GATConv(
            hidden_dim,
            hidden_dim,
            heads=1,
            concat=False,
            edge_dim=edge_dim,
            fill_value=0.0,
        )

        self.dropout = dropout
        self.pooling = pooling

        self.regressor = nn.Linear(
            get_pooling_dim(hidden_dim, pooling),
            1
        )

    def forward(
        self, x, edge_index, batch, edge_attr=None
    ):

        if edge_attr is None:
            raise ValueError(
                "EdgeGAT requires bond features."
            )

        x = F.relu(
            self.conv1(
                x,
                edge_index,
                edge_attr=edge_attr
            )
        )

        x = F.relu(
            self.conv2(
                x,
                edge_index,
                edge_attr=edge_attr
            )
        )

        x = apply_pooling(x, batch, self.pooling)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training
        )

        return self.regressor(x).view(-1)


# ==========================================
# 6. Graph Isomorphism Network with Edges
# ==========================================

class GINERegressor(nn.Module):

    def __init__(
        self,
        input_dim=6,
        hidden_dim=64,
        dropout=0.2,
        pooling="mean",
        edge_dim=14,
    ):
        super().__init__()

        self.node_encoder = nn.Linear(
            input_dim,
            hidden_dim
        )

        mlp1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.conv1 = GINEConv(
            mlp1,
            edge_dim=edge_dim,
            train_eps=True,
        )

        self.conv2 = GINEConv(
            mlp2,
            edge_dim=edge_dim,
            train_eps=True,
        )

        self.dropout = dropout
        self.pooling = pooling

        self.regressor = nn.Linear(
            get_pooling_dim(hidden_dim, pooling),
            1
        )

    def forward(
        self, x, edge_index, batch, edge_attr=None
    ):

        if edge_attr is None:
            raise ValueError(
                "GINE requires bond features."
            )

        x = F.relu(self.node_encoder(x))

        x = F.relu(
            self.conv1(
                x,
                edge_index,
                edge_attr=edge_attr
            )
        )

        x = F.relu(
            self.conv2(
                x,
                edge_index,
                edge_attr=edge_attr
            )
        )

        x = apply_pooling(x, batch, self.pooling)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training
        )

        return self.regressor(x).view(-1)
