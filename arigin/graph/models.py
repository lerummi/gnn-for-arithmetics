import torch
from torch.nn import Linear, Module
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv
from torch_geometric.nn import pool
from torch_geometric.nn import norm
from torch_geometric.nn import (
    TransformerConv, 
    MaskLabel,
    LayerNorm
)


class GCN(Module):
    """
    A convolutional graph neural network.
    
    Parameters
    ----------
    in_channels : int
        Number of input features.
    hidden_channels : int
        Number of hidden features.
    emb_channels : int
        Number of embedding features.
    out_channels : int
        Number of output features.
    gat_activation : callable
        GAT activation function.
    dropout_inter_layer : float
        Dropout rate for intermediate layers.
    """
    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            emb_channels: int,
            out_channels: int,
            gat_activation = F.elu,
            dropout_inter_layer=0.1,
            ):

        super().__init__()

        self.embedding_1 = Linear(in_channels, hidden_channels)
        self.embedding_2 = Linear(emb_channels, emb_channels)
        self.embedding_3 = Linear(emb_channels, hidden_channels)

        self.gatconv_1 = GCNConv(
                hidden_channels,
                hidden_channels, 
                improved=True
            )
        
        self.gatconv_2 = GCNConv(
                hidden_channels,
                hidden_channels, 
                improved=True
            )
        
        self.gatconv_3 = GCNConv(
                hidden_channels,
                hidden_channels, 
                improved=True
            )
        
        self.gatconv_4 = GCNConv(
                hidden_channels,
                hidden_channels, 
                improved=True
            )

        self.head = Linear(hidden_channels, out_channels)

        self.dropout_inter_layer = dropout_inter_layer
        self.gat_activation = gat_activation


    def embedding(self, x):

        x = self.embedding_1(x)
        x = self.gat_activation(x)
        x = F.dropout(x, p=self.dropout_inter_layer, training=self.training)
        x = self.embedding_2(x)
        x = self.gat_activation(x)
        x = F.dropout(x, p=self.dropout_inter_layer, training=self.training)
        x = self.embedding_3(x)
        x = self.gat_activation(x)
        x = F.dropout(x, p=self.dropout_inter_layer, training=self.training)

        return x

    def forward(self, x, edge_index, edge_weight):

        x_ = self.embedding(x)

        x_ = self.gatconv_1(x_, edge_index, edge_weight)
        x_ = self.gat_activation(x_)
        x_ = F.dropout(x_, p=self.dropout_inter_layer, training=self.training)
        x_ = self.gatconv_2(x_, edge_index, edge_weight)
        x_ = self.gat_activation(x_)
        x_ = F.dropout(x_, p=self.dropout_inter_layer, training=self.training)
        x_ = self.gatconv_3(x_, edge_index, edge_weight)
        x_ = self.gat_activation(x_)
        x_ = F.dropout(x_, p=self.dropout_inter_layer, training=self.training)
        x_ = self.gatconv_4(x_, edge_index, edge_weight)
        x_ = self.gat_activation(x_)
        x_ = F.dropout(x_, p=self.dropout_inter_layer, training=self.training)

        x_ = self.head(x_)

        return x_
    

class MathModel(torch.nn.Module):
    def __init__(
            self, 
            in_channels: int,
            emb_channels: int,
            hidden_channels: int, 
            out_channels: int, 
            heads: int = 1,
            edge_dim: int = None,
            activation = F.gelu,
            dropout=0.2):

        super().__init__()

        self.n_conv = 3
        self.n_pool = 3

        self.init_embedding = Linear(in_channels, hidden_channels * heads)

        self.conv = torch.nn.ModuleList()
        for _ in range(self.n_pool):
            self.conv.append(torch.nn.ModuleList())
            for _ in range(self.n_conv):
                self.conv[-1].append(
                    GATv2Conv(
                    hidden_channels * heads,
                    hidden_channels, 
                    heads=heads,
                    bias=False,
                    add_self_loops=False,
                    edge_dim=edge_dim,
                    dropout=dropout
                    )
                )

        self.pool = torch.nn.ModuleList()
        for _ in range(self.n_pool):
            self.pool.append(
                pool.SAGPooling(
                    hidden_channels * heads,
                    ratio=0.5,
                    GNN=GATv2Conv,
                    heads=1,
                )
            )

        self.final_embedding = Linear(hidden_channels * heads, emb_channels)
        self.norm = LayerNorm(hidden_channels * heads)

        self.head = torch.nn.ModuleList()
        self.head.append(Linear(emb_channels, emb_channels))
        self.head.append(Linear(emb_channels, out_channels))

        self.activation = activation

    def embedding(self, x, edge_index, edge_attr, batch):

        x = self.init_embedding(x)

        out = 0
        for ip, p in enumerate(self.pool):
            for c in self.conv[ip]:
                x_ = c(x, edge_index, edge_attr)
                x = self.norm(x + self.activation(x_))

            out += pool.global_max_pool(x, batch)

            x, edge_index, edge_attr, batch, _, _ = p(x, edge_index, edge_attr, batch=batch)

        return self.final_embedding(out)

    def forward(self, x, edge_index, edge_attr, batch):

        x = self.embedding(x, edge_index, edge_attr, batch)

        for h in self.head:
            x = self.activation(x)
            x = h(x)

        return x
