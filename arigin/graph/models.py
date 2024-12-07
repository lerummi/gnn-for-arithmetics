from torch.nn import Linear, Module, ModuleList
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
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
    

class UniMP(Module):
    def __init__(
            self, 
            in_channels, 
            num_classes, 
            hidden_channels, 
            num_layers,
            heads, 
            dropout=0.3
        ):
        super().__init__()

        self.label_emb = MaskLabel(num_classes, in_channels)

        self.convs = ModuleList()
        self.norms = ModuleList()
        for i in range(1, num_layers + 1):
            if i < num_layers:
                out_channels = hidden_channels // heads
                concat = True
            else:
                out_channels = num_classes
                concat = False
            conv = TransformerConv(in_channels, out_channels, heads,
                                   concat=concat, beta=True, dropout=dropout)
            self.convs.append(conv)
            in_channels = hidden_channels

            if i < num_layers:
                self.norms.append(LayerNorm(hidden_channels))

    def forward(self, x, y, edge_index, label_mask):
        x = self.label_emb(x, y, label_mask)
        for conv, norm in zip(self.convs, self.norms):
            x = norm(conv(x, edge_index)).relu()
        return self.convs[-1](x, edge_index)