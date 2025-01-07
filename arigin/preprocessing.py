import torch
import numpy as np
from typing import Optional
from torch_geometric.data import Data
from sklearn.base import BaseEstimator, TransformerMixin

from arigin.graph.generation import GraphEntities
from arigin.features import node_features, edge_features


class GraphEntityToDataSet(BaseEstimator, TransformerMixin):

    def __init__(
            self, 
            node_transformer: Optional[TransformerMixin] = node_features, 
            edge_transformer: Optional[TransformerMixin] = edge_features,
            target_transformer: Optional[TransformerMixin] = None
        ):

        self.node_transformer = node_transformer
        self.edge_transformer = edge_transformer
        self.target_transformer = target_transformer
        super().__init__()

    def _get_node_id_to_index(self, graph_entities: GraphEntities) -> dict:
        """
        Get a dictionary mapping node ids to their index in the nodes list.
        """

        return {
            node.id: index
            for index, node in enumerate(graph_entities["nodes"])
        }

    def _get_edges(self, graph_entities: GraphEntities) -> list:
        """
        Get a list of edges from the relationships list.
        """

        id_index_mapping = self._get_node_id_to_index(graph_entities)

        edge_index = [
            [
                id_index_mapping[relationship.source.id],
                id_index_mapping[relationship.target.id]
            ]
            for relationship in graph_entities["relationships"]
        ]
        edge_index = np.array(edge_index)
        # Add reverse edges
        edge_index = np.vstack((edge_index, edge_index[:, [1, 0]]))

        return edge_index
    
    def fit(self, X: GraphEntities, y: np.ndarray, **fit_params):
        """
        Fit the transformer to the data.
        """

        self.node_transformer.fit(X["nodes"])
        self.edge_transformer.fit(X["relationships"])
        if self.target_transformer is not None:
            self.target_transformer.fit(y)
        return self
    
    def fit_transform(self, X: GraphEntities, y: np.ndarray, **fit_params):
        return super().fit_transform(X, y, **fit_params)
    
    def transform(self, X: GraphEntities, y: np.ndarray = None, **transform_params):
        """
        Transform the data to pytorch DataSet.
        """

        x = self.node_transformer.transform(X["nodes"])
        E = self.edge_transformer.transform(X["relationships"])
        if self.target_transformer is not None:
            y = self.target_transformer.transform(y)
        Ez = np.zeros_like(E)
        # Reverse edges are added to the edge features
        E = np.vstack(
            (
                np.hstack((E, Ez)),
                np.hstack((Ez, E))
            )
        )
        edge_index = self._get_edges(X)

        # Convert to pytorch tensors
        edge_index = torch.tensor(edge_index.T, dtype=torch.long)
        x = torch.tensor(x, dtype=torch.float)
        E = torch.tensor(E, dtype=torch.float)
        if y is not None:
            y = torch.tensor(y, dtype=torch.float)
        batch_no = torch.tensor(X["batch"], dtype=torch.long)

        return Data(x=x, edge_index=edge_index, edge_attr=E,  y=y, batch=batch_no)
