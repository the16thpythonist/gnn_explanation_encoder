import typing as t

import numpy as np

import visual_graph_datasets.typing as tv
from kgcnn.data.utils import ragged_tensor_from_nested_numpy


def tensors_from_graphs(graphs: t.List[tv.GraphDict], masks: t.List[t.Tuple[np.ndarray, np.ndarray]]):

    node_attributes = [graph['node_attributes'] for graph in graphs]
    node_masks = [mask[0] for mask in masks]
    edge_indices = [graph['edge_indices'] for graph in graphs]
    edge_attributes = [graph['edge_attributes'] for graph in graphs]
    edge_masks = [mask[1] for mask in masks]
    
    return [
        ragged_tensor_from_nested_numpy(node_attributes),
        ragged_tensor_from_nested_numpy(node_masks),
        ragged_tensor_from_nested_numpy(edge_attributes),
        ragged_tensor_from_nested_numpy(edge_masks),
        ragged_tensor_from_nested_numpy(edge_indices),
    ]