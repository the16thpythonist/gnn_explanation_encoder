import os
import tempfile

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks

from visual_graph_datasets.testing import mock_graph_dict
from gnn_explanation_encoder.utils import graph_node_mask, graph_edge_mask
from gnn_explanation_encoder.models import AttributionalGraphImplicitEncoder
from gnn_explanation_encoder.models import load_model


def test_graph_attributional_implicit_encoder_basically_works():

    # ~ Constructing the model
    size = 16
    encoder = AttributionalGraphImplicitEncoder(
        distance_func=lambda a, b: tf.reduce_sum(tf.square(a - b), axis=-1),
        conv_units=[16, 16, 16],
        dense_units=[16, 16],
        size=size,
    )
    assert isinstance(encoder, AttributionalGraphImplicitEncoder)
    assert isinstance(encoder, ks.models.Model)
    
    # ~ Using the model to encode graphs
    num_elements = 10
    graphs = [mock_graph_dict(10) for _ in range(num_elements)]
    masks = [(graph_node_mask(graph), graph_edge_mask(graph))
             for graph in graphs]
    encoded = encoder.encode_graphs(graphs, masks)
    # We know that the the result should be as many encoded vectors as the graphs used as input and 
    # each vector should have as many elements as specified in the encoder constructor.
    assert isinstance(encoded, np.ndarray)
    assert encoded.shape == (num_elements, size)
    
    # ~ Saving the model
    with tempfile.TemporaryDirectory() as path:
        encoder.save(path)
        files = os.listdir(path)
        assert len(files) != 0
    
    # ~ loading the model
    
        encoder_loaded = load_model(path)
        encoded_loaded = encoder_loaded.encode_graphs(graphs, masks)
        
        # The ultimate test ist that the two outputs have to be exactly the same
        assert np.isclose(encoded, encoded_loaded).all()
    
    
    