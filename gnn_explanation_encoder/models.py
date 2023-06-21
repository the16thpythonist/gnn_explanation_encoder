import typing as t

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
import visual_graph_datasets.typing as tv
from kgcnn.layers.conv.gcn_conv import GCN
from kgcnn.layers.conv.gin_conv import GINE
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.layers.modules import DenseEmbedding
from kgcnn.layers.modules import LazyConcatenate
from kgcnn.layers.modules import DropoutEmbedding
from kgcnn.layers.conv.gat_conv import AttentionHeadGATV2
from gnn_explanation_encoder.data import tensors_from_graphs


class ImplicitEncoderBase(ks.models.Model):
    
    def __init__(self,
                 distance_func: t.Callable,
                 **kwargs,
                 ):
        ks.models.Model.__init__(self, **kwargs)
        self.distance_func = distance_func
    
    def train_step(self, data):
        # Extract the input and target from the data tuple
        x, y = data
        
        half_index = int(len(x) / 2)
        x_1 = x[:half_index]
        x_2 = x[half_index:]

        with tf.GradientTape() as tape:
            # Forward passes
            y_pred_1 = self(x_1, training=True)
            y_pred_2 = self(x_2, training=True)
            
            y_dist = self.distance_func(y_pred_1, y_pred_2)
            
            # Compute the loss
            loss = self.compiled_loss(y, y_dist, regularization_losses=self.losses)

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # Update metrics (optional)
        self.compiled_metrics.update_state(y, y_dist)

        # Return a dictionary mapping metric names to current values
        return {m.name: m.result() for m in self.metrics}



class DenseImplicitEncoder(ImplicitEncoderBase):
    
    def __init__(self,
                 distance_func: t.Callable,
                 units: t.List[int],
                 size: int,
                 activation: str = 'relu',
                 **kwargs,
                 ):
        ks.models.Model.__init__(self, distance_func, **kwargs)
        self.units = units
        self.size = size
        
        self.dense_layers = []
        for k in self.units:
            lay = ks.layers.Dense(
                units=k,
                activation=activation
            )
            self.dense_layers.append(lay)
            
        self.lay_out = ks.layers.Dense(units=size, activation='linear')
    
    
    def call(self, inputs):
        # Define the forward pass through the model
        x = inputs[0]
        for lay in self.dense_layers:
            x = lay(x)
            
        out = self.lay_out(x)
        return out
    
    
    
class GcnImplicitEncoder(ImplicitEncoderBase):
    
    def __init__(self,
                 distance_func: t.Callable,
                 units: t.List[int],
                 size: int,
                 activation: str = 'kgcnn>leaky_relu',
                 pooling_method: str = 'sum',
                 **kwargs
                 ):
        ImplicitEncoderBase.__init__(self, distance_func, **kwargs)
        self.units = units
        self.size = size
        
        self.conv_layers = []
        for k in self.units:
            lay = GCN(
                units=k,
                pooling_method='sum',
                activation=activation,
            )
            self.conv_layers.append(lay)
            
        self.lay_pooling = PoolingNodes(pooling_method=pooling_method)
        self.lay_out = DenseEmbedding(units=size, activation='linear')
    
    def call(self, inputs):
        node_input, edge_input, edge_indices = inputs
        
        x = node_input
        for lay in self.conv_layers:
            x = lay([x, edge_input, edge_indices])
            
        out = self.lay_pooling(x)
        out = self.lay_out(out)
        return out
    
    

class ExternalAttentionHeadGATV2(AttentionHeadGATV2):
    
    def call(self, inputs, **kwargs):
        node, edge, edge_mask, edge_index = inputs

        w_n = self.lay_linear_trafo(node, **kwargs)
        n_in = self.lay_gather_in([node, edge_index], **kwargs)
        n_out = self.lay_gather_out([node, edge_index], **kwargs)
        wn_out = self.lay_gather_out([w_n, edge_index], **kwargs)
        if self.use_edge_features:
            e_ij = self.lay_concat([n_in, n_out, edge], **kwargs)
        else:
            e_ij = self.lay_concat([n_in, n_out], **kwargs)
            
        # a_ij = self.lay_alpha_activation(e_ij, **kwargs)
        # a_ij = self.lay_alpha(a_ij, **kwargs)
        # h_i = self.lay_pool_attention([node, wn_out, a_ij, edge_index], **kwargs)
        h_i = self.lay_pool_attention([node, wn_out, edge_mask, edge_index], **kwargs)

        if self.use_final_activation:
            h_i = self.lay_final_activ(h_i, **kwargs)
        return h_i
    
    
    
class AttributionalGraphImplicitEncoder(ImplicitEncoderBase):
    
    def __init__(self,
                 distance_func: t.Callable,
                 conv_units: t.List[int],
                 size: int,
                 dense_units: t.List[int] = [],
                 activation: str = 'kgcnn>leaky_relu',
                 pooling_method: str = 'sum',
                 **kwargs
                 ):
        ImplicitEncoderBase.__init__(self, distance_func, **kwargs)
        self.conv_units = conv_units
        self.dense_units = dense_units
        self.size = size
        self.activation = activation
        self.pooling_method = pooling_method
        
        self.conv_layers = []
        for k in self.conv_units:
            # lay = AttentionHeadGATV2(
            #     units=k,
            #     activation=activation,
            #     use_edge_features=True,
            #     use_final_activation=True,
            #     has_self_loops=False,
            # )
            lay = ExternalAttentionHeadGATV2(
                units=k,
                activation=activation,
                use_edge_features=True,
                use_final_activation=True,
                has_self_loops=False,
            )
            self.conv_layers.append(lay)
            
        self.lay_dropout = DropoutEmbedding(rate=0.0)
        
        self.lay_concat = LazyConcatenate(axis=-1)    
        self.lay_pooling = PoolingNodes(pooling_method=pooling_method)
        self.dense_layers = []
        for k in self.dense_units:
            lay = DenseEmbedding(
                units=k,
                activation=activation,
            )
            self.dense_layers.append(lay)
        
        self.lay_out = DenseEmbedding(units=size, activation='linear')
        
    def get_config(self):
        config = super(AttributionalGraphImplicitEncoder, self).get_config()
        config.update({
            'distance_func': self.distance_func,
            'conv_units': self.conv_units,
            'dense_units': self.dense_units,
            'size': self.size,
            'activation': self.activation,
            'pooling_method': self.pooling_method,
        })
        return config
        
    def call(self, inputs, training=True, **kwargs):
        node_input, node_mask, edge_input, edge_mask, edge_indices = inputs
        node_mask = tf.expand_dims(node_mask, axis=-1)
        edge_mask = tf.expand_dims(edge_mask, axis=-1)
        
        # x = node_input * node_mask
        x_list = []
        x = node_input
        for lay in self.conv_layers:
            x = lay([x, edge_input, edge_mask, edge_indices])
            #x = lay([x, edge_input, edge_indices])
            if training:
                x = self.lay_dropout(x)
            
            x_list.append(x)
            
        x = self.lay_concat(x_list)
        x = x * node_mask
        out = self.lay_pooling(x)
        for lay in self.dense_layers:
            out = lay(out)
            
        out = self.lay_out(out)
        
        return out
    
    def encode_graphs(self, 
                      graphs: t.List[tv.GraphDict],
                      masks: t.List[np.ndarray],
                      ) -> t.List[np.ndarray]:
        x = tensors_from_graphs(graphs, masks)
        encoded = self(x)
        return encoded.numpy()
        
    def encode_graph(self,
                     graph: tv.GraphDict,
                     mask: np.ndarray
                     ) -> np.ndarray:
        return self.encode_graphs([graph], [mask])[0]
    
    def predict_pairwise_distances(self,
                                   graphs_1: t.List[tv.GraphDict],
                                   graphs_2: t.List[tv.GraphDict],
                                   masks_1: t.List[t.Tuple[np.ndarray, np.ndarray]],
                                   masks_2: t.List[t.Tuple[np.ndarray, np.ndarray]],
                                   ) -> float:
        
        x_1 = tensors_from_graphs(graphs_1, masks_1)
        x_2 = tensors_from_graphs(graphs_2, masks_2)
        
        encoded_1 = self(x_1)
        encoded_2 = self(x_2)
        
        distances = self.distance_func(encoded_1, encoded_2)
        return distances.numpy()
        
        
CUSTOM_OBJECTS = {
    'AttributionalGraphImplicitEncoder': AttributionalGraphImplicitEncoder
}


def load_model(path: str):
    with ks.utils.custom_object_scope(CUSTOM_OBJECTS):
        return ks.models.load_model(path)