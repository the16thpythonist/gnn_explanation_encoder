"""
This string will be saved to the experiment's archive folder as the "experiment description"

CHANGELOG

0.1.0 - initial version
"""
import os
import pathlib
import random

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as ks 

from sklearn.metrics import r2_score
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from visual_graph_datasets.data import DualVisualGraphDatasetReader
from graph_attention_student.visualization import plot_regression_fit
from gnn_explanation_encoder.models import AttributionalGraphImplicitEncoder
from gnn_explanation_encoder.data import tensors_from_graphs

PATH = pathlib.Path(__file__).parent.absolute()

DUAL_VISUAL_GRAPH_DATASET_PATH = os.path.join(PATH, 'results', 'generate_pairwise_explanation_distance_dataset', 'debug', 'rb_motifs__distances')
NUM_TEST = 3000

DIMENSIONS = 128
EPOCHS = 5
BATCH_SIZE = 8
LEARNING_RATE = 1e-4

__DEBUG__ = True
__TESTING__ = False


@Experiment(base_path=folder_path(__file__),
            namespace=file_namespace(__file__),
            glob=globals())
def experiment(e: Experiment):
    e.log('starting experiment...')
    e['device_context'] = tf.device('cpu:0')
    e['device_context'].__enter__()

    if __TESTING__:
        e.log('TESTING MODE')
        e.EPOCHS = 1

    e.log('loading dataset of pairwise distance annotations...')
    reader = DualVisualGraphDatasetReader(
        path=DUAL_VISUAL_GRAPH_DATASET_PATH,
        logger=e.logger,
        log_step=1000,
    )
    index_data_map = reader.read()
    indices = list(range(len(index_data_map)))
    max_index = max(indices) + 1
    e.log(f'loaded {len(index_data_map)} elements')
    
    e.log(f'preparing dataset for training...')
    graphs_1 = [None for _ in range(max_index)]
    graphs_2 = [None for _ in range(max_index)]
    masks_1 = [None for _ in range(max_index)]
    masks_2 = [None for _ in range(max_index)]
    ys = [None for _ in range(max_index)]
    for index, data in index_data_map.items():
        
        channel_1 = data['metadata']['channel_1']
        graph_1 = data['metadata']['data_1']['metadata']['graph']
        graphs_1[index] = graph_1
        masks_1[index] = [
            graph_1['node_importances'][:, channel_1],
            graph_1['edge_importances'][:, channel_1]
        ]
        
        channel_2 = data['metadata']['channel_2']
        graph_2 = data['metadata']['data_2']['metadata']['graph']
        graphs_2[index] = graph_2
        masks_2[index] = [
            graph_2['node_importances'][:, channel_2],
            graph_2['edge_importances'][:, channel_2]
        ]
    
        ys[index] = data['metadata']['semantic_distance']
    
    test_indices = random.sample(indices, k=NUM_TEST)
    train_indices = list(set(indices).difference(set(test_indices)))
    
    x_train = [
        tensors_from_graphs([graphs_1[i] for i in train_indices], [masks_1[i] for i in train_indices]),
        tensors_from_graphs([graphs_2[i] for i in train_indices], [masks_2[i] for i in train_indices]),
    ]
    y_train = np.array([ys[i] for i in train_indices])
    
    x_test = [
        tensors_from_graphs([graphs_1[i] for i in test_indices], [masks_1[i] for i in test_indices]),
        tensors_from_graphs([graphs_2[i] for i in test_indices], [masks_2[i] for i in test_indices]),
    ]
    y_test = np.array([ys[i] for i in test_indices])
    
    e.log(f'setting up the model...')
    model = AttributionalGraphImplicitEncoder(
        distance_func=lambda a, b: tf.sqrt(tf.reduce_sum(tf.square(a - b), axis=-1)),
        conv_units=[DIMENSIONS, DIMENSIONS, DIMENSIONS],
        dense_units=[DIMENSIONS, DIMENSIONS],
        size=DIMENSIONS
    )
    model.compile(
        optimizer=ks.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=ks.losses.MeanSquaredError(),
        metrics=[
            ks.metrics.MeanSquaredError(),
            ks.metrics.MeanAbsoluteError(),
        ]
    )
    
    e.log('starting model training...')
    res = model.fit(
        [*x_train[0], *x_train[1]], y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )
    
    e.log('evaluating test results...')
    y_pred_train = model.distance_func(model(x_train[0]), model(x_train[1])).numpy()
    r2_train = r2_score(y_train, y_pred_train)
    y_pred_test = model.distance_func(model(x_test[0]), model(x_test[1])).numpy()
    r2_test = r2_score(y_test, y_pred_test)
    e.log(f'test performance'
          f' - r2_train {r2_train:.2f}'
          f' - r2_test: {r2_test:.2f}')
    
    fig, (ax_train, ax_test) = plt.subplots(ncols=2, nrows=1, figsize=(16, 8))
    ax_train.set_title(f'train r2: {r2_train:.2f}')
    plot_regression_fit(
        ax=ax_train,
        values_true=y_train,
        values_pred=y_pred_train,
    )
    ax_test.set_title(f'test r2: {r2_test:.2f}')
    plot_regression_fit(
        ax=ax_test,
        values_true=y_test,
        values_pred=y_pred_test,
    )
    fig.savefig(os.path.join(e.path, 'regression.pdf'))
    


@experiment.analysis
def analysis(e: Experiment):
    e.log('starting analysis...')


experiment.run_if_main()