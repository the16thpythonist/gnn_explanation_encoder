"""
This string will be saved to the experiment's archive folder as the "experiment description"

CHANGELOG

0.1.0 - initial version
"""
import os
import time
import random
import pathlib
import itertools
from collections import defaultdict

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as ks

from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from visual_graph_datasets.generation.colors import make_star_motif, make_ring_motif
from visual_graph_datasets.generation.colors import RED, BLUE, GREEN
from visual_graph_datasets.processing.colors import ColorProcessing
from visual_graph_datasets.data import nx_from_graph
from vgd_counterfactuals.generate.colors import get_neighborhood
from graph_attention_student.data import tensors_from_graphs
from gnn_explanation_encoder.models import GcnImplicitEncoder


PATH = pathlib.Path(__file__).parent.absolute()
__DEBUG__ = True

SEED_GRAPHS = [
    make_ring_motif(GREEN, RED, k=5),
    make_star_motif(GREEN, BLUE, k=5),
    make_star_motif(GREEN, RED, k=5)
]

RATIO_TRAIN = 0.2
NUM_TEST = 30
EPOCHS = 25
BATCH_SIZE = 16

LOG_STEP = 5


@Experiment(base_path=folder_path(__file__),
            namespace=file_namespace(__file__),
            glob=globals())
def experiment(e: Experiment):
    e.log('starting experiment...')

    processing = ColorProcessing()
    
    e.log('creating the graph clusters...')
    # In the first step, the graph clusters have to be created from the seed graphs. We can use some already existing functionality 
    # for this from the counterfactuals library where there is a function that can be used to create the entire 1-edit neighborhood 
    # of a given color graph. But for that we need the graphs in their COGILES representation
    seed_elements_map = defaultdict(list)
    for seed_index, graph in enumerate(SEED_GRAPHS):
        cogiles = processing.unprocess(graph)
        results = get_neighborhood(cogiles)
        for data in results:
            _graph = processing.process(data['value'])
            seed_elements_map[seed_index].append(_graph)
            
        # Visualizing the seed graphs
        fig, _ = processing.visualize_as_figure(cogiles, width=400, height=400)
        fig.savefig(os.path.join(e.path, f'seed_{seed_index}.png'))

    elements = list(itertools.chain.from_iterable(seed_elements_map.values()))
    num_elements = len(elements)
    e.log(f'created a total of {num_elements} graphs from {len(SEED_GRAPHS)} clusters')

    indices = list(range(num_elements))
    test_indices = random.sample(indices, k=NUM_TEST)
    train_indices = list(set(indices).difference(set(test_indices)))
    e.log(f'using {len(train_indices)} training graphs and {len(test_indices)} test graphs')
    
    @e.hook('calculate_distance', default=True)
    def calculate_distance(e, graph_1, graph_2):
        g_1 = nx_from_graph(graph_1)
        g_2 = nx_from_graph(graph_2)
        
        distance = nx.graph_edit_distance(
            g_1, g_2,
            node_match=lambda a, b: np.isclose(a['node_attributes'], b['node_attributes']).all(),
            edge_match=lambda a, b: np.isclose(a['edge_attributes'], b['edge_attributes']).all(),
            timeout=10,
        )
        return distance
    
    
    e.log('calculating pairwise distances...')
    start_time = time.time()
    graphs_1 = []
    graphs_2 = []
    ys = []
    c = 0
    for i in train_indices:
        for j in train_indices:
            if random.random() < RATIO_TRAIN:
                graph_i = elements[i]
                graph_j = elements[j]
                graphs_1.append(graph_i)
                graphs_2.append(graph_j)
                
                distance = e.apply_hook('calculate_distance', graph_1=graph_i, graph_2=graph_j)
                ys.append(distance)
                
                if c % LOG_STEP == 0:
                    e.log(f' * {c} distances calculated'
                        f' - distance: {distance}'
                        f' - elapsed_time: {time.time() - start_time:.1f}s')
                c += 1

    x_1 = tensors_from_graphs(graphs_1)
    x_2 = tensors_from_graphs(graphs_2)
    y = np.array(ys)
    e.log(f'created {len(y)}(~{RATIO_TRAIN*100:.1f}%) pairwise distances for model training')

    e.log(f'setting up model...')
    model = GcnImplicitEncoder(
        distance_func=lambda a, b: tf.reduce_sum(tf.square(a - b), axis=-1),
        units=[32, 32, 32],
        size=2,
    )
    model.compile(
        loss=ks.losses.MeanSquaredError(),
        metrics=[
            ks.metrics.MeanSquaredError()
        ]
    )
    
    e.log('starting model training...')
    model.fit(
        [*x_1, *x_2], y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )
    
    test_graphs = [elements[i] for i in test_indices]
    x_test = tensors_from_graphs(test_graphs)
    encoded = model(x_test).numpy()
    
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
    ax.set_title(f'2-dimensional latent graph embeddings of test graphs\n'
                 f'using {RATIO_TRAIN*100:.1f}% of pairwise distances for training')
    ax.scatter(
        encoded[:, 0], encoded[:, 1],
        color='lightgray',
        linewidths=1,
        edgecolors='black',
        s=150,
    )
    
    y_min, y_max = ax.get_ylim()
    y_margin = 0.2 * (y_max - y_min)
    ax.set_ylim([y_min - y_margin, y_max + y_margin])
    x_min, x_max = ax.get_xlim()
    x_margin = 0.2 * (x_max - x_min)
    ax.set_xlim([x_min - x_margin, x_max + x_margin])
    
    fig.savefig(os.path.join(e.path, 'encoded.pdf'))
    

@experiment.analysis
def analysis(e: Experiment):
    e.log('starting analysis...')


experiment.run_if_main()
