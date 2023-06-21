"""
This string will be saved to the experiment's archive folder as the "experiment description"

CHANGELOG

0.1.0 - initial version
"""
import os
import pathlib
import random
import typing as t

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as ks
from matplotlib.backends.backend_pdf import PdfPages

import visual_graph_datasets.typing as tv
from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from visual_graph_datasets.data import MultiVisualGraphDatasetReader
from visual_graph_datasets.processing.colors import ColorProcessing
from visual_graph_datasets.visualization.base import layout_node_positions, draw_image
from visual_graph_datasets.visualization.colors import visualize_color_graph, colors_layout
from visual_graph_datasets.visualization.importances import plot_node_importances_border
from visual_graph_datasets.visualization.importances import plot_edge_importances_border
from graph_attention_student.keras import load_model
from gnn_explanation_encoder.models import AttributionalGraphImplicitEncoder
from gnn_explanation_encoder.data import tensors_from_graphs

PATH = pathlib.Path(__file__).parent.absolute()

DUAL_VISUAL_GRAPH_DATASET_PATH = os.path.join(PATH, 'results', 'generate_pairwise_explanation_distance_dataset', 'debug', 'rb_motifs__distances')
MODEL_PATH: str = os.path.join(PATH, 'assets', 'models', 'rb_dual_motifs')
DOMAIN_VALUE_KEY: str = 'value'

DIMENSIONS = 256
EPOCHS = 2
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
OPTIMIZER_CB = lambda: ks.optimizers.experimental.AdamW(learning_rate=LEARNING_RATE)

MIN_DIST = 0.01
NUM_NEIGHBORS = 10

CHANNEL_COLOR_MAP = {
    0: 'lightskyblue',
    1: 'lightsalmon',
}

VALUES = (
    {
        'name': 'blue_star', 
        'value': 'Y(BH)(BH)(BH)H',
        'channel': 0
    },
    {
        'name': 'blue_ring',
        'value': 'G-1(H)B(H)B-1(H)',
        'channel': 0,
    },
    {
        'name': 'blue_star+blue_ring',
        'value': 'G-1BB-1HHHHY(B)(B)(B)',
        'channel': 0,
    },
    {
        'name': 'red_star',
        'value': 'Y(RH)(RH)(RH)H',
        'channel': 1,
    },
    {
        'name': 'red_ring',
        'value': 'G-1(H)R(H)R-1(H)',
        'channel': 1,  
    },
    {
        'name': 'red_star+red_ring',
        'value': 'G-1RR-1HHHHY(R)(R)(R)',
        'channel': 1,
    },
)
NUM_EXAMPLES: int = 15
NUM_EXAMPLE_NEIGHBORS: int = 10

__DEBUG__ = True
__TESTING__ = False


@Experiment(base_path=folder_path(__file__),
            namespace=file_namespace(__file__),
            glob=globals())
def experiment(e: Experiment):
    e.log('starting experiment...')
    tf.device('cpu:0').__enter__()
    
    if __TESTING__:
        e.log('TESTING MODE')
        e.EPOCHS = 1

    e.log(f'loading dataset...')
    reader = MultiVisualGraphDatasetReader(
        path=DUAL_VISUAL_GRAPH_DATASET_PATH,
        logger=e.logger,
        log_step=1000,    
    )
    index_data_map = reader.read()
    index_element_map = reader.index_element_map
    e.log(f'loaded {len(index_data_map)} pairwise elements')
    e.log(f'based on {len(index_element_map)} individual graph elements')
    
    module = reader.read_process()
    processing = module.processing
    predictor = load_model(MODEL_PATH)

    e.hook('get_target_value', default=True)
    def get_target_value(e, semantic_distance, structural_distance):
        return max(semantic_distance, structural_distance)

    e.log('converting input to tensors...')
    graphs_1, graphs_2 = [], []
    masks_1, masks_2 = [], []
    ys = []
    for index, data in index_data_map.items():
        graph_1 = data['metadata']['data_1']['metadata']['graph']
        channel_1 = data['metadata']['channel_1']
        graphs_1.append(graph_1)
        masks_1.append([
            graph_1['node_importances'][:, channel_1],
            graph_1['edge_importances'][:, channel_1],
        ])
        
        graph_2 = data['metadata']['data_2']['metadata']['graph']
        channel_2 = data['metadata']['channel_2']
        graphs_2.append(graph_2)
        masks_2.append([
            graph_2['node_importances'][:, channel_2],
            graph_2['edge_importances'][:, channel_2]
        ])
        
        #target = 0.1 * (1 * data['metadata']['structural_distance'] + 10 * data['metadata']['semantic_distance'])
        sem_dist = data['metadata']['semantic_distance']
        str_dist = data['metadata']['structural_distance']
        
        target = e.apply_hook(
            'get_target_value',
            semantic_distance=sem_dist,
            structural_distance=str_dist,    
        )
        ys.append(target)
    
    x_1 = tensors_from_graphs(graphs_1, masks_1)
    x_2 = tensors_from_graphs(graphs_2, masks_2)
    y = np.array(ys)
        
    e.log('setting up the model...')
    encoder = AttributionalGraphImplicitEncoder(
        #distance_func=lambda a, b: tf.sqrt(tf.reduce_sum(tf.square(a - b), axis=-1)),
        distance_func=lambda a, b: tf.reduce_sum(tf.square(a - b), axis=-1),
        #distance_func=lambda a, b: tf.reduce_sum(tf.abs(a - b), axis=-1),
        conv_units=[128, 128, 128],
        dense_units=[DIMENSIONS, DIMENSIONS],
        size=DIMENSIONS,
    )
    encoder.compile(
        optimizer=OPTIMIZER_CB(),
        loss=ks.losses.MeanSquaredError(),
    )

    #dataset = tf.data.Dataset.from_tensor_slices(([*x_1, *x_2], y))

    e.log('starting model training...')
    results = encoder.fit(
        [*x_1, *x_2], y, 
        #dataset,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )
    
    e.log('calculating all encodings...')
    graphs = []
    for graph_index, data in index_element_map.items():
        graph = data['metadata']['graph']
        graph['graph_index'] = graph_index
        graph['graph_value'] = data['metadata'][DOMAIN_VALUE_KEY]
        graphs.append(graph)

    encoded = []
    indices_encoded = []
    for channel_index in range(2):
        _indices = [index
                    for index, graph in enumerate(graphs)
                    if graph['graph_fidelity'][channel_index] > 0.2]
        _graphs = [graphs[index] for index in _indices]
        _masks = [[graphs[index]['node_importances'][:, channel_index], 
                   graphs[index]['edge_importances'][:, channel_index]]
                   for index in _indices]
        
        indices_encoded += _indices
        x = tensors_from_graphs(_graphs, _masks)
        encoded.append(encoder(x).numpy())
        
    e.log('fitting UMAP transformation...')
    umap = UMAP(
        n_components=2, 
        n_neighbors=NUM_NEIGHBORS, 
        min_dist=MIN_DIST,
        random_state=1,
        metric='euclidean',
        # metric='cosine'
    )
    umap.fit(np.concatenate(encoded, axis=0))
    
    e.log('plotting clustering results...')
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
    
    embedded_all = []
    for channel_index, enc in enumerate(encoded):
        embedded = umap.transform(enc)
        ax.scatter(
            embedded[:, 0], embedded[:, 1],
            color=CHANNEL_COLOR_MAP[channel_index],
            s=5,
        )
        
        embedded_all.append(embedded)
        
    embedded_all = np.concatenate(embedded_all, axis=0)
    e.log(f'shape of the complete embedding array: {embedded_all.shape}')

    # ~ mapping the special values into the encoding space
    for data in VALUES:
        graph = processing.process(data['value'])
        out, ni, ei = predictor.predict_graph(graph)
        
        x = tensors_from_graphs([graph], [[ni[:, data['channel']], ei[:, data['channel']]]])
        embedded = umap.transform(encoder(x).numpy())
        ax.scatter(
            embedded[0, 0], embedded[0, 1],
            color='black'
        )
        ax.annotate(
            data['name'],
            [embedded[0, 0], embedded[0, 1]]
        )
        
    # ~ Randomly choosing the examples
    e.log('choosing random example clusters...')
    example_seed_indices = random.sample(indices_encoded, k=NUM_EXAMPLES)
    example_graphs_list: t.List[t.List[tv.GraphDict]] = []
    for example_index, graph_index in enumerate(example_seed_indices):
        graph = graphs[graph_index]
        channel = np.argmax(graph['graph_fidelity'])
        ni, ei = graph['node_importances'], graph['edge_importances']
        
        x = tensors_from_graphs([graph], [[ni[:, channel], ei[:, channel]]])
        embedded = umap.transform(encoder(x).numpy())
        ax.scatter(
            embedded[0, 0], embedded[0, 1],
            color='black',
            marker='x'
        )
        ax.annotate(
            f'({example_index})',
            [embedded[0, 0], embedded[0, 1]]
        )
        
        # finding the nearest neighbors
        distances = euclidean_distances(embedded, embedded_all)[0]
        indices = np.argsort(distances)
        neighbor_indices = indices[:NUM_EXAMPLE_NEIGHBORS]
        neighbor_graphs = [graph] + [graphs[indices_encoded[int(i)]] for i in neighbor_indices]
        example_graphs_list.append(neighbor_graphs)

    ax.set_title(f'{DIMENSIONS} dimensional graph embeddings reduced to 2 dimensions with UMAP\n'
                 f'{len(index_data_map)} pairwise distances used for encoder training')
    fig.savefig(os.path.join(e.path, 'encoded.pdf'))
    
    e.log('plotting the examples...')
    pdf_path = os.path.join(e.path, 'examples.pdf')
    with PdfPages(pdf_path) as pdf:
        for example_index, example_graphs in enumerate(example_graphs_list):
            fig, rows = plt.subplots(
                ncols=(NUM_EXAMPLE_NEIGHBORS + 1),
                nrows=1,
                figsize=((NUM_EXAMPLE_NEIGHBORS + 1) * 10, 10),
                squeeze=False
            )
            fig.suptitle(f'neighbors in example cluster ({example_index})')
            
            channel = np.argmax(example_graphs[0]['graph_fidelity'])
            
            for c, graph in enumerate(example_graphs):
                ax = rows[0][c]

                graph_index = graph['graph_index']
                ni, ei = graph['node_importances'], graph['edge_importances']
                node_positions = graph['node_positions']
                
                draw_image(ax, index_element_map[graph_index]['image_path'])
                plot_node_importances_border(ax, graph, node_positions, ni[:, channel])
                plot_edge_importances_border(ax, graph, node_positions, ei[:, channel])
                ax.set_title(f'fidelity ch.{channel} - {graph["graph_fidelity"][channel]:.2f}')
            
            pdf.savefig(fig)
            plt.close(fig)
            
    e.log('saving the model...')
    model_path = os.path.join(e.path, 'model')
    os.mkdir(model_path)
    encoder.save(model_path)


@experiment.analysis
def analysis(e: Experiment):
    e.log('starting analysis...')


experiment.run_if_main()