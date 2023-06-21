"""
This string will be saved to the experiment's archive folder as the "experiment description"

CHANGELOG

0.1.0 - initial version
"""
import os
import pathlib
import random
import time
import datetime
import shutil

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.backends.backend_pdf import PdfPages

from scipy.spatial.distance import cosine
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from visual_graph_datasets.data import VisualGraphDatasetReader
from visual_graph_datasets.data import nx_from_graph
from visual_graph_datasets.data import MultiVisualGraphDatasetWriter
from visual_graph_datasets.data import MultiVisualGraphDatasetReader
from visual_graph_datasets.visualization.base import create_frameless_figure
from visual_graph_datasets.visualization.base import draw_image
from visual_graph_datasets.visualization.importances import plot_node_importances_border
from visual_graph_datasets.visualization.importances import plot_edge_importances_border
from graph_attention_student.keras import load_model
from graph_attention_student.models.megan import Megan
from graph_attention_student.data import tensors_from_graphs
from graph_attention_student.utils import array_normalize
from gnn_explanation_encoder.utils import binarize_node_mask

PATH = pathlib.Path(__file__).parent.absolute()

VISUAL_GRAPH_DATASET_PATH: str = '/media/ssd/.visual_graph_datasets/datasets/rb_dual_motifs'
MODEL_PATH: str = os.path.join(PATH, 'assets', 'models', 'rb_dual_motifs')
DATASET_NAME: str = 'rb_motifs__distances'
DOMAIN_VALUE_KEY: str = 'value'
NUM_CHANNELS: int = 2

PAIRWISE_DISTANCES_RATIO: float = 1e-2
FIDELITY_THRESHOLD = 0.1
FIDELITY_RANGE: float = 6.0
TIMEOUT = 10
IMAGE_WIDTH: int = 1000
IMAGE_HEIGHT: int = 1000


LOG_STEP = 200
__DEBUG__ = True
__TESTING__ = False


@Experiment(base_path=folder_path(__file__),
            namespace=file_namespace(__file__),
            glob=globals())
def experiment(e: Experiment):
    e.log('starting experiment...')
    
    if __TESTING__:
        e.log('TESTING MODE')
        e.PAIRWISE_DISTANCES_RATIO = 2e-6
        
    e['device_context'] = tf.device('cpu:0')
    e['device_context'].__enter__()
    
    e.log(f'loading dataset...')
    reader = VisualGraphDatasetReader(
        path=VISUAL_GRAPH_DATASET_PATH,
        logger=e.logger,
        log_step=1000,
    )
    index_data_map = reader.read()
    e.log(f'loaded data set with {len(index_data_map)} elements')
    
    module = reader.read_process()
    processing = module.processing    
    
    e.log('loading persistent model...')
    model: Megan = load_model(MODEL_PATH)
    num_channels = model.importance_channels

    e.log('making predictions for all the dataset elements...')
    graphs = []
    for index, data in index_data_map.items():
        graph = data['metadata']['graph']
        graphs.append(graph)
        
    predictions = model.predict_graphs(graphs)
    embeddings = model.embedd_graphs(graphs)
    leave_one_out = model.leave_one_out_deviations(graphs)
    e.log('processing those predictions...')
    for pred, embedding, one_out, graph in zip(predictions, embeddings, leave_one_out, graphs):
        out, ni, ei = pred
        graph['graph_prediction'] = out 
        graph['graph_fidelity'] = np.array([-one_out[0, 0], one_out[1, 0]])
        graph['graph_embedding'] = embedding
        
        graph['node_importances'] = array_normalize(ni)
        graph['edge_importances'] = array_normalize(ei)

        
    # @e.hook('calculate_distance', default=True)
    # def calculate_distance(e, graph_1, graph_2, channel_1, channel_2):
    #     mask_1 = binarize_node_mask(graph_1, graph_1['node_importances'][:, channel_1])
    #     mask_2 = binarize_node_mask(graph_2, graph_2['node_importances'][:, channel_2])
        
    #     _, sub_graph_1 = processing.extract(graph_1, mask_1)
    #     _, sub_graph_2 = processing.extract(graph_2, mask_2)
        
    #     nx_1 = nx_from_graph(sub_graph_1)
    #     nx_2 = nx_from_graph(sub_graph_2)
        
    #     structural_distance = nx.graph_edit_distance(
    #         nx_1, nx_2,
    #         # node_match=lambda a, b: np.isclose(a['node_attributes'], b['node_attributes']).all(),
    #         node_match=lambda a, b: processing.node_match(a['node_attributes'], b['node_attributes']),
    #         # edge_match=lambda a, b: np.isclose(a['edge_attributes'], b['edge_attributes']).all(),
    #         edge_match=lambda a, b: processing.edge_match(a['edge_attributes'], b['edge_attributes']),
    #         timeout=TIMEOUT,
    #     )
        
    #     if channel_1 == channel_2:
    #         semantic_distance = abs(graph_1['graph_fidelity'][channel_1] - graph_2['graph_fidelity'][channel_2])
    #     else:
    #         semantic_distance = abs(graph_1['graph_fidelity'][channel_1] + graph_2['graph_fidelity'][channel_2])
        
    #     return structural_distance, semantic_distance
    
    
    @e.hook('calculate_distance')
    def calculate_distance(e, graph_1, graph_2, channel_1, channel_2):
        
        structural_distance = cosine(graph_1['graph_embedding'][:, channel_1], graph_2['graph_embedding'][:, channel_2])
        
        if channel_1 == channel_2:
            semantic_distance = abs(graph_1['graph_fidelity'][channel_1] - graph_2['graph_fidelity'][channel_2]) / FIDELITY_RANGE
        else:
            semantic_distance = 0.5 * abs(graph_1['graph_fidelity'][channel_1] + graph_2['graph_fidelity'][channel_2]) / FIDELITY_RANGE
        
        return structural_distance, semantic_distance
    
    e.log('calculating pairwise distances...')
    num_elements = len(graphs)
    num_distances = int(1.5 * num_elements**2 * PAIRWISE_DISTANCES_RATIO)
    e.log(f'approx. for {num_distances} pairings')
    
    results = []
    c = 0
    start_time = time.time()
    indices = list(range(num_elements))
    for i in indices:
        for j in indices:
            graph_i = graphs[i]
            graph_j = graphs[j]
            
            for k in range(num_channels):
                for s in range(num_channels):
                    
                    # One important condition is that we absolutely do not want to work with any cases where the 
                    # fidelity is zero because that would be pointless.
                    if (graph_i['graph_fidelity'][k] < FIDELITY_THRESHOLD
                        or graph_j['graph_fidelity'][s] < FIDELITY_THRESHOLD):
                        continue
                    
                    # Also we only randomly choose a subset of all the pairwise combinations because otherwise 
                    # the whole thing would explode way too much.
                    if random.random() > PAIRWISE_DISTANCES_RATIO:
                        continue
                    
                    try:
                        structural_distance, semantic_distance = e.apply_hook(
                            'calculate_distance',
                            graph_1=graph_i,
                            graph_2=graph_j,
                            channel_1=k,
                            channel_2=s,
                        )
                    except (KeyError, AttributeError, RuntimeError) as exc:
                        e.log(f' * error: {exc.__class__}')
                        continue
                    
                    distance = structural_distance + semantic_distance
                    
                    results.append({
                        'index': c,
                        'target': [distance],
                        'structural_distance': structural_distance,
                        'semantic_distance': semantic_distance,
                        # 
                        'channel_1': k,
                        'data_1': {
                            **index_data_map[i],
                            'index': i,
                            'name': i,
                        },
                        'channel_2': s,
                        'data_2': {
                            **index_data_map[j],
                            'index': j,
                            'name': j,
                        },
                    })
                    
                    if c % LOG_STEP == 0:
                        elapsed_time = time.time() - start_time
                        time_per_element = elapsed_time / (c + 0.1)
                        remaining_time = time_per_element * (num_distances - c)
                        eta = datetime.datetime.now() + datetime.timedelta(seconds=remaining_time)
                        e.log(f' * ({c}/~{num_distances}) distances calculated'
                              f' - elapsed time: {elapsed_time:.1f}s'
                              f' - remaining time: {remaining_time:.1f}s'
                              f' - eta: {eta:%a %d %H:%M}'
                              f' - str_dist: {structural_distance:.2f}'
                              f' - sem_dist: {semantic_distance:.2f}')
                    
                    c += 1
                    
    num_results = len(results)
                    
    e.log('writing the results to the disk...')
    dataset_path = os.path.join(e.path, DATASET_NAME)
    os.mkdir(dataset_path)
    e['dataset_path'] = dataset_path
    
    start_time = time.time()
    writer = MultiVisualGraphDatasetWriter(
        dataset_path, 
        chunk_size=1_000,
        file_chunking=True,    
    )
    for c, data in enumerate(results):
                
        # if not writer.individual_element_exists(data['data_1']['index']):
        #     data_1 = data['data_1']
        #     figure, node_positions = processing.visualize_as_figure(
        #         value=data_1['metadata'][DOMAIN_VALUE_KEY],
        #         width=IMAGE_WIDTH,
        #         height=IMAGE_HEIGHT,
        #     )
        #     data_1['metadata']['graph']['node_positions'] = node_positions
        #     data_1['figure'] = figure
        
        # if not writer.individual_element_exists(data['data_2']['index']):
        #     data_2 = data['data_2']
        #     figure, node_positions = processing.visualize_as_figure(
        #         value=data_2['metadata'][DOMAIN_VALUE_KEY],
        #         width=IMAGE_WIDTH,
        #         height=IMAGE_HEIGHT,
        #     )
        #     data_2['metadata']['graph']['node_positions'] = node_positions
        #     data_2['figure'] = figure              

        writer.write(
            name=data['index'],
            metadata=data,
            data_dicts=[
                data['data_1'],
                data['data_2']
            ]
        )
        
        if c % LOG_STEP == 0:
            elapsed_time = time.time() - start_time
            time_per_element = elapsed_time / (c + 0.1)
            remaining_time = time_per_element * (num_results - c)
            eta = datetime.datetime.now() + datetime.timedelta(seconds=remaining_time)
            e.log(f' * ({c}/{num_results}) written'
                  f' - elapsed time: {elapsed_time:.1f}s'
                  f' - remaining time: {remaining_time/3600:.1f}hrs'
                  f' - time per element: {time_per_element:.2f}s'
                  f' - eta: {eta:%a %d %H:%M}'
                  f' - open figs: {len(plt.get_fignums())}')
            
            plt.close('all')
            
    writer.close()
    shutil.copy(
        os.path.join(VISUAL_GRAPH_DATASET_PATH, 'process.py'),
        os.path.join(dataset_path, 'process.py')
    )


@experiment.analysis
def analysis(e: Experiment):
    e.log('starting analysis...')
    
    e.log('reading the dataset...')
    reader = MultiVisualGraphDatasetReader(
        e['dataset_path'],
        logger=e.logger,
        log_step=1000,
    )
    index_data_map = reader.read()
    index_element_map = reader.index_element_map
    
    # Here we visualize a few example predictions most importantly the predicted node and edge importances
    # this is mostly for debugging purposes to see (1) if the importances make sense and (2) to see if the 
    # importances are rendered to the images correctly aka if the node_positions are correct, which can be 
    # bit finnicky
    e.log('visualizing a few example predictions...')
    num_examples = 10
    example_indices = random.sample(list(index_element_map.keys()), k=num_examples)
    example_graphs = [index_element_map[index]['metadata']['graph'] for index in example_indices]

    pdf_path = os.path.join(e.path, 'examples.pdf')
    with PdfPages(pdf_path) as pdf:
        
        for index in example_indices:
            data = index_element_map[index]
            graph = data['metadata']['graph']
            node_positions = graph['node_positions']
            ni, ei = graph['node_importances'], graph['edge_importances']
            
            fig, rows = plt.subplots(
                ncols=NUM_CHANNELS,
                nrows=1,
                figsize=(NUM_CHANNELS * 8, 8),
                squeeze=False,
            )
            
            for channel_index in range(NUM_CHANNELS):
                ax = rows[0][channel_index]
                ax.set_title(f'channel {channel_index}')
                draw_image(ax, data['image_path'])
                plot_node_importances_border(ax, graph, node_positions, ni[:, channel_index])
                plot_edge_importances_border(ax, graph, node_positions, ei[:, channel_index])
        
            pdf.savefig(fig)
            plt.close(fig)
        

experiment.run_if_main()