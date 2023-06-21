"""
This is a sub experiment derived from ...

CHANGELOG

0.1.0 - initial version
"""
import os
import pathlib

import numpy as np
import tensorflow.keras as ks
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

PATH = pathlib.Path(__file__).parent.absolute()

DUAL_VISUAL_GRAPH_DATASET_PATH = os.path.join(PATH, 'results', 'generate_pairwise_explanation_distance_dataset_aqsoldb', 'debug', 'aqsoldb__distances')
MODEL_PATH: str = os.path.join(PATH, 'assets', 'models', 'aqsoldb')
DOMAIN_VALUE_KEY: str = 'smiles'

DIMENSIONS = 32
EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
OPTIMIZER_CB = lambda: ks.optimizers.Adam(
    learning_rate=LEARNING_RATE,
    clipvalue=0.1,
)

MIN_DIST = 0.0
NUM_NEIGHBORS = 10

CHANNEL_COLOR_MAP = {
    0: 'lightskyblue',
    1: 'lightsalmon',
}

VALUES = (
    {
        'name': '1-ring', 
        'value': 'C1=CC=CC=C1C',
        'channel': 0,
    },
    {
        'name': '2-ring',
        'value': 'C1=CC=CC=C1CCC1=CC=CC=C1',
        'channel': 0,
    },
    {
        'name': 'chain',
        'value': 'CCCC(CCC)CC',
        'channel': 0,  
    },
    {
        'name': 'hydroxyl-chain',
        'value': 'OCCCC',
        'channel': 1,
    },
    {
        'name': 'hydroxyl-ring',
        'value': 'OCC1=CC=CC=C1',
        'channel': 1,
    },
)
NUM_EXAMPLES: int = 15
NUM_EXAMPLE_NEIGHBORS: int = 10

experiment = Experiment.extend(
    'train_encoder_explanation_distances.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)

@experiment.hook('get_target_value')
def get_target_value(e, semantic_distance, structural_distance):
    # return (semantic_distance * structural_distance)
    max_value = max(semantic_distance, structural_distance)
    diff_value = abs(semantic_distance - structural_distance)
    return 0.1 * (2 * max_value) ** 2
    return 0.1 * (max_value + diff_value)
    return min(semantic_distance, structural_distance)
    #return semantic_distance


experiment.run_if_main()
