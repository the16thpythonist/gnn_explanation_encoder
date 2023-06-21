"""
This is a sub experiment derived from ...

CHANGELOG

0.1.0 - initial version
"""
import os
import pathlib

from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

PATH = pathlib.Path(__file__).parent.absolute()

VISUAL_GRAPH_DATASET_PATH: str = '/media/ssd/.visual_graph_datasets/datasets/aqsoldb'
MODEL_PATH: str = os.path.join(PATH, 'assets', 'models', 'aqsoldb')
DOMAIN_VALUE_KEY: str = 'smiles'
DATASET_NAME: str = 'aqsoldb__distances'

PAIRWISE_DISTANCES_RATIO: float = 0.6e-2
FIDELITY_THRESHOLD = 0.2
FIDELITY_RANGE: float = 10.0

experiment = Experiment.extend(
    'generate_pairwise_explanation_distance_dataset.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)


@experiment.hook('hook')
def hook(e, parameter):
    e.log(f'parameter: {parameter}')


@experiment.analysis
def analysis(e):
    e.log('more analysis...')


experiment.run_if_main()
