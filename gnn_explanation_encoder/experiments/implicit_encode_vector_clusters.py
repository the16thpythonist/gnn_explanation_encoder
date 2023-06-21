"""
This string will be saved to the experiment's archive folder as the "experiment description"

CHANGELOG

0.1.0 - initial version
"""
import os
import random
import pathlib
import typing as t

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as ks

from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from gnn_explanation_encoder.models import DenseImplicitEncoder

PATH = pathlib.Path(__file__).parent.absolute()
__DEBUG__ = True

NUM_DIMENSIONS: int = 500
SEED_VALUE_RANGE: t.Tuple[int, int] = (-5, 5)
VARIANCE: float = 2
NUM_ELEMENTS: int = 300
NUM_TEST: int = 100
TRAIN_RATIO: int = 0.1

EPOCHS: int = 10
BATCH_SIZE: int = 64


@Experiment(base_path=folder_path(__file__),
            namespace=file_namespace(__file__),
            glob=globals())
def experiment(e: Experiment):
    e.log('starting experiment...')

    # As the very first step we need to create the seed vectors around which the clusters should be formed.
    # Around those vectors we will create a normally distributes cluster of other vectors.
    e.log('creating the cluster centroid seeds...')
    seed_vectors = [
        np.random.randint(*SEED_VALUE_RANGE, size=(NUM_DIMENSIONS, )),
        np.random.randint(*SEED_VALUE_RANGE, size=(NUM_DIMENSIONS, )),
    ]
    
    e.log('creating the cluster elements...')
    elements = []
    seed_elements = [[], []]
    for seed_index, center in enumerate(seed_vectors):
        covariance_matrix = np.eye(NUM_DIMENSIONS) * VARIANCE
        cluster = np.random.multivariate_normal(center, covariance_matrix, size=NUM_ELEMENTS)
        elements += [np.array(v) for v in cluster.tolist()]
        seed_elements[seed_index] += [np.array(v) for v in cluster.tolist()]
        
    e.log('calculating the distances...')
    num_elements = len(elements)
    distances = np.zeros(shape=(num_elements, num_elements))
    for i in range(num_elements):
        for j in range(num_elements):
            distances[i, j] = np.linalg.norm(elements[i] - elements[j])
    
    indices = list(range(num_elements))
    test_indices = random.sample(indices, k=NUM_TEST)
    train_indices = list(set(indices).difference(set(test_indices)))
    
    e.log('converting to tensor dataset...')
    xs1, xs2 = [], []
    ys = []
    for i in train_indices:
        for j in train_indices:
            if random.random() < TRAIN_RATIO:
                xs1.append(elements[i])
                xs2.append(elements[j])
                ys.append(distances[i, j])
                
    e.log(f'using {len(ys)}({TRAIN_RATIO*100:.1f}%) pairwise distances distances for the training')
    
    e.log('setting up the encoder model...')
    model: ks.models.Model = DenseImplicitEncoder(
        distance_func=lambda a, b: tf.reduce_sum(tf.square(a - b), axis=-1),
        units=[64, 32, 16, 8],
        size=2,
    )
    model.compile(
        optimizer=ks.optimizers.Adam(learning_rate=0.001),
        loss=ks.losses.MeanSquaredError(),
        metrics=[
            ks.metrics.MeanSquaredError(),
            ks.metrics.MeanAbsoluteError(),
        ],
    )
    
    e.log('starting training...')
    model.fit(
        [np.array(xs1), np.array(xs2)], np.array(ys),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
    )

    test_elements = [elements[i] for i in test_indices]
    encoded = model([tf.convert_to_tensor(test_elements)]).numpy()

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
    ax.scatter(
        encoded[:, 0],
        encoded[:, 1],
        color='gray'
    )
        
    fig.savefig(os.path.join(e.path, 'encoded.pdf'))


@experiment.analysis
def analysis(e: Experiment):
    e.log('starting analysis...')


experiment.run_if_main()