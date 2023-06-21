"""
This string will be saved to the experiment's archive folder as the "experiment description"

CHANGELOG

0.1.0 - initial version
"""
import os
import pathlib
import time
import orjson

import numpy as np
import matplotlib.pyplot as plt

from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

PATH = pathlib.Path(__file__).parent.absolute()
__DEBUG__ = True


NUM_ELEMENTS = 128**2
ARRAY_SIZE = 10
CLUSTERING_FACTORS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

@Experiment(base_path=folder_path(__file__),
            namespace=file_namespace(__file__),
            glob=globals())
def experiment(e: Experiment):
    e.log('starting experiment...')

    elements = {}
    for index in range(NUM_ELEMENTS):
        elements[index] = {
            'index': index,
            'array1': np.random.random(size=ARRAY_SIZE),
            'array2': np.random.random(size=ARRAY_SIZE),
        }
        
    for factor in CLUSTERING_FACTORS:
        folder_path = os.path.join(e.path, f'factor_{factor:02d}')
        os.mkdir(folder_path)
        
        start_time = time.time()
        buffer = {}
        for name, element in elements.items():
            buffer[str(name)] = element
            if len(buffer) >= factor:
                file_path = os.path.join(folder_path, 'name.json')
                with open(file_path, mode='wb') as file:
                    content = orjson.dumps(buffer, option=orjson.OPT_SERIALIZE_NUMPY)
                    file.write(content)
                    
                buffer = {}
                    
        duration = time.time() - start_time
        e[f'durations/{factor}'] = duration
        e.log(f' * factor: {factor} - duration: {duration:.2f} - remaining_elements: {len(buffer)}')
    


@experiment.analysis
def analysis(e: Experiment):
    e.log('starting analysis...')


experiment.run_if_main()