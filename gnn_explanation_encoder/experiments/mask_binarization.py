"""
This string will be saved to the experiment's archive folder as the "experiment description"

CHANGELOG

0.1.0 - initial version
"""
import os
import pathlib
import typing as t

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from visual_graph_datasets.processing.colors import ColorProcessing
from visual_graph_datasets.visualization.base import draw_image
from visual_graph_datasets.visualization.importances import plot_node_importances_border
from graph_attention_student.keras import load_model
from graph_attention_student.utils import array_normalize
from gnn_explanation_encoder.utils import binarize_node_mask

PATH = pathlib.Path(__file__).parent.absolute()
MODEL_PATH: str = os.path.join(PATH, 'assets', 'models', 'rb_dual_motifs')

VALUES: t.List[str] = [
    'R-1(GR-2(B)(B)(Y-3(RCR-4-3(R))-4(B)(RHH-3)))(G-5(RR-6-5(HYG-7-5(M-8(M-9(H)(M))(C)))(B))-7-6)',
    'HR-1(Y-2(R-3(H)(R)(B-4(R)(R-5-1(H-6(H-7(B-8(Y-9-4(RH-10-7-1(CR-11-3-10(Y-12(R-13(H)(HYY))(Y)(M)(R-14-8(BH)(Y)(B-2))))-11))-14)-10)(H))(G-15-3-1))-9)-15-11))(Y)-15-10-5',
    'H-1(B-2(H-3(Y-4-2(B)(BR-5(Y-6(R)(RG-7-1(YY))(R)(RR-8-5-2(MR-9(R)(MM)(B-4)(RRB-10-3(BYR-11-6(G)(G)))))-11)-8))-10)-4-8)-7',
    'G-1(YG-2(M-3(HC)(C-4(R-5(R-6(CHR-7-2(MBY-8-5(G)(R-9-1(H-4)(H)(R-3))(R-10-2(H-11(G-6)(HMY-12-2(G-13-7(C)(R-7))))))-13))-8))(B))-10-7(Y)-12)-9(GG)'
]

CHANNEL: int = 1

FUNCTIONS = [
    binarize_node_mask
]

__DEBUG__ = True


@Experiment(base_path=folder_path(__file__),
            namespace=file_namespace(__file__),
            glob=globals())
def experiment(e: Experiment):
    e.log('starting experiment...')
    num_functions = len(FUNCTIONS)

    model = load_model(MODEL_PATH)
    processing = ColorProcessing()
    
    pdf_path = os.path.join(e.path, 'masks.pdf')
    with PdfPages(pdf_path) as pdf:
        
        for index, value in enumerate(VALUES):
            data = processing.create(
                value=value,
                index=index,
                output_path=e.path,
                width=1000,
                height=1000,
            )
            graph = data['metadata']['graph']
            node_positions = np.array(graph['node_positions'])
            
            out, ni, ei = model.predict_graph(graph)
            ni = array_normalize(ni)
            mask = ni[:, CHANNEL]
            
            fig, rows = plt.subplots(
                ncols=1 + num_functions,
                nrows=1,
                figsize=(6 * (1 + num_functions), 6),
                squeeze=False
            )
            ax = rows[0][0]
            draw_image(ax, data['image_path'])
            plot_node_importances_border(ax, graph, node_positions, mask)
            
            for j, func in enumerate(FUNCTIONS):
                ax = rows[0][j+1]
                ax.set_title(func.__name__)
                
                mask_binary = func(graph, mask)
                
                draw_image(ax, data['image_path'])
                plot_node_importances_border(ax, graph, node_positions, mask_binary)
            
            pdf.savefig(fig)
            plt.close(fig)


@experiment.analysis
def analysis(e: Experiment):
    e.log('starting analysis...')


experiment.run_if_main()