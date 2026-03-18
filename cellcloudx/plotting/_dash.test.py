import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import os
import sys
import importlib
import copy

import skimage as ski 
sys.path.append('/home/zhouw/JupyterCode/CellCloudX')
import cellcloudx as cc
import scanpy as sc

from cellcloudx.plotting._manual_change_label import *
importlib.reload(sys.modules['cellcloudx'])
importlib.reload(sys.modules['cellcloudx.plotting'])
importlib.reload(sys.modules['cellcloudx.plotting._manual_change_label'])
importlib.reload(sys.modules['cellcloudx.plotting._dash_plot'])
importlib.reload(sys.modules['cellcloudx.plotting._dash_plot._config'])
importlib.reload(sys.modules['cellcloudx.plotting._dash_plot._dash_app'])
importlib.reload(sys.modules['cellcloudx.plotting._dash_plot._dash_function'])
importlib.reload(sys.modules['cellcloudx.plotting._dash_plot._io_data'])
importlib.reload(sys.modules['cellcloudx.plotting._dash_plot._plot_wrap'])

from functools import lru_cache
@lru_cache(maxsize=None)
def load_data():
    print('refrash', 2222)
    adata = sc.read_h5ad('/home/zhouw/WorkSpace/11Project/06cerebellum_for_3d/02Results/P0/adata.test.h5ad')
    # adata = sc.read_h5ad('/home/zhouw/WorkSpace/11Project/06cerebellum_for_3d/01Data/P0/adatasts.1_25.clean.241204.h5ad')
    adata.X = adata.X.astype(np.int64)
    idx = np.random.choice(adata.shape[0], size=500, replace=False)
    return adata
adata = load_data()

import os
ip = '127.0.0.1'
port = '10023'
#bash unset HOST SERVER_NAME
os.environ['HOST'] = ip
os.environ['port'] = port

print('refrash', 33333)

start_app = True
data = manual_label_wrap(adata, 
                  window_labels=['subtype','celltype'],
                  window_splits=None,
                  window_basis=['spatial','align3d'],
                  add_genes=False, 
                  start_app=start_app,
                  run_args={'port': port, 'host':ip, 'debug':True})

if not start_app:
    idx = 0
    ipars = copy.deepcopy(data['scatter_args'][idx])
    ipars.update({
        'scale': 10
        # 'size': None, #TODO
        # 'save': data['save_figs'][idx],
        # 'render_mode': data['window_renders'][idx],
        # 'sample': data['point_samples'][idx],
    })

    from cellcloudx.plotting._dash_plot._plot_wrap import scatter_wrap
    fig = scatter_wrap(*data, idx, **ipars)
    fig.show()