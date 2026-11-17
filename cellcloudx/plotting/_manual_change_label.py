import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad

import sys
from typing import Optional, Union, Dict
import copy 

from ._colors import *

from ._dash_plot._io_data import get_init_data
from ._dash_plot._dash_app import dash_app
from ..utilis._arrays import list_iter, vartype

def manual_label_wrap(adata, 
                        window_labels : list=None,
                        window_splits : list=None,
                        window_basis : list=None,
                        window_slices : list=None,
                        window_number: int=2,
                         colors :dict ={}, 
                         colormap =None,
                      add_genes =False,
                      use_raw = False,
                      start_app=True,
                      scatter_args={}, run_args={},
                      **karges):
    color_init = { k: adata.uns.get(f'{k}_colors', None) for k in adata.obs.columns }
    color_init.update(colors)
    color_init = {k:v for k,v in color_init.items() if (v is not None) and len(v)>=0}
    locs = dict(adata.obsm)
    metadata = adata.obs.copy()
    if add_genes:
        if use_raw:
            adata = adata.raw
        else:
            adata = adata
        metadata = pd.concat([metadata, adata.to_df()], axis=1)

    return manual_label(locs=locs, 
                        metadata=metadata, 
                        colors=color_init, 
                        colormap=colormap,
                        window_labels = window_labels,
                        window_splits = window_splits,
                        window_slices = window_slices,
                        window_basis =window_basis,
                        window_number=window_number,
                        scatter_args=scatter_args, 
                        run_args=run_args, 
                        start_app=start_app,
                        **karges)

def manual_label(locs: Dict[str, np.ndarray] = None,
                  metadata : pd.DataFrame=None, 
                  colors :dict ={}, 
                  colormap =None,
                    window_labels : list=None,
                    window_splits : list=None,
                    window_basis : list=None,
                    window_slices : list=None,
                  window_number: int=2,
                  max_window_number=6,
                    scatter_args :dict ={},
                    start_app =True,
                  run_args :dict ={}):

    data, paras = get_init_data(locs=locs, metadata=metadata,
                                colors=colors,
                                window_labels =window_labels,
                                window_splits =window_splits,
                                window_slices =window_slices,
                                window_basis =window_basis,
                                window_number=window_number,
                                max_window_number=max_window_number,
                                colormap=colormap,
                                scatter_args=scatter_args)

    if start_app:
        dash_app(data, paras, **run_args)
    else:
        return data, paras

