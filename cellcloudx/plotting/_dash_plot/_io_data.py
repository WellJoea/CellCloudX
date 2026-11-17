from typing import Optional, Union, Dict
import pandas as pd
import numpy as np
import scanpy as sc
import copy

from ...utilis._arrays import list_iter, vartype
from .._colors import *

def get_init_data(locs: Dict[str, np.ndarray] = None, 
                  metadata : pd.DataFrame=None, 
                  colors :dict ={}, 
                  colors_order :dict ={},
                    window_labels : list=None,
                    window_splits : list=None,
                    window_basis : list=None,
                    window_slices : list=None,
                    colormap=None,
                  window_number: int=2,
                  max_window_number=6,
                  scatter_args :dict ={}) -> dict:

    metadata = pd.DataFrame(metadata).copy()

    window_labels = metadata.columns if window_labels is None else window_labels
    window_splits = metadata.columns if window_splits is None else window_splits
    window_basis = list(locs.keys()) if window_basis is None else window_basis
    window_slices = None if window_slices is None else window_slices
    window_number = min(window_number or 2, len(window_labels))
    # assert window_number>=1
    # assert window_number<=2, "only 1 or 2 windows are supported"
    
    window_labels, window_splits, window_basis, window_slices  = [
        [ list_iter(i)[k] for k in range(window_number)]
        for i in [window_labels, window_splits, window_basis, window_slices]
    ]

    colors_order_init = {}
    colors_init = {}
    for i in metadata.columns:
        if vartype(metadata[i]) == 'discrete':
            if colors_order.get(i):
                order = colors_order[i]
            else:
                try:
                    order = metadata[i].cat.categories.tolist()
                except:
                    order = metadata[i].unique().tolist()
            colors_order_init[i] = order

            icolor = (colors[i] if (i in colors) and 
                                    (colors[i] is not None) and
                                    (len(colors[i])>=0)
                                else [])
            icolor = list(icolor) + random_colors(len(order) - len(icolor))
            colors_init[i] = icolor
    
    init_scatter_args = dict(
                scale=1.0,  same_scale=True, 
                size=None, opacity=None,
                figscale=None, werror=None, 
                width = None, height=None, 
                reverse_y=False, reverse_x=False,  
                showlegend=True,
                aspectmode='data',  
                aspectratio=dict(x=1, y=1, z=1), 
                clip=None, 
                sample=min(float(np.round(100000/metadata.shape[0], 2)), 0.5),
                random_state = 200504,
                show_grid =True, 
                showticklabels = False,
                render_mode = 'auto',
                colormap=None,
                template='none',
                legend_dict={ 
                    'orientation':"v",
                    'itemsizing':'constant',
                    'itemwidth': 30,
                    # 'automargin':False,
                    # 'yref':'paper',
                    # 'xref': 'paper',
                    # 'x':1.05,
                    # 'y':0.99,
                    # 'xanchor':"right",
                    # 'yanchor':"top",
                    # 'entrywidthmode': 'fraction',
                    # 'entrywidth':0.001,
                },
                colorbar_dict = dict(
                    thicknessmode="pixels", 
                    thickness=25,
                    lenmode="pixels", 
                    len=300,
                    yanchor="middle",
                    y=0.5,
                    ypad=10,
                    ticks="outside", ),
            )
    init_scatter_args.update(scatter_args)
    scatter_args = [ copy.deepcopy(init_scatter_args) for i in range(window_number) ]


    data_labels = sorted(list(metadata.columns))
    dtypes = [ vartype(metadata[i]) for i in data_labels ]

    point_samples = [i.get('sample', 0.5) for i in scatter_args]
    points_scale = [i.get('scale', 1) for i in scatter_args]
    points_size =[None] *window_number
    colormaps = ['cmap1']*window_number
    initial_data = {
        'locs' : { k:np.asarray(v) for k,v in locs.items() },
        "metadata": { i:metadata[i] for i in metadata.columns },
        'index' : metadata.index,
    }
    initial_paras = {
        'data_labels': data_labels,
        'basis_labes': sorted(list(locs.keys())),
        'data_dtypes': dtypes,
        'window_labels': window_labels,
        'window_splits': window_splits,
        
        'split_iterms': ['NONE']*window_number,
        'window_number': window_number,
        'max_window_number': max_window_number,
        'window_slices': window_slices,
        "current_lables": window_labels,
        
        'point_samples': point_samples,
        'window_renders':['webgl']*window_number,
        'points_size':points_size,
        'points_scale':points_scale,
        'sizes_max': [8]*window_number,
        'window_basis': window_basis,
        'save_figs': [None]*window_number,

        'figscale': 800,
        "history": [],
        'undo_stack': [],
        'redo_stack': [],
        'colors': colors_init or {},
        'colormaps': colormaps,
        'colors_order': colors_order_init or {},
        'scatter_args': scatter_args,
    }
    return initial_data, initial_paras
