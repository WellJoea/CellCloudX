import numpy as np
import pandas as pd
import scanpy as sc
import plotly.express as px 
from typing import Optional, Union, Dict
import colorsys
import scipy.sparse  as ssp
import matplotlib
from cellcloudx.plotting._colors import cmap1px, cmap2px, matplotlib_to_plotly
def manual_label_wrap(adata, label_type, 
                      split_type : Union[str,]=None,
                      locs_type : Union[str,]=None,
                      add_genes =False,
                      use_raw = False,
                      colors = {},
                      colormap = None,
                      scatter_args={}, run_args={},
                      **karges):
    color_init = { k: adata.uns.get(f'{k}_colors', None) for k in adata.obs.columns }
    color_init.update(colors)
    color_init = {k:v for k,v in color_init.items() if (v is not None) and len(v)>=0}
    locs = dict(adata.obsm)
    labels = adata.obs.copy()
    if add_genes:
        if use_raw:
            adata = adata.raw
        else:
            adata = adata
        labels = pd.concat([labels, adata.to_df()], axis=1)

    if (locs_type is None):
        if 'spatial' in locs.keys():
            locs_type = 'spatial'
        else:
            locs_type = list(locs.keys())[0]
    return manual_label(locs=locs, 
                        labels=labels, 
                        colors=color_init, 
                        colormap=colormap,
                        label_type = label_type,
                        split_type =split_type,
                        locs_type = locs_type,
                        scatter_args=scatter_args, 
                        run_args=run_args, **karges)

def manual_label(locs: Dict[str, np.ndarray] = None,
                  labels : pd.DataFrame=None, 
                  colors :dict ={}, 
                  colormap =None,
                  label_type : Union[str, int]=None,
                  split_type : Union[str,]=None,
                  locs_type : Union[str,]=None,
                  initial_data : dict=None, scatter_args :dict ={},
                  debug :bool =True, 
                  run_args :dict ={}):
    initial_data = get_init_data(locs=locs, labels=labels,
                                colors=colors,
                                label_type =label_type,
                                split_type =split_type,
                                locs_type =locs_type,
                                colormap=colormap,
                                scatter_args=scatter_args)
    dash_app(initial_data,  debug=debug,  **run_args)

def get_init_data(locs: Dict[str, np.ndarray] = None, 
                  labels : pd.DataFrame=None, 
                  colors :dict ={}, 
                  colormap =None,
                  colors_order :dict ={},
                  label_type : Union[str, int]=None,
                  split_type : Union[str,]=None,
                  locs_type : Union[str,]=None,
                  initial_data : dict=None, scatter_args :dict ={}) -> dict:
    if initial_data is None: 
        labels = pd.DataFrame(labels).copy()
        label_type = label_type or labels.columns[0]
        if split_type is None:
            if len(labels.columns) > 1:
                split_type = labels.columns[0]
            else:
                split_type = label_type
        
        locs_type = locs_type or list(locs.keys())[0]
        colors_order_init = {}
        colors_init = {}
        for i in labels.columns:
            if vartype(labels[i]) == 'discrete':
                if colors_order.get(i):
                    order = colors_order[i]
                else:
                    try:
                        order = labels[i].cat.categories.tolist()
                    except:
                        order = labels[i].unique().tolist()
                colors_order_init[i] = order

                icolor = (colors[i] if (i in colors) and 
                                        (colors[i] is not None) and
                                        (len(colors[i])>=0)
                                    else [])
                icolor = list(icolor) + random_colors(len(order) - len(icolor))
                colors_init[i] = icolor
 
        initial_data = {
            'locs' : { k:np.asarray(v) for k,v in locs.items() },
            "labels": { i:labels[i] for i in labels.columns },
            "current_type": label_type or labels.columns[0],
            'split_type': split_type,
            'split_iterm': 'ALL_',
            'locs_type': locs_type,
            "history": [],
            'undo_stack': [],
            'redo_stack': [],
            'colors': colors_init or {},
            'colormap': colormap or 'cmap1',
            'colors_order': colors_order_init or {},
            'scatter_args': 
                dict(
                    scale=1,  same_scale=True, 
                    size=None, opacity=None,
                    figscale=None, werror=None, 
                    width = None, height=None, scene_aspectmode='data',  
                    reverse_y=False, reverse_x=False,  
                    showlegend=True,
                    scene_aspectratio=dict(x=1, y=1, z=1), clip=None, sample=None,
                    random_state = 200504, order ='guess',
                    show_grid =False, 
                    showticklabels = False,
                    colordict = {},
                    render_mode = 'auto',
                    colormap=None, template='none',
                    itemsizing='constant',
                    itemwidth=None,
                    legdict={},
                )
        }
        initial_data['scatter_args'].update(scatter_args)
        initial_data['index'] = labels.index
    else:
        if 'index' not in initial_data:
            initial_data['index'] = np.arange(len(initial_data['locs']))
    return initial_data

def vartype(vector):
    if isinstance(vector, (list, tuple)):
        vector = np.asarray(vector)
    if isinstance(vector, pd.Series):
        dtype = vector.dtype
    elif isinstance(vector, pd.DataFrame):
        dtype = vector.dtypes.iloc[0]
    elif isinstance(vector, np.ndarray):
        dtype = vector.dtype
    else:
        raise('wrong numpy array or pandas object.')

    if (
        isinstance(dtype,  pd.CategoricalDtype) 
        or pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_bool_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
    ):  
        return "discrete"
    elif pd.api.types.is_numeric_dtype(dtype):
        return "continuous"
    else:
        raise('wrong numpy array or pandas object.')

def color_palette(len_color):
    #plt.colormaps()
    #plt.get_cmap('tab20')
    if len_color <= 20:
        palette = sc.pl.palettes.default_20
    elif len_color <= 28:
        palette = sc.pl.palettes.default_28
    elif len_color <= len(sc.pl.palettes.default_102):  # 103 colors
        palette = sc.pl.palettes.default_102
    else:
        palette = random_colors(len_color)
    return palette

def random_colors(n):
    colors = []
    while len(colors) < n:
        colors += np.random.randint(0, 0xFFFFFF, n-len(colors)).tolist()
        colors = list(set(colors))
    colors = ['#%06X' % color for color in colors]
    return colors

def inverse_colors(colors):
    return ['#%02x%02x%02x' % (255 - int(color[1:3], 16), 
                                255 - int(color[3:5], 16), 
                                255 - int(color[5:7], 16))
             for color in colors]

def generate_color(index=1, total=1):
    hue = (index / total) if total > 0 else 0
    r, g, b = colorsys.hls_to_rgb(hue, 0.75, 0.6)
    return '#{:02X}{:02X}{:02X}'.format(int(r*255), int(g*255), int(b*255))

def get_contrasting_text_color(bg_hex):
    bg_hex = bg_hex.lstrip('#')
    r, g, b = int(bg_hex[0:2], 16), int(bg_hex[2:4], 16), int(bg_hex[4:6], 16)
    brightness = (r * 299 + g * 587 + b * 114) / 1000
    return "#FFFFFF" if brightness < 128 else "#000000"

def px_colormap():
    px_cor = {}
    for i in px.colors.named_colorscales():
        px_cor[i] = i
        px_cor[f'{i}_r'] = f'{i}_r'
    px_cor['cmap1']  = cmap1px
    px_cor['cmap1px'] = cmap1px
    px_cor['cmap2']  = cmap2px
    px_cor['cmap2px']= cmap2px
    return px_cor

def get_px_colormap(icolor):
    px_cor = px_colormap()
    if icolor in px_cor.keys():
        return px_cor.get(icolor)
    else:
        # return icolor
        # raise ValueError(f'No such color map {icolor} in px.colors.named_colorscales()')
        return matplotlib_to_plotly(icolor)

def scatter2d(IData : dict, group=None, 
              split=None, split_iterm=None,
              basis=None, animation_frame=None, 
               scale=1,  same_scale=True, 
               size=None, opacity=None,
               save=None,
               figscale=800, werror=20, 
                render_mode='auto', 
                width = None, height=None, scene_aspectmode='data',  
                reverse_y=False, reverse_x=False,  
                showlegend=True,
                scene_aspectratio=dict(x=1, y=1, z=1), clip=None, sample=None,
                random_state = 200504, order ='guess',
                show_grid =True, 
                showticklabels = False,
                colordict = {},
                colormap=None, template='none',
                itemsizing='constant',
                itemwidth=None,
                legdict={},
                  **kargs):

    group = group or IData.get('current_type')
    split = split or IData.get('split_type')
    split_iterm = split_iterm or IData.get('split_iterm')
    basis = basis or IData.get('locs_type')
    colormap = colormap or IData.get('colormap')
    animation_frame = animation_frame or IData.get('animation_frame') or None

    Data = pd.DataFrame(IData['labels'])
    Data.index = IData['index']
    Data[list('xy')] =  np.array(IData['locs'][basis])[:,:2]
    Data['index'] = Data.index
    Data['custom_index'] = np.arange(len(Data)).astype(str)
    Colors = IData.get('colors') or {}
    Colors.update(colordict)
    Colors = {k:v for k,v in Colors.items() if (v is not None) and len(v)>=0}

    if not sample is None:
        if (type(sample) == int) and (sample>1):
            Data = Data.sample(n=sample, 
                                    replace=False,
                                    random_state=random_state)
        if (type(sample) == float) and (sample<=1):
            Data = Data.sample(frac=sample, 
                                    replace=False,
                                    random_state=random_state)
    if split_iterm != 'ALL_':
        Data = Data[Data[split]==split_iterm]

    dimdict = dict(zip(list('xy'), list('xy')))
    dimdict['custom_data'] = [ Data['custom_index']]
    dimdict['render_mode'] = render_mode
    dimdict['animation_frame'] = animation_frame

    if not group is None:
        ctype   = vartype(Data[group])
        dimdict['color']=group
        if ctype == 'discrete':
            colors_order = list(IData['colors_order'].get(group, []))
            try:
                order = Data[group].cat.categories.tolist()
            except:
                order = Data[group].unique().tolist()
            for i in order:
                if i not in colors_order:
                    colors_order.append(i)

            IColor = Colors.get(group) or color_palette(len(colors_order))
            IColor = [ i[:7] if (type(i)== str) and (i.startswith('#')) else i for i in IColor ]
            IColor = IColor + random_colors(len(colors_order) - len(IColor))

            IData['colors'][group] = IColor # update colors
            IData['colors_order'][group] = colors_order # update colors orders
            category_orders={group: colors_order}

            dimdict.update({'category_orders': category_orders, 
                            'color_discrete_sequence':IColor}) #'animation_frame': group
        elif ctype == 'continuous':
            if not clip is None:
                Data[group] = np.clip(Data[group], clip[0], clip[1])
            if order =='guess' or  order == True:
                Data = Data.sort_values(by = group, ascending=True)
            IColor = get_px_colormap(colormap)
            dimdict.update({'color_continuous_scale': IColor})

    if figscale:
        werror = werror or 20
        # width = figscale+werror if width is None else width
        height= figscale if height is None else height

    fig = px.scatter(Data, hover_name="index", hover_data=["index"],
                     width=width, height=height,
                    **dimdict, **kargs)

    if reverse_y:
        fig.update_yaxes(autorange="reversed")
    if reverse_x:
        fig.update_xaxes(autorange="reversed")


    itemwidth or 30
    fig.update_layout(#legend=dict(itemsizing = 'constant'),
                    legend=dict( itemsizing = itemsizing, itemwidth=itemwidth, **legdict),
                        showlegend=showlegend,
                        scene_aspectmode=scene_aspectmode,
                        template=template,
                        scene_aspectratio=scene_aspectratio,
                        scene=dict(
                            xaxis=dict(visible=show_grid, showticklabels=showticklabels),
                            yaxis=dict(visible=show_grid, showticklabels=showticklabels),
                            zaxis=dict(visible=show_grid, showticklabels=showticklabels),
                        ),
                        plot_bgcolor='#FFFFFF',
                        title=f"current type: {group}",
                        xaxis_title="X Axis",
                        yaxis_title="Y Axis",
                        dragmode='select' , # select tools
                        ) #
                        #margin=dict(l=20, r=20, t=20, b=20),template='simple_white', 
                        #paper_bgcolor='#000000',
                        #plot_or='#000000'
                        #fig.update_xaxes(visible=False, showticklabels=False)

    fig.update_traces(text=[{"index": Data.index}], 
                        # selected=dict(marker=dict(color='red', size=12)),
                        # unselected=dict(marker=dict(opacity=0.5)) 
                    )
    if not size is None:
        fig.update_traces(marker=dict(opacity=opacity, line=dict(width=0,color='DarkSlateGrey')),
                            selector=dict(mode='markers'))
    else:
        fig.update_traces(marker=dict(opacity=opacity, size=scale, line=dict(width=0,color='DarkSlateGrey')),
                            selector=dict(mode='markers'))
    if same_scale:
        fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1,
        )
    if save:
        if save.endswith('.html'):
            fig.write_html(save)
        else:
            fig.write_image(save)
    return fig

# init_data = get_init_data(dict(adata.obsm), adata.obs[['celltype', 'level_3']], 
#                             colors = { k: adata.uns.get(f'{k}_colors', None) for k in adata.obs.columns })

# scatter2d(init_data)


def dash_app(initial_data, app=None, server=None, debug=True, **run_args):
    import dash
    from dash import Dash, dcc, html, callback_context
    from dash.dependencies import Input, Output, State
    from datetime import datetime
    import copy

    if app is None:
        app = Dash(__name__)
    if server is None:
        server = app.server

    app.layout = html.Div([
        dcc.Store(id='data-store', data=initial_data),
        dcc.Store(id='selection-store', data=[]),
        
        html.Div([
            html.Div([
                html.Label("Basis Type:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='basis-type-dropdown',
                    options=[{'label': i, 'value': i} for i in initial_data["locs"].keys()],
                    value=initial_data['locs_type'],
                    clearable=False,
                    style={'width': '100%'}
                )
            ], style={'flex': '1', 'backgroundColor': 'white', 'marginRight': '10px'}),

            html.Div([
                html.Label("Label Type:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='label-type-dropdown',
                    options=[{'label': i, 'value': i} for i in initial_data["labels"].keys()],
                    value=initial_data['current_type'],
                    clearable=False,
                    style={'width': '100%'}
                )
            ], style={'flex': '1', 'backgroundColor': 'white', 'marginRight': '10px'}),

            html.Div([
                html.Label("Split Type:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='split-type-dropdown',
                    options=[{'label': i, 'value': i} for i in initial_data["labels"].keys()],
                    value=initial_data['split_type'],
                    clearable=False,
                    style={'width': '100%'}
                )
            ], style={'flex': '1', 'backgroundColor': 'white', 'marginRight': '10px'}),

            html.Div([
                html.Label("Split Iterm:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='split-iterm-dropdown',
                    options=[{'label': i, 'value': i} 
                            for i in ['ALL_'] + sorted(set(initial_data["labels"][initial_data['split_type']]))],
                    value=initial_data['split_iterm'],
                    clearable=False,
                    style={'width': '100%'}
                )
            ], style={'flex': '1', 'backgroundColor': 'white', 'marginRight': '10px'}),

            html.Div([
                html.Label("Colormap:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='point-colormap-input',
                    options=[{'label': i, 'value': i} for i in sorted(px_colormap().keys())],
                    value=initial_data['colormap'],
                    clearable=False,
                    style={'width': '100%'}
                )
            ], style={'flex': '1', 'backgroundColor': 'white', 'marginRight': '10px'}),

            html.Div([
                html.Label('Point Size:', style={'fontWeight': 'bold'}),
                dcc.Input(id='point-size-input', type='number', 
                          min=0,
                        value=initial_data['scatter_args']['scale'], 
                        style={'width': '95%'}),
            ], style={'flex': '1', 'backgroundColor': 'white', 'marginRight': '10px'}),
        
            html.Div([
                html.Label('Figure Scale:', style={'fontWeight': 'bold'}),
                dcc.Input(id='figure-scale-input', type='number',
                            min=300, 
                            step=25,
                            value=initial_data['scatter_args']['figscale'], 
                            style={'width': '95%'}),
            ], style={'flex': '1', 'backgroundColor': 'white'}),

        ], style={'display': 'flex', 'marginBottom': '10px', 'width': '100%'}),

        html.Div([
                html.Div([
                    html.Label("New Label:", style={'fontWeight': 'bold'}),
                    dcc.Input(id='label-input', type='text', placeholder='Enter new label...',
                            style={'width': '95%', })
                ], style={'flex': '3', 'minWidth': '50px',
                        'backgroundColor': 'white','marginRight': '10px'}),

                html.Button('Add Selection', id='add-selection-button', 
                        style={'marginRight': '10px', 'backgroundColor': '#4CAF50', 'color': 'white'}),
                html.Button('Clear Selection', id='clear-selection-button',
                        style={'marginRight': '10px', 'backgroundColor': '#f44336', 'color': 'white'}),
                html.Button('Update Labels', id='update-button', 
                        style={'marginRight': '10px', 'backgroundColor': '#2196F3', 'color': 'white'}),
                html.Button('Undo', id='undo-button', 
                            style={'marginRight': '10px', 'backgroundColor': '#607D8B', 'color': 'white'}),
                html.Button('Redo', id='redo-button', 
                            style={'backgroundColor': '#607D8B', 'color': 'white'}),

        ], style={'display': 'flex', 'marginBottom': '10px', 'width': '100%'}),

        html.Div([
            html.Div([
                dcc.Input(
                    id='filepath-input', 
                    type='text', 
                    placeholder='Save path (e.g.: ./saved_labels.csv)',
                    style={'width': '100%'}
                ),
                html.Button(
                    'Save File', 
                    id='save-file-button', 
                    style={'marginLeft': '1px', 'width': '100%', 'backgroundColor': '#9C27B0', 'color': 'white'}
                )
            ], style={'display': 'flex', 'width': '100%'}),
            html.Div([
                html.Button(
                    'Download CSV', 
                    id='download-button', 
                    style={'marginLeft': '5px', 'backgroundColor': '#FF9800', 'color': 'white'}
                ),
            ], style={'display': 'flex', 'width': '100%'}),

            html.Div([
                dcc.Input(
                    id='htmlpath-input', 
                    type='text', 
                    placeholder='Save path (e.g.: ./saved_figure.html)',
                    style={'width': '100%'}
                ),
                html.Button(
                    'Save Figure', 
                    id='save-fig-button', 
                    style={'marginLeft': '1px', 'width': '100%', 'backgroundColor': '#9C27B0', 'color': 'white'}
                )
            ], style={'display': 'flex', 'width': '100%'}),

            html.Div([
                html.Button(
                    'Download FIG', 
                    id='download-Fig-button', 
                    style={'marginLeft': '5px', 'backgroundColor': '#FF9800', 'color': 'white'}
                ),
            ], style={'display': 'flex', 'width': '100%'}),

            html.Div(id='save-status', 
                     style={'overflowY': 'auto', 'marginLeft': '5px',
                        'backgroundColor': '#fff',
                        'border': '1px solid #ddd', 'borderRadius': '5px', 'width': '150%'}),

        ], style={'display': 'flex', 'marginBottom': '15px', 'width': '100%'}),
        # downoad csv
        dcc.Download(id="download-dataframe-csv"),
        dcc.Download(id="download-fig-html"),

        #scatter plot
        html.Div( 
            style={'display': 'flex', 'gap': '10px'},
            children=[
                html.Div(children=[
                    dcc.Graph(id='scatter-plot1', 
                              figure=scatter2d(initial_data, **initial_data['scatter_args']),
                              config={'scrollZoom': True} ),
                ]),
                html.Div(children=[
                    dcc.Graph(id='scatter-plot', 
                              figure=scatter2d(initial_data, **initial_data['scatter_args']),
                              config={'scrollZoom': True, 
                                      'responsive':True, 
                                      'showTips':True,
                                        # "modeBarButtonsToAdd": [
                                        #         "drawline",
                                        #         "drawclosedpath",
                                        #         "drawrect",
                                        #         "eraseshape",],
                                      'doubleClick': 'reset+autosize'} ),
                ]),
        ]),
        # dcc.Graph(id='scatter-plot', figure=scatter2d(initial_data, **initial_data['scatter_args'])),

        # Current Selection
        html.Div([
            html.H4("Current Selection:",
                    style={'backgroundColor': 'white', 'width': '20%', 'marginBottom': '10px'}),
            html.Div(id='selected-points-display', 
                    style={'maxHeight': '200px', 'overflowY': 'auto', 
                        'backgroundColor': '#fff',
                        'border': '1px solid #ddd', 'borderRadius': '5px', 'width': '100%'})
        ], style={'marginTop': '20px', 'width': '100%'}),
        
        # History Records
        html.Div([
            html.H4("History Records:", 
                    style={'backgroundColor': 'white', 'width': '20%', 'marginBottom': '10px'}),
            html.Div(id='history-div', 
                    style={'maxHeight': '200px', 'overflowY': 'auto', 
                        'backgroundColor': '#fff',
                        'border': '1px solid #ddd', 'borderRadius': '5px', 'width': '100%'})
        ], style={'marginTop': '20px', 'width': '100%'}),
        

    ], style={'padding': '20px', 'maxWidth': '120000px', 'margin': '0 auto', 'width': '98%'})


    @app.callback(
        Output('scatter-plot', 'figure', allow_duplicate=True),
        Output('data-store', 'data', allow_duplicate=True),
        Input('basis-type-dropdown', 'value'), 
        State('data-store', 'data'),
        prevent_initial_call=True
    )
    def update_plot1(new_basis, data):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate
        data['locs_type'] = new_basis
        updated_fig = scatter2d(data, **data['scatter_args'])
        return updated_fig, data

    @app.callback(
        Output('scatter-plot', 'figure', allow_duplicate=True),
        Output('data-store', 'data', allow_duplicate=True),
        # Input('update-point-size-button', 'n_clicks'),
        Input('point-size-input', 'value'), 
        State('data-store', 'data'),
        prevent_initial_call=True
    )
    def update_plot2(new_size, data):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate

        if new_size is None or new_size <= 0:
            raise dash.exceptions.PreventUpdate

        data['scatter_args']['scale'] = new_size
        updated_fig = scatter2d(data, **data['scatter_args'])
        return updated_fig, data

    @app.callback(
        Output('scatter-plot', 'figure', allow_duplicate=True),
        Output('data-store', 'data', allow_duplicate=True),
        # Input('update-point-size-button', 'n_clicks'),
        Input('figure-scale-input', 'value'), 
        State('data-store', 'data'),
        prevent_initial_call=True
    )
    def update_plot3(new_size, data):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate

        if new_size is None or new_size <= 0:
            raise dash.exceptions.PreventUpdate

        data['scatter_args']['figscale'] = new_size
        updated_fig = scatter2d(data, **data['scatter_args'])
        return updated_fig, data

    @app.callback(
        Output('scatter-plot', 'figure', allow_duplicate=True),
        Output('data-store', 'data', allow_duplicate=True),
        Input('point-colormap-input', 'value'), 
        State('data-store', 'data'),
        prevent_initial_call=True
    )
    def update_plot4(new_colormap, data):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate
        data['colormap'] = new_colormap
        updated_fig = scatter2d(data, **data['scatter_args'])
        return updated_fig, data

    @app.callback(
        Output('split-iterm-dropdown', 'options'),
        Output('split-iterm-dropdown', 'value'),
        Input('split-type-dropdown', 'value'),
        State('data-store', 'data'),
        prevent_initial_call=True
    )
    def update_split_iterm_options(selected_split_type, data):
        try:
            available_labels = sorted(set(data["labels"][selected_split_type]))
            options = [{'label': 'ALL_', 'value': 'ALL_'}] + [
                {'label': str(label), 'value': label} for label in available_labels
            ]
            data["split_iterm"] = 'ALL_'
            return options, 'ALL_'
        except KeyError:
            data["split_iterm"] = 'ALL_'
            return [{'label': 'ALL_', 'value': 'ALL_'}], 'ALL_'

    @app.callback(
        Output('selection-store', 'data'),
        Output('selected-points-display', 'children'),
        Input('add-selection-button', 'n_clicks'),
        Input('clear-selection-button', 'n_clicks'),
        State('scatter-plot', 'selectedData'),
        State('selection-store', 'data'),
        prevent_initial_call=True
    )
    def update_selection(add_clicks, clear_clicks, selectedData, current_selection):
        ctx = callback_context
        if not ctx.triggered:
            if not current_selection:
                return current_selection, html.Span("No points selected", style={'color': '#999'})
            raise dash.exceptions.PreventUpdate
        
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if trigger_id == 'clear-selection-button':
            current_selection = []
        elif trigger_id == 'add-selection-button' and selectedData:
            new_indices = [ int(point['customdata'][0]) for point in selectedData.get('points', [])]
            # new_indices = [point['pointIndex'] for point in selectedData.get('points', [])]
            current_selection = list(set(current_selection + new_indices))
            
        # 显示选中点信息
        display_content = []
        selected_count = len(current_selection)
        sample_points = sorted(current_selection)  # 显示所有选中点

        if selected_count:
            sample_text = ', '.join(map(str, sample_points))
            display_content = html.Div([
                html.Span(f"Selected Points ({selected_count}): ", 
                        style={'fontWeight': 'bold', 'color': '#333'}),
                html.Span(sample_text, style={'color': '#666'})
            ])
        else:
            display_content = html.Span("No points selected", style={'color': '#999'})

        return current_selection, display_content

    # 回调2：更新数据和历史记录
    @app.callback(
        Output('data-store', 'data'),
        Output('scatter-plot', 'figure'),
        Output('history-div', 'children'),
        Input('label-type-dropdown', 'value'),
        Input('split-type-dropdown', 'value'),
        Input('split-iterm-dropdown', 'value'),
        Input('update-button', 'n_clicks'),
        State('label-input', 'value'),
        State('selection-store', 'data'),
        State('data-store', 'data')
    )
    def update_data(selected_type, split_type, split_iterm, n_clicks, new_label, selection_store, data):
        ctx = callback_context
        triggered = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # 处理标签类型切换
        if triggered == 'label-type-dropdown':
            data["current_type"] = selected_type
        
        if triggered == 'split-type-dropdown':
            data["split_type"] = split_type
        
        if triggered == 'split-iterm-dropdown':
            data["split_iterm"] = split_iterm

        # 处理标签更新
        if triggered == 'update-button' and selection_store and new_label:
            current_type = data["current_type"]
            # data["labels"][current_type].iloc[np.array(selection_store).astype(np.int64)] = new_label
            # print(selection_store)
            for i in selection_store:
                data["labels"][current_type][int(i)] = new_label
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            record = f"{timestamp}: Updated {len(selection_store)} points to '{new_label}' ({current_type})"
            data["history"].append(record)
            # data["history"].append(','.join(selection_store))
        
        # 生成历史记录显示
        history_display = []
        for record in data.get("history", []):
            history_display.append(html.Li(record, style={'marginBottom': '5px'}))
        
        if not history_display:
            history_display = html.Span("No history records yet", style={'color': '#999'})
        else:
            history_display = html.Ul(history_display, 
                                    style={'listStyleType': 'none', 'padding': '0'})
        
        return data, scatter2d(data, **data['scatter_args']), history_display

    # 回调3：文件下载
    @app.callback(
        Output("save-status", "children", allow_duplicate=True),
        Input("download-button", "n_clicks"),
        State("data-store", "data"),
        prevent_initial_call=True
    )
    def download_csv(n_clicks, data):
        # df = pd.DataFrame({
        #     "x": data["x"],
        #     "y": data["y"],
        #     "label": data["labels"][data["current_type"]]
        # }, index=data.get('index', np.arange(len(data['x']))))
        df = pd.DataFrame(data["labels"], index=data['index'])
        df.to_csv('labels.csv', index=True)
        return html.Span(f"Successfully download to: labels.csv", 
                        style={'color': 'green', 'backgroundColor': '#fff',})
    # 回调4：文件保存
    @app.callback(
        Output("save-status", "children"),
        Input("save-file-button", "n_clicks"),
        State("filepath-input", "value"),
        State("data-store", "data"),
        prevent_initial_call=True
    )
    def save_file(n_clicks, file_path, data):
        if not file_path:
            return html.Span("Please specify a valid file path!", style={'color': 'red'})
        try:
            df = pd.DataFrame(data["labels"], index=data['index'])
            df.to_csv(file_path, index=True)
            return html.Span(f"Successfully saved to: {file_path}", 
                            style={'color': 'green', 'backgroundColor': '#fff',})
        except Exception as e:
            return html.Span(f"Save csv failed: {str(e)}", style={'color': 'red'})

    @app.callback(
        Output("save-status", "children", allow_duplicate=True),
        Input("download-Fig-button", "n_clicks"),
        State("data-store", "data"),
        prevent_initial_call=True
    )
    def download_fig(n_clicks, data):
        data['scatter_args']['save'] = "figure.html"
        scatter2d(data, **data['scatter_args'])
        data['scatter_args']['save'] = None
        return html.Span(f"Successfully download to: figure.html", 
                        style={'color': 'green', 'backgroundColor': '#fff',})
    
    @app.callback(
        Output("save-status", "children", allow_duplicate=True),
        Input("save-fig-button", "n_clicks"),
        State("htmlpath-input", "value"),
        State("data-store", "data"),
        prevent_initial_call=True
    )
    def save_fig(n_clicks, fig_path, data):
        if not fig_path:
            return html.Span("Please specify a valid file path!", style={'color': 'red'})
        try:
            data['scatter_args']['save'] = fig_path
            scatter2d(data, **data['scatter_args'])
            data['scatter_args']['save'] = None
            return html.Span(f"Successfully saved to: {fig_path}", 
                            style={'color': 'green', 'backgroundColor': '#fff',})
        except Exception as e:
            return html.Span(f"Save fig failed: {str(e)}", style={'color': 'red'})

    app.run_server( debug=debug, **run_args)
    return app



def dash_app2(initial_data, app=None, server=None, debug=True, **run_args):
    import dash
    from dash import Dash, dcc, html, callback_context
    from dash.dependencies import Input, Output, State
    from datetime import datetime
    import pandas as pd
    import copy

    # 初始化应用
    if app is None:
        app = Dash(__name__)
    if server is None:
        server = app.server

    # CSS样式配置
    app.css.append_css({
        'external_url': [
            'https://codepen.io/chriddyp/pen/bWLwgP.css',
            'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css'
        ]
    })

    # 样式常量
    STYLE = {
        'controlGroup': {'display': 'flex', 'marginBottom': '10px', 'width': '100%'},
        'graphContainer': {'flex': 1, 'position': 'relative', 'height': '60vh'},
        'floatingControls': {'position': 'absolute', 'top': '10px', 'right': '10px', 'zIndex': 100}
    }

    # 创建通用控件组
    def create_control_group(controls):
        return html.Div(
            style=STYLE['controlGroup'],
            children=[
                html.Div(
                    style={
                        'flex': str(ctrl.get('flex', 1)),
                        'backgroundColor': 'white',
                        'marginRight': '10px',
                        'minWidth': ctrl.get('minWidth', 'auto')
                    },
                    children=[
                        html.Label(ctrl['label'], style={'fontWeight': 'bold'}),
                        ctrl['component']
                    ]
                ) for ctrl in controls
            ]
        )

    # 创建带控制按钮的图表容器
    def create_graph_container(graph_id):
        return html.Div(
            style=STYLE['graphContainer'],
            children=[
                dcc.Graph(
                    id=graph_id,
                    config={
                        'scrollZoom': True,
                        'doubleClick': 'reset+autosize',
                        'modeBarButtonsToAdd': ['select2d', 'lasso2d']
                    }
                ),
                html.Div(
                    style=STYLE['floatingControls'],
                    children=[
                        html.Button('🗑️', 
                                  id=f'clear-{graph_id}', 
                                  className='icon-btn',
                                  title='Clear selection'),
                        html.Button('💾', 
                                  id=f'export-{graph_id}', 
                                  className='icon-btn',
                                  title='Export figure')
                    ]
                )
            ]
        )

    # 基础控件配置
    controls = [
        {
            'label': "Basis Type",
            'component': dcc.Dropdown(
                id='basis-type-dropdown',
                options=[{'label': k, 'value': k} for k in initial_data["locs"]],
                value=initial_data['locs_type'],
                clearable=False
            ),
            'flex': 2
        },
        {
            'label': "Label Type",
            'component': dcc.Dropdown(
                id='label-type-dropdown',
                options=[{'label': k, 'value': k} for k in initial_data["labels"]],
                value=initial_data['current_type'],
                clearable=False
            ),
            'flex': 2
        },
        {
            'label': "Colormap",
            'component': dcc.Dropdown(
                id='colormap-selector',
                options=[{'label': k, 'value': k} for k in px.colors.named_colorscales()],
                value=initial_data.get('colormap', 'Viridis')
            ),
            'flex': 1
        }
    ]

    # 应用布局
    app.layout = html.Div([
        dcc.Store(id='data-store', data=initial_data),
        dcc.Store(id='selection-store', data=[]),
        dcc.Store(id='history-store', data=[]),
        
        create_control_group(controls),
        
        html.Div(
            style={'display': 'flex', 'gap': '20px', 'height': '60vh'},
            children=[
                create_graph_container('main-plot'),
                create_graph_container('comparison-plot')
            ]
        ),
        
        html.Div([
            html.Div([
                html.H4("Selection History", className='panel-header'),
                html.Div(id='selection-history', className='scroll-panel')
            ], className='info-panel'),
            
            html.Div([
                html.H4("System Status", className='panel-header'),
                html.Div(id='status-messages', className='scroll-panel')
            ], className='info-panel')
        ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '20px'})
    ], style={'padding': '20px', 'maxWidth': '1600px', 'margin': '0 auto'})

    # 图表更新回调工厂函数
    def create_figure_update_callback(graph_id):
        def callback(basis_type, label_type, colormap, data):
            # 深拷贝原始数据避免污染
            updated_data = copy.deepcopy(data)
            
            # 更新数据参数
            updated_data.update({
                'locs_type': basis_type,
                'current_type': label_type,
                'colormap': colormap
            })
            
            # 生成新图表
            updated_figure = scatter2d(
                updated_data,
                **updated_data.get('scatter_args', {})
            )
            
            # 更新图表样式
            updated_figure.update_layout(
                colorway=[colormap],
                hovermode='closest'
            )
            
            return updated_figure, updated_data
        
        return callback

    # 注册双图回调
    for gid in ['main-plot', 'comparison-plot']:
        app.callback(
            Output(gid, 'figure'),
            Output('data-store', 'data'),
            Input('basis-type-dropdown', 'value'),
            Input('label-type-dropdown', 'value'),
            Input('colormap-selector', 'value'),
            State('data-store', 'data'),
            prevent_initial_call=True
        )(create_figure_update_callback(gid))

    # 通用导出处理
    @app.callback(
        Output('status-messages', 'children'),
        Input('export-main-plot', 'n_clicks'),
        Input('export-comparison-plot', 'n_clicks'),
        State('data-store', 'data'),
        prevent_initial_call=True
    )
    def handle_export(main_clicks, comp_clicks, data):
        ctx = callback_context
        if not ctx.triggered:
            return ""
            
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        graph_name = "Main" if "main" in trigger_id else "Comparison"
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"{graph_name}_Plot_{timestamp}.html"
            
            scatter2d(
                data,
                save=filename,
                **data.get('scatter_args', {})
            )
            
            export_status = html.Div([
                html.I(className="fas fa-check-circle success-icon"),
                f"成功导出 {filename}",
                dcc.Link("立即下载", href=f"/assets/{filename}", className='download-link')
            ], className='status-message')
            
        except Exception as e:
            export_status = html.Div([
                html.I(className="fas fa-times-circle error-icon"),
                f"导出失败: {str(e)}"
            ], className='status-message error')
        
        return export_status

    # 清除选择回调
    @app.callback(
        Output('selection-store', 'data'),
        Input('clear-main-plot', 'n_clicks'),
        Input('clear-comparison-plot', 'n_clicks'),
        prevent_initial_call=True
    )
    def clear_selections(*_):
        return []

    # 运行应用
    app.run_server(debug=debug, **run_args)
    return app



def dash_app1(initial_data, app=None, server=None, debug=True, **run_args):
    import dash
    from dash import Dash, dcc, html, callback_context
    from dash.dependencies import Input, Output, State
    from datetime import datetime
    import pandas as pd
    import copy
    
    # 初始化应用
    if app is None:
        app = Dash(__name__, suppress_callback_exceptions=True)
    if server is None:
        server = app.server

    # =============== 样式配置 ===============
    STYLES = {
        'layout': {
            'padding': '2rem',
            'maxWidth': '1440px',
            'margin': '0 auto',
            'backgroundColor': '#f5f6fa'
        },
        'controlGroup': {
            'display': 'grid',
            'gridTemplateColumns': 'repeat(auto-fit, minmax(240px, 1fr))',
            'gap': '1rem',
            'marginBottom': '1.5rem'
        },
        'graphContainer': {
            'position': 'relative',
            'height': '65vh',
            'borderRadius': '8px',
            'boxShadow': '0 2px 6px rgba(0,0,0,0.1)'
        },
        'floatingControls': {
            'position': 'absolute',
            'top': '1rem',
            'right': '1rem',
            'zIndex': 100,
            'display': 'flex',
            'gap': '0.5rem'
        },
        'iconButton': {
            'width': '36px',
            'height': '36px',
            'borderRadius': '50%',
            'border': 'none',
            'cursor': 'pointer',
            'transition': 'all 0.2s'
        }
    }

    # =============== 工具函数 ===============
    def create_control(label, component, width=1):
        return html.Div(
            className='control-item',
            style={'gridColumn': f'span {width}'},
            children=[
                html.Label(label, className='control-label'),
                component
            ]
        )

    def create_graph(id_suffix):
        return html.Div(
            style=STYLES['graphContainer'],
            children=[
                dcc.Graph(
                    id=f'scatter-plot-{id_suffix}',
                    config={
                        'scrollZoom': True,
                        'displayModeBar': True,
                        'modeBarButtonsToAdd': ['select2d', 'lasso2d'],
                        'displaylogo': False
                    }
                ),
                html.Div(
                    style=STYLES['floatingControls'],
                    children=[
                        html.Button(
                            '🗑️',
                            id=f'clear-{id_suffix}',
                            style={**STYLES['iconButton'], 'backgroundColor': '#ff4757'},
                            title='Clear selection'
                        ),
                        html.Button(
                            '⤵️', 
                            id=f'export-{id_suffix}',
                            style={**STYLES['iconButton'], 'backgroundColor': '#2ed573'},
                            title='Export figure'
                        )
                    ]
                )
            ]
        )

    # =============== 控件配置 ===============
    controls = [
        create_control(
            "Basis Type",
            dcc.Dropdown(
                id='basis-type',
                options=[{'label': k, 'value': k} for k in initial_data["locs"]],
                value=initial_data['locs_type'],
                clearable=False
            ),
            width=2
        ),
        create_control(
            "Label Type",
            dcc.Dropdown(
                id='label-type',
                options=[{'label': k, 'value': k} for k in initial_data["labels"]],
                value=initial_data['current_type'],
                clearable=False
            ),
            width=2
        ),
        create_control(
            "Colormap",
            dcc.Dropdown(
                id='colormap',
                options=[{'label': k.title(), 'value': k} for k in px_colormap()],
                value=initial_data.get('colormap', 'viridis')
            )
        ),
        create_control(
            "Point Size",
            dcc.Slider(
                id='point-size',
                min=1,
                max=20,
                step=1,
                value=initial_data['scatter_args'].get('scale', 5),
                marks={i: str(i) for i in range(0, 21, 5)}
            )
        )
    ]

    # =============== 应用布局 ===============
    app.layout = html.Div(
        style=STYLES['layout'],
        children=[
            dcc.Store(id='data-store', data=initial_data),
            dcc.Store(id='selection-store', data={'main': [], 'comparison': []}),
            dcc.Store(id='history-tracker', data=[]),
            
            html.Div(
                style=STYLES['controlGroup'],
                children=controls
            ),
            
            html.Div(
                className='graph-grid',
                style={
                    'display': 'grid',
                    'gridTemplateColumns': '1fr 1fr',
                    'gap': '2rem',
                    'marginBottom': '2rem'
                },
                children=[
                    create_graph('main'),
                    create_graph('comparison')
                ]
            ),
            
            html.Div(
                className='action-panel',
                children=[
                    html.Div(
                        className='input-group',
                        children=[
                            dcc.Input(
                                id='label-input',
                                placeholder='Enter new label...',
                                type='text',
                                style={'flex': 1}
                            ),
                            html.Button(
                                'Apply Label', 
                                id='apply-label',
                                className='action-btn primary'
                            ),
                            html.Button(
                                'Undo', 
                                id='undo-action',
                                className='action-btn secondary'
                            ),
                            html.Button(
                                'Export All', 
                                id='export-all',
                                className='action-btn success'
                            )
                        ]
                    )
                ]
            )
        ]
    )

    # =============== 回调工厂 ===============
    def create_graph_update_callback(graph_id):
        def callback(basis_type, label_type, colormap, point_size, data):
            updated_data = copy.deepcopy(data)
            
            # 更新参数
            updated_data.update({
                'locs_type': basis_type,
                'current_type': label_type,
                'colormap': colormap,
                'scatter_args': {
                    **updated_data.get('scatter_args', {}),
                    'scale': point_size
                }
            })
            
            # 生成图表
            fig = scatter2d(updated_data, **updated_data['scatter_args'])
            fig.update_layout(
                colorway=[colormap],
                hovermode='closest',
                plot_bgcolor='rgba(240,240,240,0.1)'
            )
            
            return fig, updated_data
        
        return callback

    # 注册双图回调
    for gid in ['main', 'comparison']:
        app.callback(
            Output(f'scatter-plot-{gid}', 'figure'),
            Output('data-store', 'data', allow_duplicate=True),
            Input('basis-type', 'value'),
            Input('label-type', 'value'),
            Input('colormap', 'value'),
            Input('point-size', 'value'),
            State('data-store', 'data'),
            prevent_initial_call=True
        )(create_graph_update_callback(gid))

    # =============== 交互回调 ===============
    @app.callback(
        Output('selection-store', 'data'),
        Input('scatter-plot-main', 'selectedData'),
        Input('scatter-plot-comparison', 'selectedData'),
        State('selection-store', 'data'),
        prevent_initial_call=True
    )
    def update_selections(main_selection, comp_selection, current):
        ctx = callback_context
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        new_data = copy.deepcopy(current)
        if 'main' in trigger_id:
            new_data['main'] = [p['customdata'][0] for p in (main_selection['points'] if main_selection else [])]
        elif 'comparison' in trigger_id:
            new_data['comparison'] = [p['customdata'][0] for p in (comp_selection['points'] if comp_selection else [])]
        
        return new_data

    # =============== 数据操作回调 ===============
    @app.callback(
        Output('data-store', 'data', allow_duplicate=True),
        Output('history-tracker', 'data'),
        Input('apply-label', 'n_clicks'),
        State('label-input', 'value'),
        State('selection-store', 'data'),
        State('data-store', 'data'),
        State('history-tracker', 'data'),
        prevent_initial_call=True
    )
    def apply_labels(_, new_label, selections, data, history):
        if not new_label or not any(selections.values()):
            raise dash.exceptions.PreventUpdate
        
        updated_data = copy.deepcopy(data)
        current_type = updated_data['current_type']
        affected_points = list(set(selections['main'] + selections['comparison']))
        
        # 更新标签
        for idx in affected_points:
            updated_data['labels'][current_type][idx] = new_label
        
        # 记录历史
        new_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': 'label_update',
            'details': {
                'label': new_label,
                'points': affected_points,
                'type': current_type
            }
        }
        
        return updated_data, history + [new_entry]

    # =============== 运行应用 ===============
    app.run_server(debug=debug, **run_args)
    return app