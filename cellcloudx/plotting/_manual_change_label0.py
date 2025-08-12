import numpy as np
import pandas as pd
import scanpy as sc
import plotly.express as px 
from typing import Optional, Union
import colorsys

def manual_label_wrap(adata, basis, current_type, split_type, 
                      colors = {},
                      scatter_args={}, run_args={},
                      **karges):
    color_init = { k: adata.uns.get(f'{k}_colors', None) for k in adata.obs.columns }
    color_init.update(colors)
            
    return manual_label(locs=adata.obsm[basis], labels=adata.obs, 
                        colors=color_init, 
                        current_type = current_type,
                        split_type =split_type,
                        scatter_args=scatter_args, 
                        run_args=run_args, **karges)

def manual_label(locs: np.ndarray = None, labels : pd.DataFrame=None, 
                  colors :dict ={}, 
                  current_type : Union[str, int]=None,
                  split_type : Union[str,]=None,
                  initial_data : dict=None, scatter_args :dict ={},
                  debug :bool =True, 
                  run_args :dict ={}):
    initial_data = get_init_data(locs=locs, labels=labels,
                                colors=colors,
                                current_type =current_type,
                                split_type =split_type,
                                scatter_args=scatter_args)
    dash_app(initial_data,  debug=debug,  **run_args)

def get_init_data(locs: np.ndarray = None, labels : pd.DataFrame=None, 
                  colors :dict ={}, 
                  current_type : Union[str, int]=None,
                  split_type : Union[str,]=None,
                  initial_data : dict=None, scatter_args :dict ={}) -> dict:
    if initial_data is None: 
        labels = pd.DataFrame(labels).copy()
        current_type = current_type or labels.columns[0]
        if split_type is None:
            if len(labels.columns) > 1:
                split_type = labels.columns[0]
            else:
                split_type = current_type
        initial_data = {
            'locs' : np.asarray(locs),
            "labels": { i:labels[i] for i in labels.columns },
            "current_type": current_type or labels.columns[0],
            'split_type': split_type,
            'split_iterm': 'ALL_',
            "history": [],
            'undo_stack': [],
            'redo_stack': [],
            'colors': colors or {},
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
                    colormap='Viridis', template='none',
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

def scatter2d(IData : dict, group=None, split=None, split_iterm=None,
               scale=1,  same_scale=True, 
               size=None, opacity=None,
               figscale=800, werror=20, 
                width = None, height=None, scene_aspectmode='data',  
                reverse_y=False, reverse_x=False,  
                showlegend=True,
                scene_aspectratio=dict(x=1, y=1, z=1), clip=None, sample=None,
                random_state = 200504, order ='guess',
                show_grid =True, 
                showticklabels = False,
                colordict = {},
                colormap='Viridis', template='none',
                itemsizing='constant',
                itemwidth=None,
                legdict={},
                  **kargs):

    group = group or IData.get('current_type')
    split = split or IData.get('split_type')
    split_iterm = split_iterm or IData.get('split_iterm')

    Data = pd.DataFrame(IData['labels'])
    Data.index = IData['index']
    Data[list('xy')] =  np.array(IData['locs'])[:,:2]
    Data['index'] = Data.index
    Data['custom_index'] = np.arange(len(Data)).astype(str)
    Colors = IData.get('colors', {})
    Colors.update(colordict)

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

    if not group is None:
        ctype   = vartype(Data[group])
        dimdict['color']=group
        if ctype == 'discrete':
            try:
                order = Data[group].cat.categories.tolist()
            except:
                order = Data[group].unique().tolist()

            order = Data[group].unique().tolist()
            colormap = Colors.get(group, color_palette(len(order)))
            category_orders={group: order}
            
            dimdict.update({'category_orders': category_orders, 
                            'color_discrete_sequence':colormap}) #'animation_frame': group
        elif ctype == 'continuous':
            if not clip is None:
                Data[group] = np.clip(Data[group], clip[0], clip[1])
            if order =='guess' or  order == True:
                Data = Data.sort_values(by = group, ascending=True)
            colormap = Colors.get(group, colormap)
            dimdict.update({'color_continuous_scale':colormap})

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
                        dragmode='select'  # select tools
                        ) #
                        #margin=dict(l=20, r=20, t=20, b=20),template='simple_white', 
                        #paper_bgcolor='#000000',
                        #plot_or='#000000'
                        #fig.update_xaxes(visible=False, showticklabels=False)

    fig.update_traces(text=[{"index": Data.index}])
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
    return fig

def vartype(vector):
    if type(vector) in [pd.Series,  pd.DataFrame]:
        if vector.dtype in ['float32', 'float64', 'float', 'int32', 'int64', 'int']:
            return 'continuous'
        elif vector.dtype in ['category', 'object', 'bool', pd.CategoricalDtype]:
            return 'discrete'
        else:
            raise('cannot judge vector type.')

    elif isinstance(vector, np.ndarray):
        if np.issubdtype(vector.dtype, np.floating) or np.issubdtype(vector.dtype, np.integer):
            return 'continuous'
        elif np.issubdtype(vector.dtype, str) or \
                np.issubdtype(vector.dtype, np.character) or \
                np.issubdtype(vector.dtype, np.object_) or \
                isinstance(vector.dtype, ( np.object_, object)):
            return 'discrete'
        else:
            raise('cannot judge vector type.')
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

    return palette

def generate_color(index=1, total=1):
    hue = (index / total) if total > 0 else 0
    r, g, b = colorsys.hls_to_rgb(hue, 0.75, 0.6)
    return '#{:02X}{:02X}{:02X}'.format(int(r*255), int(g*255), int(b*255))

def get_contrasting_text_color(bg_hex):
    bg_hex = bg_hex.lstrip('#')
    r, g, b = int(bg_hex[0:2], 16), int(bg_hex[2:4], 16), int(bg_hex[4:6], 16)
    brightness = (r * 299 + g * 587 + b * 114) / 1000
    return "#FFFFFF" if brightness < 128 else "#000000"

def dash_app(initial_data, debug=True, **run_args):
    import dash
    from dash import Dash, dcc, html, callback_context
    from dash.dependencies import Input, Output, State
    from datetime import datetime
    import copy

    app = Dash(__name__)
    server = app.server
    app.layout = html.Div([
        dcc.Store(id='data-store', data=initial_data),
        dcc.Store(id='selection-store', data=[]),
        
        html.Div([
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
                html.Label('Point Size:', style={'fontWeight': 'bold'}),
                dcc.Input(id='point-size-input', type='number', 
                        value=initial_data['scatter_args']['scale'], 
                        style={'width': '95%'}),
                # html.Button('Update Point Size', id='update-point-size-button', 
                #             style={'backgroundColor': '#673AB7', 'color': 'white'})
            ], style={'flex': '1', 'backgroundColor': 'white', 'marginRight': '10px'}),
        
            html.Div([
                html.Label('Figure Scale:', style={'fontWeight': 'bold'}),
                dcc.Input(id='figure-scale-input', type='number', 
                        value=initial_data['scatter_args']['figscale'], 
                        style={'width': '95%'}),
                # html.Button('Update Point Size', id='update-point-size-button', 
                #             style={'backgroundColor': '#673AB7', 'color': 'white'})
            ], style={'flex': '1', 'backgroundColor': 'white', 'marginRight': '10px'}),

        ], style={'display': 'flex', 'marginBottom': '20px', 'width': '100%'}),

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
                html.Button('Download CSV', id='download-button', 
                        style={'marginRight': '10px', 'backgroundColor': '#FF9800', 'color': 'white'}),

                html.Button('Undo', id='undo-button', 
                            style={'marginRight': '10px', 'backgroundColor': '#607D8B', 'color': 'white'}),
                html.Button('Redo', id='redo-button', 
                            style={'marginRight': '10px', 'backgroundColor': '#607D8B', 'color': 'white'}),

        ], style={'display': 'flex', 'marginBottom': '20px', 'width': '100%'}),


        html.Div([
            dcc.Input(id='filepath-input', type='text', 
                    placeholder='Save path (e.g.: ./saved_labels.csv)',
                    style={'width': '80%', 'marginRight': '10px'}),
            html.Button('Save File', id='save-button', 
                    style={'marginRight': '5px', 'backgroundColor': '#9C27B0', 'color': 'white'})
        ], style={'display': 'flex', 'marginBottom': '20px', 'width': '100%'}),
        
        dcc.Graph(id='scatter-plot', figure=scatter2d(initial_data, **initial_data['scatter_args'])),
        
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
        
        # downoad csv
        dcc.Download(id="download-dataframe-csv"),
        
        # save status
        html.Div(id='save-status', style={'marginTop': '10px', 'width': '100%'})
    ], style={'padding': '20px', 'maxWidth': '120000px', 'margin': '0 auto', 'width': '98%'})


    @app.callback(
        Output('scatter-plot', 'figure', allow_duplicate=True),
        Output('data-store', 'data', allow_duplicate=True),
        # Input('update-point-size-button', 'n_clicks'),
        Input('point-size-input', 'value'), 
        State('data-store', 'data'),
        prevent_initial_call=True
    )
    def update_plot(new_size, data):
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
    def update_plot(new_size, data):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate

        if new_size is None or new_size <= 0:
            raise dash.exceptions.PreventUpdate

        data['scatter_args']['figscale'] = new_size
        print(data['scatter_args'], new_size)
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


    # 回调1：更新选中点存储
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
        Output("download-dataframe-csv", "data"),
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
        return dcc.send_data_frame(df.to_csv, "labels.csv", index=True)

    # 回调4：文件保存
    @app.callback(
        Output("save-status", "children"),
        Input("save-button", "n_clicks"),
        State("filepath-input", "value"),
        State("data-store", "data"),
        prevent_initial_call=True
    )
    def save_file(n_clicks, file_path, data):
        if not file_path:
            return html.Span("Please specify a valid file path!", style={'color': 'red'})
        
        try:
            # df = pd.DataFrame({
            #     "x": data["x"],
            #     "y": data["y"],
            #     "label": data["labels"][data["current_type"]]
            # }, index=data.get('index', np.arange(len(data['x']))))
            df = pd.DataFrame(data["labels"], index=data['index'])
            df.to_csv(file_path, index=True)
            return html.Span(f"Successfully saved to: {file_path}", 
                            style={'color': 'green', 'backgroundColor': '#fff',})
        except Exception as e:
            return html.Span(f"Save failed: {str(e)}", style={'color': 'red'})

    app.run_server( debug=debug, **run_args)
    return app
