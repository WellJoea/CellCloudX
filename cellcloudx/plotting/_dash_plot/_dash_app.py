import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import time
import sys
from typing import Optional, Union, Dict
import copy 

from .._colors import *
from ._config import STYLES
from ._dash_function import *
from ...utilis._arrays import list_iter, vartype

def update_paras(paras, **karges):
    paras.update({
        'figscale':800,
        'window_height': 800,
        'commonbar_state': True,
        'windowbar_state': True,
    })
    paras.update(karges)
    return paras

def update_data_window_iterms(paras, window_number=None):
    if window_number is None:
        window_number = paras['window_number']
    iterms = ['window_labels', 'window_splits', 'split_iterms',  'window_slices',
                'point_samples', 'window_renders', 'points_size', 'points_scale', 'sizes_max', 'window_basis',
                'save_figs','colormaps', 'scatter_args']
    for ii in iterms:
        iidat = paras[ii]
        if ii =='scatter_args':
            iidat = paras[ii]
            paras[ii] = [ copy.deepcopy(list_iter(iidat, l2d=True)[k]) for k in range(window_number) ]
        else:
            paras[ii] = [ list_iter(iidat, l2d=True)[k] for k in range(window_number) ]
    return paras

def get_selectidex(selectData, contain=False, locs=None,):
    if contain:
        from shapely.geometry import Point, Polygon
        # from matplotlib.path import Path
        new_indices = set({})
        xyz = list('xyz')[:locs.shape[1]]
        index = np.arange(locs.shape[0])
        for idata in selectData:
            if idata is not None:
                if 'range' in idata:
                    select = []
                    for k in xyz:
                        irange = idata['range'].get(k)
                        isel = ((locs[k]>= irange[0]) and (locs[k]<= irange[1]))
                        select.append(isel)
                    select = set(index[np.all(select, axis=0)])
                    new_indices = new_indices | select
                elif 'lassoPoints' in idata:
                    xyz_range = idata['range']
                    polygon = Polygon(list(zip([xyz_range[k] for k in xyz])))
                    select = [ polygon.contains(Point(*ipt)) for ipt in locs ]
                    select = set(index[select])
                    # polygon = Path(list(zip([xyz_range[k] for k in xyz])))
                    # select = [ polygon.contains_point(ipt) for ipt in locs ]
                    new_indices = new_indices | select
        new_indices = sorted(list(new_indices))
    else:
        new_indices = [ [int(point['customdata'][0]) for point in idata.get('points', [])] 
                            for idata in selectData if idata is not None ]

def dash_app(initial_data, initial_paras,  app=None, server=None, 
             suppress_callback_exceptions=True,
             debug=False, **run_args):
    import dash
    from dash import Dash, dcc, html, callback_context, Input, Output, State
    from dash.dependencies import Input, Output, State
    import dash_bootstrap_components as dbc

    if app is None:
        # app = Dash(__name__, )
        app = Dash(__name__, 
                   suppress_callback_exceptions=suppress_callback_exceptions,
                   external_stylesheets=[
                        dbc.themes.BOOTSTRAP, 
                        dbc.icons.BOOTSTRAP,
                        '/assets/button_styles.css',
                        '/assets/styles.css',
                  ])
    if server is None:
        server = app.server

    app.enable_dev_tools(
        dev_tools_hot_reload=True,
        dev_tools_hot_reload_interval=1000  # 1秒检测一次变更
    )


    initial_paras = update_paras(initial_paras)
    initial_paras = update_data_window_iterms(initial_paras)
    Window_Number = initial_paras['window_number']

    # =============== app layout ===============
    app.layout = html.Div([
        dcc.Store(id='data-store', data={'index': initial_data['index']}),
        dcc.Store(id='paras-store', data=initial_paras),
        dcc.Store(id='selection-store', data=[]),
        dcc.Store(id='current-record', data=[]),
        dcc.Store(id='history-store', data=[]),
        dcc.Store(id='history-tracker', data=[]),
        # dcc.Store(id="graph-widths", data={"graph1": 50, "graph2": 50}),

        html.Div(
            id="sidebar",
            style=STYLES['siderbar'],
            className="sidebar_lstyle ",
            children=[
                html.Div([
                    html.H4("Parameters", className="display-8", style={"textAlign": "center"}),
                    html.Button("◀", id="collapse-button", 
                                 className="btn btn-secondary collapse_button_lstyle",
                                )
                ]),
                html.Hr(),

                ## common control
                html.Div(
                    id= "commonbar_tune",
                    style={'width':'100%', 'transition': 'all 0.2s'},
                    children=create_menu_common(initial_paras, tune=True),
                ),
                html.Hr(),
        
                # windown panel control
                html.Div(
                    id= "windows_tune",
                    style={'width':'100%', "transition": "all 0.2s"},
                    children=create_menu_window(initial_paras, tune=True),
                ),
                html.Hr(),

        ]),

        html.Button(
            "▶",
            id="expand-button",
            className="btn btn-secondary expand_button_lstyle",
        ),

        # Graph container
        html.Div(
            # className='contents',
            id="contents",
            style=STYLES['contents'],
            children=[
                html.Div(
                    id="graph-container",
                    style={
                        # # "height": "calc(100vh - 4rem)",
                        # 'width': '100%',
                        # "background-color": "#fff",
                        # "border-radius": "8px",
                        # "box-shadow": "0 4px 8px rgba(0, 0, 0, 0.17)",
                        "display": "flex",
                        'marginTop': '0rem',
                        'borderRadius': '1px 1px 5px 5px',
                        'border': '0px solid #ddd', 
                        # "minHeight": "100px", "minWidth": "100px", # 设置最小宽高 TODO 
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)", 
                        # "overflow": "hidden"
                    },
                    children = create_graphs(initial_paras, data=initial_data),
                ),
            ],
        ),

        # Action panel
        html.Div(
            className='action-panel',
            children=[
                create_action(initial_paras),
                create_record(initial_paras),
            ]
        ),
    ])

    #selection store
    @app.callback(
        Output('selection-store', 'data'),
        Output('selected-points-display', 'children'),
        Input('add-selection-0', 'n_clicks'),
        Input('delete-selection-0', 'n_clicks'),
        Input('clear-0', 'n_clicks'),
        Input('upload-0', 'n_clicks'),
        State('upload-input-0', 'value'),
        State('selection-store', 'data'),
        [ State(f'scatter-plot-{idx}', 'selectedData')
          for idx in range(Window_Number) ],
        prevent_initial_call=True
    )
    def update_selection(add_clicks, delete_clicks, clear_clicks, upload_clicks,
                          upload_txt, current_data, *selectData):
        ctx = callback_context
        if not ctx.triggered:
            # if not current_data:
            #     return [], html.Span("No points selected", style={'color': '#999'})
            raise dash.exceptions.PreventUpdate
        
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if trigger_id == 'add-selection-0':
            new_indices = [ [int(point['customdata'][0])  for point in idata.get('points', [])] 
                            for idata in selectData if idata is not None ] #TODO contain
            new_indices = list(set(sum(new_indices,[])))
            current_data.append(new_indices)
            # current_data = list(set(sum(new_indices,current_data)))

        elif trigger_id == 'delete-selection-0':
            current_data = current_data[:-1]
        elif trigger_id == 'clear-0':
            current_data = []
        elif trigger_id == 'upload-0':
            if (upload_txt is not None) and (upload_txt != ''):
                new_indices = [int(i) for i in upload_txt.split(',') if i.isdigit()]
                current_data.append(new_indices)

        display_data = list(set(sum(current_data,[])))
        selected_count = len(display_data)
        sample_points = sorted(display_data)  # 显示所有选中点

        if selected_count:
            sample_text = ', '.join(map(str, sample_points))
            display_data = html.Div([
                html.Span(f"Selected Points ({selected_count}): ", 
                        style={'fontWeight': 'bold', 'color': '#333'}),
                html.Span(sample_text, style={'color': '#666'})
            ])
            disabled = [ False, {'backgroundColor': 'red', 'cursor': 'pointer'} ] #TODO
        else:
            display_data = html.Span("No points selected", style={'color': '#999'})
            disabled = [ True, {'backgroundColor': 'grey', 'cursor': 'not-allowed'} ]

        return current_data, display_data

    # clean edit data
    @app.callback(
        Output('edit-input-0', 'value'),
        Input('edit-0', 'n_clicks'),
        State('edit-input-0', 'value'),
    )
    def clean_edit_inpute(edit_clicks, edit_input):
        ctx = callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate
        if edit_clicks:
            return None

    # update data
    @app.callback(
        Output('data-store', 'data'),
        Output('history-div', 'children'),

        Input('update-0', 'n_clicks'),
        Input('Undo-0', 'n_clicks'),
        Input('Redo-0', 'n_clicks'),

        State('edit-label-0', 'value'),
        State('edit-input-0', 'value'),
        State('label-input', 'value'),
        State('selection-store', 'data'),
        State('current-record', 'data'),
        State('history-store', 'data'),

        State('history-tracker', 'data'),
        State('data-store', 'data'),

    )
    def update_data(selected_type, split_type, split_iterm, n_clicks, new_label, selection_store, data):
        ctx = callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate
    
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
            
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
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
    
    
    # slide bar
    @app.callback(
        [Output("sidebar", "style"),
        Output("contents", "style"),
        Output("expand-button", "style")],
        [Input("collapse-button", "n_clicks"),
        Input("expand-button", "n_clicks")],
        [State("sidebar", "style"),
        State("contents", "style")],
        prevent_initial_call=True
    )
    def toggle_sidebar(collapse_clicks, expand_clicks, sidebar_style, content_style):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate
        
        # if not ctx.triggered:
        #     return sidebar_style, content_style, STYLES['expand_button_lstyle']

        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if button_id == "collapse-button": 
            sidebar_style = STYLES['hideden_siderbar']
            content_style = STYLES['hideden_contents']
            expand_button_lstyle = {**STYLES['expand_button_style'], "display": "flex"}
            return sidebar_style, content_style, expand_button_lstyle
        elif button_id == "expand-button":  # 展开侧边栏
            sidebar_style = STYLES['siderbar']
            content_style = STYLES['contents']
            expand_button_lstyle = STYLES['expand_button_style']
            return sidebar_style, content_style, expand_button_lstyle
        else:
            raise dash.exceptions.PreventUpdate

    # common bar
    @dash.callback(
        Output("commonbar_tune", "children"),
        Output("paras-store", "data"),
        Input("common-bi", "n_clicks"),
        State("paras-store", "data"),
        prevent_initial_call=True
    )
    def toggle_common( c_clicks, paras,):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if trigger_id in ["common-bi", "common-close-bi"]:
            is_open = not paras['commonbar_state']      
            paras['commonbar_state'] = is_open
            return create_menu_common(paras, tune=is_open), paras
        else:
            raise dash.exceptions.PreventUpdate

    @dash.callback(
        Output("contents", "children", allow_duplicate=True),
        Output("paras-store", "data", allow_duplicate=True),
        Input("figure-scale", "value"),
        Input("window-height", "value"),
        State("paras-store", "data"),
        prevent_initial_call=True
    )
    def common_para( figscale, window_height, paras,):
        ctx = callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate
        triggered = ctx.triggered[0]['prop_id'].split('.')[0]
        if triggered == 'figure-scale':
            paras['figscale'] = figscale
            return create_graphs(paras, data=initial_data), paras
        
        elif triggered == 'window-height':
            paras['window_height'] = window_height
            return create_graphs(paras, data=initial_data), paras
        else:
            raise dash.exceptions.PreventUpdate

    @dash.callback(
        Output("contents", "children", allow_duplicate=True),
        Output("windows_tune", "children", allow_duplicate=True),
        Output("paras-store", "data", allow_duplicate=True),
        Input("window-number-slider", "value"),
        State("paras-store", "data"),
        prevent_initial_call=True
    )
    def common_para0( window_number, paras,):
        ctx = callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate

        triggered = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if triggered == 'window-number-slider':
            Window_Number = window_number
            paras['window_number'] = window_number
            paras = update_data_window_iterms(paras, window_number=window_number)
            return [create_graphs(paras, data=initial_data),  
                    create_menu_window(paras), 
                    paras]
        else:
            raise dash.exceptions.PreventUpdate

    # windows bar
    @dash.callback(
        Output("windows_tune", "children"),
        Output("paras-store", "data", allow_duplicate=True),
        Input("windows-bi", "n_clicks"),
        State("paras-store", "data"),
        prevent_initial_call=True
    )
    def toggle_windows( c_clicks, paras):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if trigger_id in ["windows-bi"]:
            is_open = not paras['windowbar_state']      
            paras['windowbar_state'] = is_open
            return create_menu_window(paras, tune=is_open), paras
        else:
            raise dash.exceptions.PreventUpdate

    for idx in range(Window_Number):
        @dash.callback(
            Output(f'scatter-plot-{idx}', "figure", allow_duplicate=True),
            Output("paras-store", "data", allow_duplicate=True),
            [
             Input(f'basis-type-{idx}', "value"),
             Input(f"label-type-{idx}", "value"),             
             Input(f'split-iterm-{idx}', "value"),

             Input(f"animation-type-{idx}", "value"),
             Input(f"point-size-{idx}", "value"),
             Input(f"point-scale-{idx}", "value"),
             Input(f"size-max-{idx}", "value"),

             Input(f'colormap-{idx}', "value"),
             Input(f"render-panel-{idx}", "value"),             
             Input(f'point-sample-{idx}', "value"),
             Input(f"contain-hidden-{idx}", "value"), #TODO
             Input(f"legend-switch-{idx}", "value"),
            ],
            State("paras-store", "data"),
            prevent_initial_call=True
        )
        def tune_window( wdb, llt, sti, ant, ptz, ptc, szm, clp ,
                         rrp, pts, cnh, lds, paras, idx=idx):
            ctx = dash.callback_context
            if not ctx.triggered:
                raise dash.exceptions.PreventUpdate
            
            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if trigger_id in [f'basis-type-{idx}']:
                paras['window_basis'][idx] = wdb
                return create_figure(paras, idx, data=initial_data), paras

            elif trigger_id in [f'label-type-{idx}']:
                paras['window_labels'][idx] = llt
                return create_figure(paras, idx, data=initial_data), paras

            elif trigger_id in [f'split-iterm-{idx}']:
                paras['split_iterms'][idx] = sti #clean==[]
                return create_figure(paras, idx, data=initial_data), paras

            elif trigger_id in [f'animation-type-{idx}']:
                paras['window_slices'][idx] = ant
                return create_figure(paras, idx, data=initial_data), paras

            elif trigger_id in [f'point-size-{idx}']:
                paras['points_size'][idx] = ptz
                return create_figure(paras, idx, data=initial_data), paras

            elif trigger_id in [f'point-scale-{idx}']:
                paras['points_scale'][idx] = ptc
                return create_figure(paras, idx, data=initial_data), paras

            elif trigger_id in [f'size-max-{idx}']:
                paras['sizes_max'][idx] = szm
                return create_figure(paras, idx, data=initial_data), paras

            elif trigger_id in [f'colormap-{idx}']:
                paras['colormaps'][idx] = clp
                return create_figure(paras, idx, data=initial_data), paras
        
            elif trigger_id in [f'render-panel-{idx}']:
                paras['window_renders'][idx] = rrp
                return create_figure(paras, idx, data=initial_data), paras

            elif trigger_id in [f'point-sample-{idx}']:
                paras['point_samples'][idx] = pts
                return create_figure(paras, idx, data=initial_data), paras

            elif trigger_id in [f'contain-hidden-{idx}']:
                pass #TODO #cnh

            elif trigger_id in [f"legend-switch-{idx}"]:
                paras['scatter_args'][idx]['showlegend'] = lds
                return create_figure(paras, idx, data=initial_data), paras
            else:
                raise dash.exceptions.PreventUpdate
    
    for idx in range(Window_Number):
        @dash.callback(
            Output("paras-store", "data", allow_duplicate=True),
            Output(f'split-iterm-{idx}', 'options'),
            [           
             Input(f'split-type-{idx}', "value"),
            ],
            State("paras-store", "data"),
            prevent_initial_call=True
        )
        def tune_split(stt,  paras, idx=idx):
            ctx = dash.callback_context
            if not ctx.triggered:
                raise dash.exceptions.PreventUpdate
            
            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

            if trigger_id in [f'split-type-{idx}']:
                paras['window_splits'][idx] = stt

                alabels = paras['colors_order'].get(stt, [])
                options = [{'label': 'NONE', 'value': 'NONE'}] + \
                          [{'label': str(label), 'value': label} for label in alabels]
                return paras, options
            else:
                raise dash.exceptions.PreventUpdate

    app.run_server(debug=debug, **run_args)
    return 

