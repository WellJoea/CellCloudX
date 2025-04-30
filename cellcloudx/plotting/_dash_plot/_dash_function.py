import pandas as pd
import numpy as np
import time
import dash
from dash import Dash, dcc, html, callback_context, Input, Output, State
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

from ._plot_wrap import scatter_wrap
from .._colors import *

None_options = [{'label': 'None', 'value': 'None'}]

def create_record(paras):
    return dbc.Row([
        # Current Selection
        html.Div([
            html.H5("Current Selection:",
                    className='title_font_lstyle',
                    style={'backgroundColor': 'white', 'width': '20%', 'marginBottom': '10px'}),
            html.Div(id='selected-points-display', 
                    style={'maxHeight': '200px', 'minHeight': '40px', 'overflowY': 'auto', 
                        'backgroundColor': '#fff',
                        'border': '1px solid #ddd', 'borderRadius': '5px', 'width': '100%'})
        ], style={'marginTop': '20px', 'width': '100%'}),
        
        # History Records
        html.Div([
            html.H5("History Records:", 
                    className='title_font_lstyle',
                    style={'backgroundColor': 'white', 'width': '40%', 'marginBottom': '10px'}),
            html.Div(id='history-div', 
                    style={'maxHeight': '200px', 'minHeight': '50px',
                           'overflowY': 'auto', 
                            'backgroundColor': '#fff',
                            'border': '1px solid #ddd', 'borderRadius': '5px', 'width': '100%'})
        ], style={'marginTop': '20px', 'width': '100%'}),
    ],
    style={
            # "display": "flex", #TODO grid
            # 'width': '100%',
            'border': '2px solid #ddd', 'borderRadius': '1px 1px 5px 5px',
            "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
            "margin-left": "0.5rem",
            "margin-right": "0.5rem",
            "padding": "0rem",
            "transition": "margin-left 0.5s ease",
            "background-color": "#fdfdfd",
        } 
    )

def create_action(paras):
    return dbc.Row(
        children=action_tune(paras, idx=0),
        className="g-0 bg-white rounded-3 shadow-sm",  ## 默认允许换行
        style={
            # "display": "flex", #TODO grid
            # 'width': '100%',
            'border': '2px solid #ddd', 'borderRadius': '1px 1px 5px 5px',
            "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
            "margin-left": "0.5rem",
            "margin-right": "0.5rem",
            "padding": "0rem",
            "transition": "margin-left 0.5s ease",
            "background-color": "#fdfdfd",
        } 
    )

def action_tune(paras, idx):
    # https://unicode-explorer.com/emoji/
    row1 = [
            dbc.Button(
                '➕',
                id=f'add-selection-{idx}',
                className="custom-active-button iconButton_lstyle ",
                style={'backgroundColor': '#D39986'},
                title='Add selection'
            ),
            dbc.Button(
                '➖',
                id=f'delete-selection-{idx}',
                style={'backgroundColor': '#E8D5AD'},
                disabled=False,
                className="custom-active-button iconButton_lstyle ",
                title='Delete selection'
            ),
            dbc.Button(
                '🗑️',
                id=f'clear-{idx}',
                style={'backgroundColor': '#FAE43B'},
                title='Clear selection',
                disabled=False,
                className="custom-active-button iconButton_lstyle ",
            ),
            html.Div([
                    dbc.Button(
                        '📝',
                        id=f'edit-{idx}',
                        style={ #'display': 'flex',  
                               'flexShrink': 0, 'flexGrow': 0, 
                               'backgroundColor': '#2ed573'},
                        disabled=False,
                        className="custom-active-button iconButton_lstyle ",
                        title='Edit selection',
                    ),
                    dbc.Input(id=f'edit-input-{idx}', type='text',
                               placeholder='Enter new label...',
                            #    clearable=True, TODO
                               style={'width': '15rem', 'height': '2.45rem',}
                    ),
                    dcc.Dropdown(
                        id=f'edit-label-{idx}',
                        options=[{'label': k, 'value': k} for k in paras["data_labels"]],
                        value=paras['window_labels'][idx],
                        clearable=True,
                        searchable=True,
                        style={ 'width': '10rem', 'height': '2.5rem', "zIndex": 1e6, }
                    ),
                ], style={#'flex': '3', 'minWidth': '50px', 'marginRight': '10px',
                        #  'minWidth': '50rem',
                         'display': 'flex', 
                         'alignItems': 'center', 
                         'gap': '0.25rem',
                         'flexWrap': 'nowrap',
                         'padding':'0',
                         'backgroundColor': 'white'}
            ),
            dbc.Button(
                '🔄',
                id=f'update-{idx}',
                style={'backgroundColor': '#A831BC'},
                disabled=False,
                className="custom-active-button iconButton_lstyle ",
                title='Update selection'
            ),
            dbc.Button(
                '↩️', 
                id=f'Undo-{idx}',
                style={'backgroundColor': '#A831BC'},
                disabled=False,
                className="custom-active-button iconButton_lstyle ",
                title='Undo'
            ),
            dbc.Button(
                '↪️', 
                id=f'Redo-{idx}',
                style={'backgroundColor': '#A831BC'},
                disabled=False,
                className="custom-active-button iconButton_lstyle ",
                title='Redo'
            ),
            dbc.Button(
                '🗃️', 
                id=f'save-{idx}',
                style={'backgroundColor': '#475D97'},
                disabled=False,
                className="custom-active-button iconButton_lstyle ",
                title='save csv'
            ),
            dbc.Button(
                '🗂️', 
                id=f'save-as-{idx}',
                style={'backgroundColor': '#475D97'},
                disabled=False,
                className="custom-active-button iconButton_lstyle ",
                title='save as csv'
            ),            
        ]
    row2 =[
        html.Div([
            dbc.Button(
                '📤',
                id=f'upload-{idx}',
                style={'backgroundColor': '#AAAAAA'},
                disabled=False,
                className="custom-active-button iconButton_lstyle ",
                title='upload selection'
            ),
            dbc.Input(id=f'upload-input-{idx}', type='text',
                        placeholder='Enter new index, (0,1,2,3)...',
                        # clearable=True,
                        style={'width': '80%', 'height': '2.45rem', 'marginLeft': '0.25rem' }),
        ], style={#'flex': '3', 'minWidth': '50px', 'marginRight': '10px',
                    'minWidth': '408px',
                    'display': 'flex',
                    'gap': '0.25rem',
                    # 'flexWrap': 'nowrap',
                    'backgroundColor': 'white'}
        ),
    ]
    style1={
            # 'position': 'fixed',    
            'top': '1rem',           
            'right': '1rem',       
            'zIndex': 1e5,       
            'display': 'flex',   
            'gap': '0.5rem',
            "padding": "0.3rem 0.3rem",
    }
    # rows = [ dbc.Row(html.Div(irow, style=style1)) for irow in [row1, row2] ] #Two rows
    rows = html.Div(sum([row1, row2], []), style=style1)
    return rows

def create_menu_common(paras, tune=None):
    if tune is None:
        tune = True #TODO
    col_children = [
                tune_controls(
                    f"window number",
                    dcc.Slider(
                            id="window-number-slider",
                            min=1,
                            max=paras['max_window_number'],
                            step=1,
                            value=paras['window_number'],
                            # marks={i: str(i) for i in range(1, 21, 2)},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                    width=12
                ),
                tune_controls(
                    f"figure scale",
                    dcc.Input(id='figure-scale', type='number',
                        min=300, 
                        # step=100,
                        value=paras['figscale'], 
                        style={'width': '95%', 'marginBottom': '5px',}
                    ),
                    width=6,
                ),
                tune_controls(
                    f"window height",
                    dcc.Input(id='window-height', type='number',
                        min=300, 
                        value=paras['window_height'], 
                        style={'width': '95%', 'marginBottom': '5px',}
                    ),
                    width=6,
                ),
            ]

    children_on = [
        html.Div(
            id="common-bi",
            children=[
                html.I(className="bi bi-caret-down-fill bi-lg", id='common-bi-icon'),
                html.Span("Common Controls", className="title_font_lstyle"),
            ], style={"textAlign": "left", "witdh": "100%", "margin-left": "5px"}),
        dbc.Row(
            children=col_children,
            className="g-0 bg-white rounded-3 shadow-sm",  # Bootstrap
            style={#"minHeight": "100px", "minWidth": "100px", #  TODO 
                    'width': '100%',
                'border': '2px solid #ddd', 
                'borderRadius': '1px 1px 5px 5px',
                'marginBottom': '10px',
                "boxShadow": "0 2px 4px rgba(0,0,0,0.1)", } ),
    ]
    children_off = [
        html.Div(
            id="common-bi",
            children=[
                html.I(className="bi bi-caret-up-fill", 
                        style={"width": "40px", "height": "40px"}),
                html.Span("Common Controls", className="title_font_lstyle"),
            ], style={"textAlign": "left", "margin-left": "5px"}),
    ]
    return children_on if tune else children_off

def create_menu_window(paras, tune=None):
    if tune is None:
        tune = paras.get('windowbar_state', True)
    if tune:
        col_children = [window_panel_controls(paras, idx) 
                        for idx in range(paras['window_number'])]
        children_on = [
            html.Div(
                id="windows-bi",
                children=[
                    html.I(className="bi bi-caret-down-fill bi-lg", id='windows-bi-icon'),
                    html.Span("Windows Controls", className="title_font_lstyle"),
            ], style={"textAlign": "left", "witdh": "100%", "margin-left": "5px"}),
            dbc.Row(
                children=col_children,
                className="g-0 bg-white rounded-3 shadow-sm",  # Bootstrap工具类
                style={#"minHeight": "100px", "minWidth": "100px", 
                        'width': '100%',
                    'border': '2px solid #ddd', 
                    'borderRadius': '1px 1px 5px 5px',
                    'marginBottom': '10px',
                    "transition": "all 0.2s",
                    "boxShadow": "0 2px 4px rgba(0,0,0,0.1)", } ),

        ]
        return children_on
    else:
        children_off = [
            html.Div(
                id="windows-bi",
                children=[
                    html.I(className="bi bi-caret-up-fill", id='windows-bi-icon'),
                    html.Span("Windows Controls", className="title_font_lstyle"),
                ], style={"textAlign": "left", "margin-left": "5px"}),
        ]
        return children_off

def window_panel_controls(paras, idx):
    style={
        'paddingBottom': '10px',
        'marginBottom': '5px',
        'position': 'relative',
    }
    if idx != paras['window_number'] - 1:
        style['borderBottom'] = '3px solid #e0e0e0'

    para_row = dbc.Row(
        className="g-0 align-items-center",
        children=[
        tune_controls(
            f"Animation",
            dcc.Dropdown(
                id=f'animation-type-{idx}',
                options=None_options+[{'label': k, 'value': k} for k in paras["data_labels"]],
                value=paras['window_slices'][idx],
                clearable=True,
                searchable=True,
            ),
            width=6
        ),
        tune_controls(
            f"Colormap",
            dcc.Dropdown(
                id=f'colormap-{idx}',
                options=[{'label': k.title(), 'value': k} for k in px_colormap()],
                value=paras['colormaps'][idx],
            ),
            width=6
        ),
        tune_controls(
            f"Point Size",
            dcc.Dropdown(
                id=f'point-size-{idx}',
                options=None_options+[{'label': k, 'value': k} 
                                        for k,v in zip(paras["data_labels"], paras["data_dtypes"])
                                        if v =='continuous'],
                value=paras['points_size'][idx],
                clearable=True,
                searchable=True,
            ),
            width=6,
        ),
        tune_controls(
            f"Render",
            dcc.Dropdown(
                id=f'render-panel-{idx}',
                options=[{'label': k, 'value': k} for k in ['auto', 'webgl', 'svg']],
                value=paras['window_renders'][idx],
                clearable=False
            ),
            width=6,
        ),
        tune_controls(
            f"Point Scale",
            html.Div([
                dcc.Input(id=f'point-scale-{idx}', type='number',
                    min=0, 
                    step=0.01,
                    value=paras['points_scale'][idx], 
                    style={'width': '100%',}
                ),
            ], title = 'Set the mark size when not using point size',),
            width=4,
        ),

        tune_controls(
            f"Size Max",
            html.Div([
                dcc.Input(id=f'size-max-{idx}', type='number',
                    min=0, 
                    step=0.01,
                    value=paras['sizes_max'][idx], 
                    style={'width': '100%',},
                ),
            ], title = 'Set the maximum mark size when using point size',),
            width=4,
        ),

        tune_controls(
            f"Point Sample",
            dcc.Input(id=f'point-sample-{idx}', type='number',
                min=0, 
                max=1,
                step=0.01,
                # step=100,
                value=paras['point_samples'][idx], 
                style={'width': '100%',}
            ),
            width=4,
        ),
        tune_controls(
            f"Contain",
            html.Div([
            dbc.Switch( id=f"contain-hidden-{idx}",
                        value=True,
                        className="d-inline-block ms-2",
                        
                        persistence=False),
            ], title = 'contain hidden in point sample'),
            width=3,
            add_label_style={'display': 'block'},
        ),
        tune_controls(
            f"Legend",
            html.Div([
            dbc.Switch( id=f"legend-switch-{idx}",
                        value=True,
                        className="d-inline-block ms-2",
                        persistence=False),
            ], title = 'legend switch(on/off)'),
            width=3,
            add_label_style={'display': 'block'},
        ),
        html.Div([
                html.I(className=f"bi bi-{idx+1}-circle", 
                        id=f"window-panel-{idx}-bi-circle",
                        style={'color' : "#C99986", 
                                'fontSize': '1.4rem',
                                'position': 'absolute',
                                # 'padding': '0rem',
                                'bottom': '0.5rem',
                                'right': '0.5rem', 
                                'zIndex': 1e5
                        }
                )
        ]),
    ],style=style)
    return para_row
    
def create_graphs(paras, data=None):
    graphs = []
    for idx in range(paras['window_number']):
        is_last = idx == paras['window_number']-1

        col_children = [
            windown_tune(paras, idx),
            dcc.Graph(
                id=f'scatter-plot-{idx}',
                figure= create_figure(paras, idx, data=data),
                config={
                    'scrollZoom': True,
                    # 'displayModeBar': True,
                    'modeBarButtonsToAdd': ['select2d', 'lasso2d'],
                    # 'displaylogo': True
                }, style={'width': '100%', 'height': paras['window_height']}
            ),
            html.Div([
                html.I(className=f"bi bi-{idx+1}-circle-fill", 
                        id=f"window-{idx}-bi-circle",
                        style={'color' : "#C99986", 
                                'fontSize': '1.7rem',
                                'position': 'absolute',
                                'padding': '0rem',
                                'top': '0rem',
                                'right': '0.5rem', 
                                'zIndex': 1e5
                        }
                )
            ]),
            window_widget(paras, idx),
        ]

        graphs.append(
            dbc.Col(
                col_children,
                width={"lg":6, "md": 6, "sm": 12, 'xs':12},  # 响应式宽度 TODO
                # width=12,
                style={
                    # "display": "grid", #TODO grid
                    "padding": "5px",
                    "borderRight": "4px solid #eee" if not is_last else None,
                    "position": "relative",
                    "overflow": "hidden",
                    # "overflow": "visible"
                },
                className="position-relative"  # Bootstrap定位辅助类
            )
        )

    return dbc.Row(
        children=graphs,
        className="g-0 bg-white rounded-3 shadow-sm",  ## 默认允许换行
        style={#"minHeight": "100px", "minWidth": "300px", # 设置最小宽高 TODO 
                "display": "flex", #TODO grid
                'width': '100%',
                'border': '2px solid #ddd', 'borderRadius': '1px 1px 5px 5px',
                "boxShadow": "0 2px 4px rgba(0,0,0,0.1)", } 
    )

def window_widget(paras, idx):
    #https://unicode-explorer.com/emoji/
    return dbc.Row(
        style={
            # 'position': 'fixed',    
            'top': '1rem',           
            'left': '1rem',       
            'zIndex': 1e3,       
            'display': 'flex',
            'gap': '0.5rem'
        },
        children=[
            html.Div([
                dbc.Button(
                    '🖨️', 
                    className="custom-active-button iconButton_lstyle ",
                    id=f'save-as-figure-{idx}',
                    style={'backgroundColor': '#475D97'},
                    disabled=False,
                    title='save as figure',
                ),
                dbc.Input(id=f'figure-path-{idx}', type='text', 
                            placeholder='Save as (e.g.: ./saved_figure.html)',
                            style={'width': '30%', 'height': '2.40rem', 'overflow': 'hidden' })
                ], 
                style={
                        'flex': '3',
                        'display': 'flex', 
                        'alignItems': 'center', 
                        'gap': '0.25rem',
                        'flexWrap': 'nowrap',
                        #  'padding':'0',
                        #  'backgroundColor': 'white',
                         #'minWidth': '50px',
                }
            ),
        ]
    )

def windown_tune(paras, idx):
    return dbc.Row(
        className="g-0 align-items-center",
        children=[
        tune_controls(
            f"Basis",
            dcc.Dropdown(
                id=f'basis-type-{idx}',
                options=[{'label': k, 'value': k} for k in paras["basis_labes"]],
                value=paras['window_basis'][idx],
                clearable=True,
                searchable=True,
                # style={'width': '100%'},
                # className="form-control-sm",
            ),
            width=2
        ),
        tune_controls(
            f"Label",
            dcc.Dropdown(
                id=f'label-type-{idx}',
                options=[{'label': k, 'value': k} for k in paras["data_labels"]],
                value=paras['window_labels'][idx],
                clearable=True,
                searchable=True,
            ),
            width=2
        ),
        tune_controls(
            f"Split",
            dcc.Dropdown(
                id=f'split-type-{idx}',
                options=None_options + [{'label': k, 'value': k} 
                                        for k,v in zip(paras["data_labels"], paras["data_dtypes"])
                                        if v =='discrete'],
                value=paras['window_splits'][idx],
                clearable=True,
                searchable=True,
            ),
            width=2
        ),
        tune_controls(
            f"Split Iterms",
            dcc.Dropdown(
                id=f'split-iterm-{idx}',
                options=None_options + [{'label': k, 'value': k} 
                                        for k in paras['colors_order'].get(paras['window_splits'][idx], [])],
                value=paras['split_iterms'][idx],
                clearable=True,
                searchable=True,
                multi=True,
            ),
            width=6,
        ),

    ],style={
        'marginBottom': '2px',
        'borderBottom': '2px solid #e0e0e0',  # 增加行分割线
        'paddingBottom': '2px'
    })

def create_figure(paras, idx, data=None):
    # ipars = copy.deepcopy(paras['scatter_args'][idx])
    ipars = paras['scatter_args'][idx]
    ipars.update({
        'scale': paras['points_scale'][idx],
        'size':  paras['points_size'][idx],
        'size_max': paras['sizes_max'][idx],
        'save': paras['save_figs'][idx],
        'render_mode': paras['window_renders'][idx],
        'sample': paras['point_samples'][idx],
        'figscale': paras['figscale'],
    })  
    return scatter_wrap(data, paras, idx, **ipars)

def panel_controls(label, component, width=3):
    return dbc.Col(
        children=[
            html.Label(label, #className="form-label",
                        style={'fontWeight': 'bold', 'marginBottom': '3px'}),
            component
        ],
        width=width, 
        style={'padding': '0 10px', # 增加列间距
                "position": "relative",  # 为装饰元素定位
                "overflow": "visible",    # 允许装饰元素溢出
                'width': '100%',
                #"borderRight": "2px solid #ddd", #动态添加边框
        }    
    )

def tune_controls(label, component, width=3, md=3, lg=3, 
                    add_label_style={}, **kargs):  # 修改默认 width 为 3（Bootstrap 的 12 列系统）
    if label is None:
        children = component
    else:
        children=[
            html.Label(label, 
                        className="button_font_lstyle",
                        style=add_label_style) ,
            component
        ]
    return dbc.Col(
        children=children,
        width=width, #lg=lg, md=md,
        style={'padding': '3px', # 增加列间距
                "position": "relative",  # 为装饰元素定位
                "overflow": "visible"    # 允许装饰元素溢出
                #"borderRight": "2px solid #ddd", #动态添加边框
        }, **kargs
    )