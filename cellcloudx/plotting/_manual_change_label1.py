import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad

import sys
from typing import Optional, Union, Dict
import copy 

import dash_daq as daq

from ._colors import *
from ._manual_change_label1 import *
from ._plotlypl import scatter_wrap
from ..utilis._arrays import list_iter, vartype

print(random_colors(5))
def dash_app(initial_data, app=None, server=None, 
             suppress_callback_exceptions=True,
             debug=True, **run_args):
    import dash
    from dash import Dash, dcc, html, callback_context, Input, Output, State
    from dash.dependencies import Input, Output, State
    import dash_bootstrap_components as dbc
    from datetime import datetime

    if app is None:
        # app = Dash(__name__, )
        app = Dash(__name__, 
                   suppress_callback_exceptions=suppress_callback_exceptions,
                   external_stylesheets=[dbc.themes.BOOTSTRAP])
    if server is None:
        server = app.server

    STYLES = {
        'siderbar':{
            "position": "fixed",      # 固定定位（不随页面滚动）
            "top": 0,                 # 顶部对齐
            "left": 0,                # 左侧对齐
            # "bottom": 0,             # 底部对齐（实现全高度）
            "width": "17%",         # 宽度20rem（约320px）
            "padding": "2rem 1rem",   # 内边距上下2rem，左右1rem
            "background-color": "#f8f9fa",  # 浅灰色背景
            "overflow": "auto",       # 内容溢出时显示滚动条
            "box-shadow": "2px 0px 5px rgba(0,0,0,0.1)"  # 右侧阴影
        },
        'hideden_siderbar':{
            "position": "fixed",
            "top": 0,
            "left": 0,
            "bottom": 0,
            "width": "20rem",
            "padding": "2rem 1rem",
            "background-color": "#f8f9fa",
            "overflow": "auto",
            "box-shadow": "2px 0px 5px rgba(0,0,0,0.1)",
            "margin-left": "-22rem",  # 向左移出视口（隐藏）
            "transition": "margin-left 0.5s ease"  # 过渡动画效果
        },
        'contents' : {
            # 'display': 'grid',
            # 'gridTemplateColumns': '1fr 1fr',
            # 'gap': '2rem',
            # 'marginBottom': '2rem',
            # "height": "100%",
            # "overflow": "hidden"
            "margin-left": "17.5%",
            "margin-right": "0.5rem",
            'marginTop': '2rem',
            "padding": "0rem",
            "transition": "margin-left 0.5s ease",
            "background-color": "#fdfdfd",
            # 'gridTemplateColumns': 'repeat(auto-fit, minmax(100, 1fr))',  # 自适应列
        },
        'hideden_contents' : {
            "margin-left": "1rem",
            "margin-right": "0.5rem",
            "padding": "0rem",
            "transition": "margin-left 0.5s ease",
            "background-color": "#fdfdfd",
        },
        'collapse_button_style':{
            "position": "fixed",       # 固定定位
            "top": "10px",             # 距离顶部10px
            "left": "0",               # 贴左对齐
            "zIndex": 1e64,            # 确保在最上层
            "width": "40px",           # 按钮宽度
            "height": "40px",         # 按钮高度
            "border-radius": "50%",   # 圆形按钮
            "border": "1px solid #ddd",# 浅灰色边框
            "background-color": "#E14B0A",  # 背景色与侧边栏一致
            "box-shadow": "2px 2px 6px rgba(0, 0, 0, 0.1)",  # 立体阴影
            "transition": "margin-left 0.5s ease",
            "display": "flex",
            "justify-content": "center",  # 水平居中
            "align-items": "center",   # 垂直居中
            "cursor": "pointer",       # 鼠标手型指针
        },
        'expand_button_style':{
            "position": "fixed",       # 固定定位
            "top": "10px",             # 距离顶部10px
            "left": "0",               # 贴左对齐
            "zIndex": 1e64,            # 确保在最上层
            "width": "40px",           # 按钮宽度
            "height": "40px",         # 按钮高度
            "border-radius": "50%",   # 圆形按钮
            "border": "1px solid #ddd",# 浅灰色边框
            "background-color": "#3BDEC5",  # 背景色与侧边栏一致
            "box-shadow": "2px 2px 6px rgba(0, 0, 0, 0.1)",  # 立体阴影
            "transition": "margin-left 0.5s ease",
            "display": "none",         # 默认隐藏
            "justify-content": "center",  # 水平居中
            "align-items": "center",   # 垂直居中
            "cursor": "pointer",       # 鼠标手型指针
        },

        'layout': {
            'padding': '2rem',        # 内边距
            'margin': '0 auto',        # 水平居中
            'backgroundColor': '#f5f6fa'  # 浅灰色背景
        },
        'controlGroup': {
            'display': 'grid',         # 网格布局
            'gridTemplateColumns': 'repeat(auto-fit, minmax(240px, 1fr))',  # 自适应列
            'gap': '1rem',             # 元素间距
            'marginBottom': '1.5rem'   # 底部外边距
        },
        'graphContainer': {
            "display": "flex",         # 弹性布局
            "height": "calc(100vh - 4rem)",  # 视口高度减4rem
            "background-color": "#fff",# 白色背景
            "border-radius": "8px",    # 圆角
            "box-shadow": "0 4px 8px rgba(0, 0, 0, 0.17)",  # 阴影
            "overflow": "hidden"       # 内容裁剪
        },
        'floatingControls': {  # 悬浮操作按钮组
            'position': 'fixed',    # 绝对定位
            'top': '1rem',             # 距顶部1rem
            'right': '1rem',           # 距右侧1rem
            'zIndex': 100,             # 层级高于图表
            'display': 'flex',         # 弹性布局
            'gap': '0.5rem'            # 按钮间距
        },
        'iconButton': {  # 图标按钮样式
            'width': '36px',           # 按钮宽度
            'height': '36px',         # 按钮高度
            'borderRadius': '50%',    # 圆形按钮
            'border': 'none',         # 无边框
            'cursor': 'pointer',      # 手型指针
            'transition': 'all 0.2s'  # 过渡动画效果
        }
    }

    None_options = [{'label': 'None', 'value': 'None'}]

    def create_control(label, component, width=4):
        return html.Div(
            className='control-item',
            style={'gridColumn': f'span {width}'},
            children=[
                html.Label(label, className='control-label'),
                component
            ]
        )

    def tune_controls(label, component, width=3):  # 修改默认 width 为 3（Bootstrap 的 12 列系统）
        return dbc.Col(
            children=[
                html.Label(label, #className="form-label",
                           style={'fontWeight': 'bold', 'marginBottom': '3px'}),
                component
            ],
            width=width, #lg=width,
            style={'padding': '0 10px', # 增加列间距
                    "position": "relative",  # 为装饰元素定位
                    "overflow": "visible"    # 允许装饰元素溢出
                   #"borderRight": "2px solid #ddd", #动态添加边框
            }    
        )

    def windown_tune(data, idx):
        return dbc.Row(
            className="g-0 align-items-center",
            children=[
            tune_controls(
                f"Basis {idx+1}",
                dcc.Dropdown(
                    id=f'basis-type-{idx}',
                    options=[{'label': k, 'value': k} for k in sorted(data["locs"].keys())],
                    value=data['window_basis'][idx],
                    clearable=True,
                    searchable=True,
                    # style={'width': '100%'},
                    # className="form-control-sm",
                ),
                width=3
            ),
            tune_controls(
                f"Label {idx+1}",
                dcc.Dropdown(
                    id=f'label-type-{idx}',
                    options=[{'label': k, 'value': k} for k in sorted(data["metadata"].keys())],
                    value=data['window_labels'][idx],
                    clearable=False
                ),
                width=3
            ),
            tune_controls(
                f"Split {idx+1}",
                dcc.Dropdown(
                    id=f'split-type-{idx}',
                    options=None_options + [{'label': k, 'value': k} for k in sorted(data["metadata"].keys())],
                    value=data['window_splits'][idx],
                    clearable=True,
                    searchable=True,
                ),
                width=3
            ),
            tune_controls(
                f"Split Iterms {idx+1}",
                dcc.Dropdown(
                    id=f'split-iterm-{idx}',
                    options=None_options + [{'label': k, 'value': k} for k in sorted(data["metadata"].keys())],
                    value=data['window_splits'][idx],
                    clearable=False,
                    multi=True,
                ),
                width=3,
            ),
            tune_controls(
                f"Animation {idx+1}",
                dcc.Dropdown(
                    id=f'animation-type-{idx}',
                    options=None_options+[{'label': k, 'value': k} for k in  sorted(data["metadata"].keys())],
                    value=data['window_slices'][idx],
                    clearable=False
                ),
                width=3
            ),
            tune_controls(
                f"Colormap {idx+1}",
                dcc.Dropdown(
                    id=f'colormap-{idx}',
                    options=[{'label': k.title(), 'value': k} for k in px_colormap()],
                    value=data['colormap'][idx],
                )
            ),
            tune_controls(
                f"Render {idx+1}",
                dcc.Dropdown(
                    id=f'render-type-{idx}',
                    options=[{'label': k, 'value': k} for k in ['auto', 'webgl', 'svg']],
                    value=data['window_renders'][idx],
                    clearable=False
                ),
                width=2
            ),
            tune_controls(
                f"Point Size {idx+1}",
                dcc.Slider(
                    id=f'point-size-{idx}',
                    min=1,
                    max=20,
                    step=1,
                    value=data['points_size'][idx],
                    marks={i: str(i) for i in range(0, 21, 5)}
                ),
                width=3
            ),
            tune_controls(
                f"Legend {idx+1}",
                daq.BooleanSwitch(
                    on=True,
                    color="#9B51E0",
                    ),
                width=1
            ),

        ],style={
            'marginBottom': '2px',
            'borderBottom': '2px solid #e0e0e0',  # 增加行分割线
            'paddingBottom': '2px'
        })
    
    def create_figure(data, idx):
        ipars = copy.deepcopy(data['scatter_args'][idx])
        ipars.update({
            'scale': 10, #data['points_size'][idx], #TODO
            'size': None, #TODO
            'save': data['save_figs'][idx],
            'render_mode': data['window_renders'][idx],
            'sample': data['point_samples'][idx],
        })  
        return scatter_wrap(data, idx, **ipars)

    def create_graph(data, idx):
        return html.Div(
            style={
                "width": "100%",
                "background-color": "#f5f5f5",
                "overflow": "hidden",
                'borderBottom': '5px solid #e0e0e0',  # 增加行分割线
            },
            children = [ 
                windown_tune(data, idx),
                dcc.Graph(
                    id=f'scatter-plot-{idx}',
                    figure= create_figure(data, idx),
                    config={
                        'scrollZoom': True,
                        'displayModeBar': True,
                        'modeBarButtonsToAdd': ['select2d', 'lasso2d'],
                        'displaylogo': False
                    }, style={'width': '100%',}
                ),
                html.Div(
                    style={
                        # 'position': 'fixed',    # 绝对定位
                        'top': '1rem',             # 距顶部1rem
                        'right': '1rem',           # 距右侧1rem
                        'zIndex': 1000,             # 层级高于图表
                        'display': 'flex',         # 弹性布局
                        'gap': '0.5rem'            # 按钮间距
                    },
                    children=[
                        html.Button(
                            '🗑️',
                            id=f'clear-{idx}',
                            style={**STYLES['iconButton'], 'backgroundColor': '#ff4757'},
                            title='Clear selection'
                        ),
                        html.Button(
                            '⤵️', 
                            id=f'export-{idx}',
                            style={**STYLES['iconButton'], 'backgroundColor': '#2ed573'},
                            title='Export figure'
                        )
                    ]
                ),
                html.Br(),
            ]
        )

    # def create_graphs(data):
    #     return [create_graph(data, i) for i in range(data['window_number'])]

    def create_graphs(data):
        graphs = []
        for idx in range(data['window_number']):
            is_last = idx == data['window_number']-1
    
            col_children = [
                windown_tune(data, idx),
                dcc.Graph(
                    id=f'scatter-plot-{idx}',
                    figure= create_figure(data, idx),
                    config={
                        'scrollZoom': True,
                        'displayModeBar': True,
                        'modeBarButtonsToAdd': ['select2d', 'lasso2d'],
                        'displaylogo': True
                    }, style={'width': '100%',}
                ),
                html.Div(
                    style={
                        # 'position': 'fixed',    # 绝对定位
                        'top': '1rem',             # 距顶部1rem
                        'right': '1rem',           # 距右侧1rem
                        'zIndex': 1000,             # 层级高于图表
                        'display': 'flex',         # 弹性布局
                        'gap': '0.5rem'            # 按钮间距
                    },
                    children=[
                        html.Button(
                            '🗑️',
                            id=f'clear-{idx}',
                            style={**STYLES['iconButton'], 'backgroundColor': '#ff4757'},
                            title='Clear selection'
                        ),
                        html.Button(
                            '⤵️', 
                            id=f'export-{idx}',
                            style={**STYLES['iconButton'], 'backgroundColor': '#2ed573'},
                            title='Export figure'
                        )
                    ]
                ),
                # html.Div(style={
                #     "position": "absolute",
                #     "right": "-2px",
                #     "top": "15%",
                #     "bottom": "15%",
                #     "width": "4px",
                #     "background": "linear-gradient(to right, rgba(0,0,0,0.1), transparent)",
                #     "zIndex": 100
                # }) if not is_last else None
            ]

            graphs.append(
                dbc.Col(
                    col_children,
                    style={
                        "padding": "5px",
                        "borderRight": "4px solid #eee" if not is_last else None,
                        "position": "relative",
                        "overflow": "hidden",
                        # "overflow": "visible"
                    },
                    # width=6,
                    className="position-relative"  # Bootstrap定位辅助类
                )
            )
        
        return dbc.Row(
            children=graphs,
            className="g-0 bg-white rounded-3 shadow-sm",  # Bootstrap工具类
            style={"minHeight": "100px", "minWidth": "100px", # 设置最小宽高 TODO 
                   'width': '100%',
                   'border': '2px solid #ddd', 'borderRadius': '1px 1px 5px 5px',
                    "boxShadow": "0 2px 4px rgba(0,0,0,0.1)", } 
        )
    # CONTROL = common_controls(initial_data, idx=-1)
    # for iwin in range(initial_data['window_number']):
    #     CONTROL += windown_controls(initial_data, iwin)

    # =============== app layout ===============
    app.layout = html.Div([
        dcc.Store(id='data-store', data=initial_data),
        dcc.Store(id='selection-store', data={'main': [], 'comparison': []}),
        dcc.Store(id='history-tracker', data=[]),
        # dcc.Store(id="graph-widths", data={"graph1": 50, "graph2": 50}),

        html.Div(
            id="sidebar",
            style=STYLES['siderbar'],
            children=[
                html.Div([
                    html.H2("Parameters", className="display-6", style={"textAlign": "center"}),
                    html.Button("◀", id="collapse-button", 
                                className="btn btn-secondary",
                                style=STYLES['collapse_button_style'],)
                ]),
                html.Hr(),

                ## common control
                html.H4("common control", style={"marginTop": "10px"}),
                html.Label("point size:"),
                dcc.Slider(
                    id="point-size-slider",
                    min=1,
                    max=20,
                    value=10,
                    marks={i: str(i) for i in range(1, 21, 2)},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
                html.Hr(),
        
                # panal 1
                # html.H4("panal 1", style={"marginTop": "10px"}),
                # html.Label("图表1颜色字段:", style={"marginTop": "10px"}),

                # panal 2
                # html.H4("panal 2", style={"marginTop": "10px"}),
                # html.Label("图表2颜色字段:"),

        ]),

        html.Button(
            "▶",
            id="expand-button",
            style=STYLES['expand_button_style'],
        ),

        html.Div(
            # className='graph-container',
            id="contents",
            style=STYLES['contents'],
            children=[
                html.Div(
                    id="graph-container",
                    style={
                        "display": "flex",
                        # # "height": "calc(100vh - 4rem)",
                        # 'width': '100%',
                        # "background-color": "#fff",
                        # "border-radius": "8px",
                        # "box-shadow": "0 4px 8px rgba(0, 0, 0, 0.17)",
                        'marginTop': '0rem',
                        'borderRadius': '1px 1px 5px 5px',
                        'border': '0px solid #ddd', 
                        "minHeight": "100px", "minWidth": "100px", # 设置最小宽高 TODO 
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)", 
                        # "overflow": "hidden"
                    },
                    children = create_graphs(initial_data,),
                ),
            ],
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
        ),
    # ],
    #add layouts
    ])

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
            return sidebar_style, content_style, STYLES['expand_button_style']

        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if button_id == "collapse-button": 
            sidebar_style = STYLES['hideden_siderbar']
            content_style = STYLES['hideden_contents']
            expand_button_style = {**STYLES['expand_button_style'], "display": "flex"}
        elif button_id == "expand-button":  # 展开侧边栏
            sidebar_style = STYLES['siderbar']
            content_style = STYLES['contents']
            expand_button_style = STYLES['expand_button_style']
        return sidebar_style, content_style, expand_button_style


    # @app.callback(
    #     Output('data-store', 'data', allow_duplicate=True),
    #     Input("Legend 1", "on"),
    # )
    # def update_output(on, data):
    #     data['legend_1'] = bool("{}".format(on))
    #     return data

    # # 应用布局
    # app.layout = html.Div([
    #     dcc.Store(id='data-store', data=initial_data),
    #     dcc.Store(id='selection-store', data=[]),

    #     html.Div(
    #         id="sidebar",
    #         style=SIDEBAR_STYLE,
    #         children=[
    #             html.Div([
    #                 html.H2("Parameters", className="display-6", style={"textAlign": "center"}),
    #                 html.Button("◀", id="collapse-button", className="btn btn-secondary", style={"width": "100%"})
    #             ]),
    #             html.Hr(),

    #             ## common control
    #             html.H4("common control", style={"marginTop": "10px"}),
    #             html.Label("point size:"),
    #             dcc.Slider(
    #                 id="point-size-slider",
    #                 min=1,
    #                 max=20,
    #                 value=10,
    #                 marks={i: str(i) for i in range(1, 21, 2)},
    #                 tooltip={"placement": "bottom", "always_visible": True}
    #             ),
    #             html.Hr(),
            
    #             # panal 1
    #             html.H4("panal 1", style={"marginTop": "10px"}),
    #             html.Label("图表1颜色字段:", style={"marginTop": "10px"}),
    #             dcc.Dropdown(
    #                 id='label-type-dropdown',
    #                 options=[{'label': i, 'value': i} for i in initial_data["locs"].keys()],
    #                 value=initial_data['locs_type'],
    #                 clearable=False,
    #                 style={"marginBottom": "20px"}
    #             ),
    #             html.Hr(),

    #             # panal 2
    #             html.H4("panal 2", style={"marginTop": "10px"}),
    #             html.Label("图表2颜色字段:"),
    #             dcc.Dropdown(
    #                 id='split-type-dropdown',
    #                 options=[{'label': i, 'value': i} for i in initial_data["labels"].keys()],
    #                 value=initial_data['split_type'],
    #                 clearable=False,
    #                 style={'width': '100%'}
    #             ),
    #             html.Hr(),
    #         ]
    #     ),
        
    #     # 展开按钮
    #     html.Button(
    #         "▶",
    #         id="expand-button",
    #         style=EXPAND_BUTTON_STYLE
    #     ),

    #     # 主内容区域，包含图表
    #     html.Div(
    #         id="content",
    #         style=CONTENT_STYLE,
    #         children=[
    #             html.H1("双图交互示例", style={"textAlign": "center", "color": "#333", "marginBottom": "30px"}),
    #             html.Div(
    #                 id="graph-container",
    #                 style=GRAPH_CONTAINER_STYLE,
    #                 children=[
    #                     html.Div(
    #                         dcc.Graph(id="scatter-plot-1", config={"scrollZoom": True}),
    #                         id="graph-1",
    #                         style={"flex": "1", **GRAPH_STYLE}
    #                     ),
    #                     html.Div(
    #                         id="drag-bar",
    #                         style=DRAG_BAR_STYLE
    #                     ),
    #                     html.Div(
    #                         dcc.Graph(id="scatter-plot-2", config={"scrollZoom": True}),
    #                         id="graph-2",
    #                         style={"flex": "1", **GRAPH_STYLE}
    #                     )
    #                 ]
    #             )
    #         ]
    #     ),

    #     # 存储拖动状态
    #     dcc.Store(id="graph-widths", data={"graph1": 50, "graph2": 50}),
    # ])

    # # 回调：切换侧边栏显示/隐藏
    # @app.callback(
    #     [Output("sidebar", "style"),
    #     Output("content", "style"),
    #     Output("expand-button", "style")],
    #     [Input("collapse-button", "n_clicks"),
    #     Input("expand-button", "n_clicks")],
    #     [State("sidebar", "style"),
    #     State("content", "style")],
    #     prevent_initial_call=True
    # )
    # def toggle_sidebar(collapse_clicks, expand_clicks, sidebar_style, content_style):
    #     ctx = dash.callback_context
    #     if not ctx.triggered:
    #         return sidebar_style, content_style, EXPAND_BUTTON_STYLE

    #     button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    #     if button_id == "collapse-button":  # 隐藏侧边栏
    #         sidebar_style = HIDDEN_SIDEBAR_STYLE
    #         content_style = HIDDEN_CONTENT_STYLE
    #         expand_button_style = {**EXPAND_BUTTON_STYLE, "display": "flex"}
    #     elif button_id == "expand-button":  # 展开侧边栏
    #         sidebar_style = SIDEBAR_STYLE
    #         content_style = CONTENT_STYLE
    #         expand_button_style = EXPAND_BUTTON_STYLE

    #     return sidebar_style, content_style, expand_button_style

    # # 回调：更新图表
    # @app.callback(
    #     [Output("scatter-plot-1", "figure"),
    #     Output("scatter-plot-2", "figure"),
    #     Output('data-store', 'data', allow_duplicate=True),],
    #     [Input("point-size-slider", "value"),
    #     Input("label-type-dropdown", "value"),
    #     Input("split-type-dropdown", "value")],
    #     State('data-store', 'data'),
    #     prevent_initial_call=True
    # )
    # def update_graphs(point_size, color1, color2, data):
    #     data['scatter_args']['scale'] = point_size
    #     fig1 = scatter_wrap(data, **data['scatter_args'])
    #     fig2 = scatter_wrap(data, **data['scatter_args'])
    #     return fig1, fig2, data

    # # 回调：拖动分隔条调整左右面板宽度
    # @app.callback(
    #     [Output("graph-1", "style"),
    #     Output("graph-2", "style"),
    #     Output("graph-widths", "data")],
    #     Input("drag-bar", "n_clicks"),
    #     State("graph-widths", "data"),
    #     prevent_initial_call=True
    # )
    # def drag_resize(n_clicks, widths):
    #     widths["graph1"] = max(20, widths["graph1"] - 5)
    #     widths["graph2"] = max(20, widths["graph2"] + 5)
    #     return (
    #         {"flex": widths["graph1"], **GRAPH_STYLE},
    #         {"flex": widths["graph2"], **GRAPH_STYLE},
    #         widths
    #     )
    app.run_server(debug=True, **run_args)
    return 

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
                    scatter_args :dict ={},
                    start_app =True,
                  run_args :dict ={}):
    initial_data = get_init_data(locs=locs, metadata=metadata,
                                colors=colors,
                                window_labels =window_labels,
                                window_splits =window_splits,
                                window_slices =window_slices,
                                window_basis =window_basis,
                                window_number=window_number,
                                colormap=colormap,
                                scatter_args=scatter_args)
    if start_app:
        dash_app(initial_data, **run_args)
    else:
        return initial_data

def get_init_data(locs: Dict[str, np.ndarray] = None, 
                  metadata : pd.DataFrame=None, 
                  colors :dict ={}, 
                  colormap =None,
                  colors_order :dict ={},
                    window_labels : list=None,
                    window_splits : list=None,
                    window_basis : list=None,
                    window_slices : list=None,
                  window_number: int=2,
                  scatter_args :dict ={}) -> dict:

    metadata = pd.DataFrame(metadata).copy()

    window_labels = metadata.columns if window_labels is None else window_labels
    window_splits = metadata.columns if window_splits is None else window_splits
    window_basis = list(locs.keys()) if window_basis is None else window_basis
    window_slices = None if window_slices is None else window_slices
    window_number = min(window_number or 2, len(window_labels))
    assert window_number>=1
    assert window_number<=2, "only 1 or 2 windows are supported"
    
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
                sample=min(10000/metadata.shape[0], 0.5),
                random_state = 200504, order ='guess',
                show_grid =True, 
                showticklabels = False,
                render_mode = 'auto',
                colormap=None, template='none',
                itemsizing='constant',
                itemwidth=None,
                legdict={},
            )
    init_scatter_args.update(scatter_args)
    scatter_args = [ copy.deepcopy(init_scatter_args) for i in range(window_number) ]

    point_samples = [i.get('sample', 0.5) for i in scatter_args]
    points_size= [i.get('scale', 1) for i in scatter_args]
    points_scale =[None] *window_number
    colormaps = ['cmap1']*window_number
    initial_data = {
        'locs' : { k:np.asarray(v) for k,v in locs.items() },
        "metadata": { i:metadata[i] for i in metadata.columns },
        'index' : metadata.index,

        'window_labels': window_labels,
        'window_splits': window_splits,
        'split_iterms': ['NONE']*window_number,
        'window_number': window_number,
        'window_slices': window_slices,
        "current_lables": window_labels,

        'point_samples': point_samples,
        'window_renders':['webgl']*window_number,
        'points_size':points_size,
        'points_scale':points_scale, #TODO
        'window_basis': window_basis,
        'save_figs': [None]*window_number,

        "history": [],
        'undo_stack': [],
        'redo_stack': [],
        'colors': colors_init or {},
        'colormap': colormaps,
        'colors_order': colors_order_init or {},
        'scatter_args': scatter_args,
    }

    return initial_data
