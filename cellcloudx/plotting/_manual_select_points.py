import sys
import numpy as np
import pandas as pd
import numpy as np
import matplotlib
import re
import matplotlib.pyplot as plt
import plotly.express as px #5.3.1
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from ..plotting._utilis import colrows
from ..plotting._colors import *
from ..plotting._spatial3d import get_spatial_info
from ..plotting._utilis import image2batyes
from ..utilis._arrays import list_iter, vartype

import numpy as np
import pandas as pd

def centerlize(X):
    X = X.copy()
    N,D = X.shape
    Xm = np.mean(X, 0)
    X -= Xm
    Xs = np.sqrt(np.sum(np.square(X))/(N*D/2))
    X /= Xs
    return X

def scale2list(x, n):
    if isinstance(x, (float, int, str, bool, np.bool, np.number, )):
        return [x for i in range(n)]
    elif isinstance(x, list):
        return x[:n]
    else:
        raise ValueError('samples should be a number or a list of numbers')

def manual_select_points(pts1, pts2,  same_scale=True, size=1, ssize=5, ksize=8,
                          samples=None, pairs=None, slw=1.0, klw=2.0,
                         showlegend=False, normal ='scale', space=0.01, color=['blue', 'green'],
                         show_grid =False,  showticklabels = False,
                          seed=0, width=1200, height=None):
    import ipywidgets as widgets
    from IPython.display import display

    pts1r, pts2r = pts1.copy(), pts2.copy()
    N1, N2 = pts1.shape[0], pts2.shape[0]
    ncells = 2
    if normal in ['center']:
        pts1 =pts1r - np.mean(pts1r, 0)
        pts2 =pts2r - np.mean(pts2r, 0)
    elif normal in ['scale', True]:
        pts1 = centerlize(pts1r)
        pts2 = centerlize(pts2r)
    else:
        pts1 = pts1.copy()
        pts2 = pts2.copy()

    if normal is not False:
        x2_shift = ((pts1[:, 0].max() - pts1[:, 0].min()) + (pts2[:, 0].max() - pts2[:, 0].min())) * space
        pts2[:,0] += pts1[:, 0].max() + x2_shift - pts2[:, 0].min() 

    group = np.repeat(np.arange(2), [N1, N2] )
    gindx = np.r_[ np.arange(N1), np.arange(N2) ]
    color = np.repeat(scale2list(color, ncells), [N1, N2] )
    size  = np.repeat(scale2list(size, ncells), [N1, N2] )
    data = pd.DataFrame(np.concatenate([pts1, pts2], 0), columns=['x', 'y'])
    data[['xr', 'yr']] = np.concatenate([pts1r, pts2r], 0)
    data['group'] = group.astype(np.int64)
    data['gindx'] = gindx.astype(np.int64)
    data['text']  = data['group'].astype(str) + '_' + data['gindx'].astype(str) + '_' + data.index.astype(str)
    data['color'] = color
    data['size'] = size

    if not samples is None:
        if isinstance(samples, (float, int)):
            samples = [samples for i in  range(ncells) ]
        elif isinstance(samples, list):
            samples = samples[:ncells]
        else:
            raise ValueError('samples should be a number or a list of numbers')

        idxs = []
        np.random.seed(seed)
        for idx, isize in enumerate([N1, N2]):
            sample = samples[idx]
            if (sample is None) or sample == 1:
                idx = np.arange(isize)
            elif (sample>1):
                if sample  <10: print('warning: sample size is smaller than 10')
                idx = np.random.choice(isize, min(int(sample), isize), replace=False)
            elif (sample<1):
                idx = np.random.choice(isize, int(isize*sample), replace=False)
            idxs.append(idx)
        sidx = np.r_[ idxs[0], idxs[1] + N1]
        data = data.iloc[sidx]

    data['kidx'] = np.arange(data.shape[0])
    # ===== FigureWidget =====
    fig = go.FigureWidget()
    fig.add_trace(
        go.Scattergl(
        x=data['x'],
        y=data['y'],
        mode='markers',
        marker=dict(color=data['color'], size=data['size'], 
                    line=dict(width=0,color='DarkSlateGrey') ),
        selected=dict(marker=dict(color='red', size=ssize)),
        unselected=dict(marker=dict(opacity=0.5)),
        customdata=data['text'],
        text=data['text'], 
        # hoverinfo='text+x+y',
        hoverinfo='text',
    ))

    fig.update_layout( width=width, height=height, 
                    #    plot_bgcolor='#FFFFFF',
                    showlegend=showlegend,
                     hovermode='closest',
                     clickmode='event+select',
                    scene=dict(
                            xaxis=dict(visible=show_grid, showticklabels=showticklabels),
                            yaxis=dict(visible=show_grid, showticklabels=showticklabels),
                            zaxis=dict(visible=show_grid, showticklabels=showticklabels),
                        ),
                    margin=dict(l=1, r=1, t=20, b=5), )

    if same_scale:
        fig.update_yaxes( scaleanchor="x",scaleratio=1,)


    # ===== State Tracking =====
    if pairs is None:
        pairs = []
    else:
        assert isinstance(pairs, (pd.DataFrame))
        pairs = pairs.copy()
        lidx = pairs['left_idx'].values.astype(np.int64)
        ridx = pairs['right_idx'].values.astype(np.int64)

        ldf = data[(data['group'] ==0)].set_index('gindx').loc[lidx, ['xr', 'yr', 'kidx']].values
        rdf = data[(data['group'] ==1)].set_index('gindx').loc[ridx, ['xr', 'yr', 'kidx']].values

        pairs = pd.DataFrame(np.concatenate([ldf, rdf], 1), 
                             columns=['left_x', 'left_y', 'lidx', 'right_x', 'right_y', 'ridx'])
        pairs['left_idx'] = lidx
        pairs['right_idx'] = ridx
        pairs[['left_idx', 'right_idx', 'lidx', 'ridx']] = pairs[['left_idx', 'right_idx', 'lidx', 'ridx']].astype(np.int64)
        pairs = [row for i, row in pairs.iterrows()]

    drop_paris = []
    temp_selection = {'l': None, 'r': None, 'org_l':None, 'org_r':None}  # Temporarily selected point indices
    lines = []  # Store all line objects

    # ===== UI Components =====
    out = widgets.Output()  # Output info box
    pairs_out = widgets.Output()  # Pair info output box

    # Control buttons
    btn_add_pair = widgets.Button(description="Add Pair", button_style='success', icon='plus')
    btn_remove_last = widgets.Button(description="Clear Last", button_style='warning', icon='step-backward')
    btn_recovery_last = widgets.Button(description="Recovery", button_style='warning', icon='step-forward')
    btn_clear_selection = widgets.Button(description="Clear Selection", button_style='warning', icon='eraser')
    btn_clear_all = widgets.Button(description="Clear All", button_style='danger', icon='trash')
    btn_export = widgets.Button(description="Export Pairs", button_style='info', icon='download')
    
    pts_size = widgets.Combobox(
        placeholder='0.5',
        description='selpts size',
        ensure_option=True,
        disabled=False
    )

    btn_lineshow = widgets.Checkbox(
        value=True,
        description='Show Current',
        layout=widgets.Layout(width='11%'),
        disabled=False,
        indent=False
    )

    btn_allshow = widgets.Checkbox(
        value=True,
        description= 'Show Matched',
        layout=widgets.Layout(width='11%'),
        disabled=False,
        indent=False
    )
    btn_switch = widgets.Checkbox(
        value=False,
        description= 'swith lines',
        layout=widgets.Layout(width='11%'),
        disabled=False,
        indent=False
    )
    btn_clearidx = widgets.Text(
        description='Clear Index:',
        layout=widgets.Layout(width='10%'),
    )
    btn_clearidxdo= widgets.Button(
        button_style='success',
        icon='play',
        layout=widgets.Layout(width='2%'),
    )
    button_box = widgets.HBox([
        btn_add_pair, btn_remove_last, btn_recovery_last, btn_clear_selection,
        btn_clear_all, btn_export, btn_clearidx, btn_clearidxdo, btn_lineshow, btn_allshow,  #btn_switch,
    ])

    # Status display
    counter_label = widgets.Label(value=f"Selected pairs: {len(pairs)}")

    # ===== Click Callback =====
    def select_point(trace, points, selector):
        """Handle scatter point selection"""
        with out:
            out.clear_output()
            if not points.point_inds:
                return

            idx = points.point_inds[0]
            tinfor = data.iloc[idx]
            group, gindx, sindx = tinfor['text'].split('_')
            
            if group == '0':
                temp_selection['l'] = int(idx)
                temp_selection['org_l'] = int(gindx)
                print(f"Selected left: point {gindx} @ ({tinfor['xr']:.3f}, {tinfor['yr']:.3f})")
            elif group == '1':
                temp_selection['r'] = int(idx)
                temp_selection['org_r'] = int(gindx)
                print(f"Selected right: point {gindx} @ ({tinfor['xr']:.3f}, {tinfor['yr']:.3f})")

            counter_label.value = f"All pairs: {len(pairs)}"
            # trace.selectedpoints = [tinfor['text']]
            fig.data[0].selectedpoints = [idx]
            show_current_line()

    def show_current_line(b = None):
        if not ((temp_selection['l']  is None) or (temp_selection['r'] is None)):
            if btn_lineshow.value:
                idf  = data.iloc[[temp_selection['l'] , temp_selection['r']]]
                mdict = dict(
                    x=idf['x'], y=idf['y'],
                    mode="markers+lines", 
                    showlegend = showlegend,
                    marker=dict(color=['red', 'orange'], size=ssize, 
                                line=dict(width=0,color='DarkSlateGrey') ),
                    line={'color': '#3a3a3a', 'width': slw},
                    selected=dict(marker=dict(size=ssize, opacity=1.0),),
                    unselected=dict(marker=dict(size=ssize, opacity=1.0),),
                )
                if len(fig.data) == 1:
                    fig.add_trace(go.Scattergl(**mdict))
                else:
                    fig.data[1].update(mdict)
                fig.data[1].visible = True
            else:
                fig.data = fig.data[:1]
                # if len(fig.data) > 1:
                #     fig.data[1].visible = False

    # ===== Add Pair =====
    def add_pair(b=None):
        """Add a new pair"""
        with out:
            out.clear_output()
            if temp_selection['l'] is None:
                print("Please select a left point first")
                return
            else:
                lidx, lgindx = temp_selection['l'], temp_selection['org_l']
                linfor = data.iloc[lidx]
                print(f"Add left: point {lgindx} @ ({linfor['xr']:.3f}, {linfor['yr']:.3f})")

            if temp_selection['r'] is None:
                print("Please select a right point first")
                return
            else:
                ridx, rgindx = temp_selection['r'], temp_selection['org_r']
                rinfor = data.iloc[ridx]
                print(f"Add right: point {rgindx} @ ({rinfor['xr']:.3f}, {rinfor['yr']:.3f})")

            # Save pair
            pair_info = pd.Series({
                'left_idx': lgindx,
                'right_idx': rgindx,
                'left_x': linfor['xr'],
                'left_y': linfor['yr'],
                'right_x': rinfor['xr'],
                'right_y': rinfor['yr'],
                'lidx': lidx,
                'ridx': ridx,
                # 'lx0': linfor['x'],
                # 'ly0': linfor['y'],
                # 'rx0': rinfor['x'],
                # 'ry0': rinfor['y'],
            })
            pairs.append(pair_info)

            # Reset temporary selection
            for i in ['l', 'r', 'org_l', 'org_r']:
                temp_selection[i] = None
            fig.data[0].selectedpoints = None

            print(f"Added pair #{len(pairs)}: Left[{lgindx}] → Right[{rgindx}]")

            # Update pair display
            update_pairs_display()
            counter_label.value = f"Created pairs: {len(pairs)}"
            show_all_matches()

    def show_all_matches(b = None):
        npoints = len(pairs)
        fig.data = fig.data[:1]
        if(not btn_allshow.value or npoints == 0):
            return
 
        colors = random_colors(npoints, seed=seed)
        for i, ipair in enumerate(pairs):
            idf  = data.iloc[[ipair['lidx'] , ipair['ridx']]]
            mdict = dict(
                x=idf['x'], y=idf['y'],
                mode="markers+lines", 
                name = f'Pair {i}',
                showlegend = showlegend,
                marker=dict(color=colors[i], size=ksize, 
                            line=dict(width=0,color='DarkSlateGrey') ),
                line={'color': colors[i], 'width': klw},
                selected=dict(marker=dict(size=ssize, opacity=1.0),),
                unselected=dict(marker=dict(size=ssize, opacity=0.5),),
                text=[f'l_{i}',  f'r_{i}'],
                hoverinfo='text',
                # hoverinfo='text+x+y'
            )
            fig.add_trace(go.Scattergl(**mdict))

    # ===== Clear Current Selection =====
    def clear_selection(b=None):
        """Clear current selection but keep pairs"""
        with out:
            out.clear_output()

            for i in ['l', 'r', 'org_l', 'org_r']:
                temp_selection[i] = None
            fig.data[0].selectedpoints = None
            print("Cleared current selection")
            show_current_line()

    def recovery_selection(b=None):
        """Recovery current selection but keep pairs"""
        with out:
            out.clear_output()
            nonlocal pairs, drop_paris
    
            for i in ['l', 'r', 'org_l', 'org_r']:
                temp_selection[i] = None
            fig.data[0].selectedpoints = None
    
            if  len(drop_paris)==0:
                print("No pairs to recovery")
                return
            ipair =drop_paris.pop(-1)
            if not isinstance(ipair, (list, tuple)):
                ipair = [ipair]
            pairs += ipair
            print(f"Recovery pair(s): {len(ipair)}")

            update_pairs_display()
            counter_label.value = f"Created pairs: {len(pairs)}"
            show_all_matches()

    # ===== Remove Last Pair =====
    def remove_last_pair(b=None):
        """Remove the last added pair and its line"""
        with out:
            out.clear_output()

            if not pairs:
                print("No pairs to remove")
                return

            last_pair = pairs.pop()
            drop_paris.append(last_pair)
            # if lines:
            #     current_shapes = list(fig.layout.shapes)
            #     if current_shapes:
            #         fig.layout.shapes = tuple(current_shapes[:-1])
            #         lines.pop()

            print(f"Removed last pair: Left[{last_pair['left_idx']}] → Right[{last_pair['right_idx']}]")

            update_pairs_display()
            counter_label.value = f"Created pairs: {len(pairs)}"
            show_all_matches()

    def remove_select_pair(b=None):
        with out:
            out.clear_output()
            nonlocal pairs, drop_paris

            if not pairs:
                print("No pairs to remove")
                return

            if (btn_clearidx.value is None) or len(btn_clearidx.value) < 1:
                return

            sidx  = [int(i) for i in  re.split( r'\s*[,\s;]\s*', btn_clearidx.value) ]
            sidx = [ len(pairs) + i if i < 0 else i for i in sidx ]

            pairs_new = [ ipair for i, ipair in enumerate(pairs) if i not in sidx ]
            out_paris = [ ipair for i, ipair in enumerate(pairs) if i in sidx ]
            pairs = pairs_new
            drop_paris.append(out_paris)

            print(f"Removed indices: {sidx}")
            update_pairs_display()
            counter_label.value = f"Created pairs: {len(pairs)}"
            btn_clearidx.value =''
            show_all_matches()

    # ===== Clear All Pairs =====
    def clear_all(b=None):
        """Clear all pairs and lines"""
        with out:
            out.clear_output()

            nonlocal pairs, drop_paris
            if len(pairs):
                drop_paris = [[i for i in pairs ]]
            pairs = []
            lines = []

            for i in ['l', 'r', 'org_l', 'org_r']:
                temp_selection[i] = None
            fig.update_layout(shapes=[])
            fig.data[0].selectedpoints = None

            print("Cleared all pairs and lines")
            update_pairs_display()
            counter_label.value = f"Created pairs: {len(pairs)}"
            show_all_matches()

    # ===== Export Pairs =====
    def export_pairs(b=None):
        """Export pair data to DataFrame"""
        with out:
            out.clear_output()

            if not pairs:
                print("No pair data to export")
                return

            df = pd.DataFrame([
                {
                    'left_idx': p['left_idx'],
                    'left_x': p['left_x'],
                    'left_y': p['left_y'],
                    'right_idx': p['right_idx'],
                    'right_x': p['right_x'],
                    'right_y': p['right_y']
                }
                for p in pairs
            ])
            df[['left_idx', 'right_idx']] = df[['left_idx', 'right_idx']].astype(int)
            with pairs_out:
                pairs_out.clear_output()
                display(df)

            print("Pair data exported to table below")

    # ===== Update Pair Display =====
    def update_pairs_display():
        """Update the displayed pair information"""
        with pairs_out:
            pairs_out.clear_output()

            if not pairs:
                return

            df = pd.DataFrame([
                {
                    'Pair #': i + 1,
                    'Left Idx': p['left_idx'].astype(int),
                    'Left X': f"{p['left_x']:.3f}",
                    'Left Y': f"{p['left_y']:.3f}",
                    '→': '→',
                    'Right Idx': p['right_idx'].astype(int),
                    'Right X': f"{p['right_x']:.3f}",
                    'Right Y': f"{p['right_y']:.3f}"
                }
                for i, p in enumerate(pairs)
            ])

            display(df.style.set_properties(**{'text-align': 'center'}))

    def switch_data(b=None):
        if btn_switch.value:
            if len(fig.data[0])<=1:
                print('no pairs to show')
            else:
                for trace in fig.data[1:]:
                    trace.on_click(select_point)
                    trace.on_selection(select_point)
        else:
            fig.data[0].on_click(select_point)
            fig.data[0].on_selection(select_point)

    # ===== Bind Events =====
    # switch_data()
    # btn_switch.observe(switch_data)
    fig.data[0].on_click(select_point)
    fig.data[0].on_selection(select_point)
    btn_add_pair.on_click(add_pair)
    btn_clear_selection.on_click(clear_selection)
    btn_remove_last.on_click(remove_last_pair)
    btn_clearidxdo.on_click(remove_select_pair)
    btn_recovery_last.on_click(recovery_selection)
    btn_clear_all.on_click(clear_all)
    btn_export.on_click(export_pairs)
    btn_lineshow.observe(show_current_line)
    btn_allshow.observe(show_all_matches)
    # for trace in fig.data[:1]:
    #     trace.on_click(select_point)
    #     trace.on_selection(select_point)
    # btn_allshow.observe(show_all_matches, names='value')


    # ===== Display UI =====
    pairs_box = widgets.VBox([
        widgets.Label("Pair List:"),
        pairs_out
    ])

    dashboard = widgets.VBox([
        fig,
        button_box,
        counter_label,
        out,
        pairs_box
    ])

    display(dashboard)

    # ===== Initial Instructions =====
    with out:
        print("Instructions:")
        print("1. Select a point on the left plot (click)")
        print("2. Select a point on the right plot (click)")
        print("3. Click 'Add Pair' to create a connection")
        print("4. Use other buttons to manage pairs:")
        print("   - 'Clear Selection': Remove current point selection")
        print("   - 'Clear Last': Remove the last added pair and line")
        print("   - 'Clear All': Remove all pairs and lines")
        print("   - 'Export Pairs': Show all pair data in the table")

    return dashboard, pairs_out, pairs

def manual_select_points0(pts1, pts2, size=1.0, same_scale=True, ssize=3, samples=None,
                          seed=0, width=500, height=500):
    import ipywidgets as widgets
    from IPython.display import display

    ncells = 2
    if not samples is None:
        if isinstance(samples, (float, int)):
            samples = [samples for i in  range(ncells) ]
        elif isinstance(samples, list):
            samples = samples[:ncells]
        else:
            raise ValueError('samples should be a number or a list of numbers')

        idxs = []
        np.random.seed(seed)
        for idx, data in enumerate([pts1, pts2]):
            isize = data.shape[0]
            sample = samples[idx]
            if (sample is None) or sample == 1:
                idx = np.arange(isize)
            elif (sample>1):
                if sample  <10: print('warning: sample size is smaller than 10')
                idx = np.random.choice(isize, min(int(sample), isize), replace=False)
            elif (sample<1):
                idx = np.random.choice(isize, int(isize*sample), replace=False)
            idxs.append(idx)
        x1, y1 = pts1[idxs[0]][:,0], pts1[idxs[0]][:,1]
        x2, y2 = pts2[idxs[1]][:,0], pts2[idxs[1]][:,1]
    else:
        x1, y1 = pts1[:,0], pts1[:,1]
        x2, y2 = pts2[:,0], pts2[:,1]
        idxs = [np.arange(pts1.shape[0]), np.arange(pts2.shape[0])]

    # ===== FigureWidget =====
    fig = make_subplots(rows=1, 
                    cols=2, 
                    subplot_titles=('Left', 'Right'),
                    horizontal_spacing=0.15)

    scatl = go.Scattergl(
        x=x1,
        y=y1,
        mode='markers',
        name='Left',
        marker=dict(color='blue', size=size),
        selected=dict(marker=dict(color='red', size=ssize)),
        unselected=dict(marker=dict(opacity=0.5)),
        customdata=idxs[0],
        text=[f'{i}' for i in idxs[0]], 
        hoverinfo='text+x+y'
    )

    scatr = go.Scattergl(
        x=x2,
        y=y2,
        mode='markers',
        name='Right',
        marker=dict(color='green', size=size),
        selected=dict(marker=dict(color='orange', size=ssize)),
        unselected=dict(marker=dict(opacity=0.5)),
        customdata=idxs[1],
        text=[f'{i}' for i in idxs[1]], 
        hoverinfo='text+x+y'
    )

    fig.add_trace(scatl, row=1, col=1)
    fig.add_trace(scatr, row=1, col=2)

    fig.update_layout( width=width, height=height,
                    #    plot_bgcolor='#FFFFFF',
                       hovermode='closest',
                       clickmode='event+select',
                       margin=dict(l=1, r=1, t=30, b=5), )

    fig.update_xaxes(range=[x1.min(),x1.max()], row=1, col=1)
    fig.update_xaxes(range=[x2.min(),x2.max()], row=1, col=2)
    fig.update_yaxes(range=[y1.min(),y1.max()], row=1, col=1)
    fig.update_yaxes(range=[y2.min(),y2.max()], row=1, col=2)


    if same_scale:
        fig.update_yaxes( scaleanchor="x",scaleratio=1,)
    fig = go.FigureWidget(fig)

    # ===== UI Components =====
    # pairs =  pd.DataFrame(columns=['left_idx', 'right_idx', 'left_x', 'left_y', 'right_x', 'right_y'])
    # drop_paris = pd.DataFrame(columns=['left_idx', 'right_idx', 'left_x', 'left_y', 'right_x', 'right_y'])
    pairs = [] 
    drop_paris = []
    temp_selection = {'l': None, 'r': None, 'org_l':None, 'org_r':None}
    
    pairs_out = widgets.Output()  
    out = widgets.Output()
    counter_label = widgets.Label(value=f"All pairs: {len(pairs)}")
    add_label = widgets.Label(value=f"current → :")

    # Control buttons
    btn_add_pair = widgets.Button(description="Add Pair", button_style='success', icon='plus')
    btn_remove_last = widgets.Button(description="Clear Last", button_style='warning', icon='step-backward')
    btn_recovery_last = widgets.Button(description="Undo Last", button_style='warning', icon='step-forward')
    btn_clear_selection = widgets.Button(description="Clear Selection", button_style='warning', icon='eraser')
    btn_clear_all = widgets.Button(description="Clear All", button_style='danger', icon='trash')
    btn_export = widgets.Button(description="Export Pairs", button_style='info', icon='download')

    pts_size = widgets.Combobox(
        placeholder='0.5',
        description='selpts size',
        ensure_option=True,
        disabled=False
    )

    button_box = widgets.HBox([
        btn_add_pair, btn_remove_last, btn_recovery_last, btn_clear_selection,
        btn_clear_all, btn_export
    ])


    def select_point_l(trace, points, selector):
        """Handle scatter point selection"""
        with out:
            if not points.point_inds:
                return
            idx = points.point_inds[0]
            org_idx = fig.data[0].customdata[idx]
            temp_selection['l'] = idx
            temp_selection['org_l'] = org_idx

            add_label.value = f"current left: {org_idx} @ ({x1[idx]:.3f}, {y1[idx]:.3f})"
            counter_label.value = f"All pairs: {len(pairs)}"
            fig.data[0].selectedpoints = [org_idx]

            # if temp_selection["r"] is not None:
            #     add_line()

    def select_point_r(trace, points, selector):
        """Handle scatter point selection"""
        with out:
            if not points.point_inds:
                return
            idx = points.point_inds[0]
            org_idx = fig.data[1].customdata[idx]
            temp_selection['r'] = idx
            temp_selection['org_r'] = org_idx
    
            add_label.value = f"current right: {org_idx} @ ({x2[idx]:.3f}, {y2[idx]:.3f})"
            counter_label.value = f"All pairs: {len(pairs)}"
            fig.data[1].selectedpoints = [org_idx]

            # if temp_selection["l"] is not None:
            #     add_line()

    def add_pair(b=None):
        """Add a new pair"""
        with out:
            out.clear_output()
            if temp_selection['l'] is None:
                print("Please select a left point first")
                return
            else:
                lidx, org_lidx = temp_selection['l'], temp_selection['org_l']
                print(f"Add left: point {org_lidx} @ ({x1[lidx]:.3f}, {y1[lidx]:.3f})")

            if temp_selection['r'] is None:
                print("Please select a right point first")
                return
            else:
                ridx, org_ridx = temp_selection['r'], temp_selection['org_r']
                print(f"Add right: point {org_ridx} @ ({x2[ridx]:.3f}, {y2[ridx]:.3f})")

            # Save pair
            pair_info = pd.Series({
                'left_idx': org_lidx,
                'right_idx': org_ridx,
                'left_x': x1[lidx],
                'left_y': y1[lidx],
                'right_x': x2[ridx],
                'right_y': y2[ridx],
                # 'line_id': f'line_{len(lines)}'
            })
            pairs.append(pair_info)

            # Create line
            line = {
                'type': 'line',
                'x0': x1[lidx],
                'y0': y1[lidx],
                'x1': x2[ridx],
                'y1': y2[ridx],
                'line': dict(color='red', width=2, dash='dot'),
                'xref': 'x1',
                'yref': 'y1',
                'layer': 'above'
            }

            # Reset temporary selection
            for i in ['l', 'r', 'org_l', 'org_r']:
                temp_selection[i] = None
            fig.data[0].selectedpoints = None
            fig.data[1].selectedpoints = None

            print(f"Added pair #{len(pairs)}: Left[{org_lidx}] → Right[{org_ridx}]")

            # Update pair display
            update_pairs_display()
            counter_label.value = f"Created pairs: {len(pairs)}"

    def clear_selection(b=None):
        """Clear current selection but keep pairs"""
        with out:
            out.clear_output()

            for i in ['l', 'r', 'org_l', 'org_r']:
                temp_selection[i] = None

            fig.data[0].selectedpoints = None
            fig.data[1].selectedpoints = None

            print("Cleared current selection")

    def recovery_selection(b=None):
        """Recovery current selection but keep pairs"""
        with out:
            out.clear_output()

            for i in ['l', 'r', 'org_l', 'org_r']:
                temp_selection[i] = None
            fig.data[0].selectedpoints = None
            fig.data[1].selectedpoints = None
    
            if not drop_paris:
                print("No pairs to recovery")
                return
            ipair =drop_paris.pop(-1)
            pairs.append(ipair)
            print(f"Recovery last pair: Left[{ipair['left_idx']}] → Right[{ipair['right_idx']}]")

            update_pairs_display()
            counter_label.value = f"Created pairs: {len(pairs)}"

    def remove_last_pair(b=None):
        """Remove the last added pair and its line"""
        with out:
            out.clear_output()

            if not pairs:
                print("No pairs to remove")
                return

            last_pair = pairs.pop()
            drop_paris.append(last_pair)
            # if lines:
            #     current_shapes = list(fig.layout.shapes)
            #     if current_shapes:
            #         fig.layout.shapes = tuple(current_shapes[:-1])
            #         lines.pop()

            print(f"Removed last pair: Left[{last_pair['left_idx']}] → Right[{last_pair['right_idx']}]")

            update_pairs_display()
            counter_label.value = f"Created pairs: {len(pairs)}"
    
    def clear_all(b=None):
        """Clear all pairs and lines"""
        with out:
            out.clear_output()
            nonlocal pairs, drop_paris
            if len(pairs):
                drop_paris = [i for i in pairs ]
            pairs = []
            lines = []

            for i in ['l', 'r', 'org_l', 'org_r']:
                temp_selection[i] = None
            fig.update_layout(shapes=[])
            fig.data[0].selectedpoints = None
            fig.data[1].selectedpoints = None

            print("Cleared all pairs and lines")

            update_pairs_display()
            counter_label.value = f"Created pairs: {len(pairs)}"

    def update_pairs_display():
        """Update the displayed pair information"""
        with pairs_out:
            pairs_out.clear_output()
            if not pairs:
                return

            df = pd.DataFrame([
                {
                    'Pair #': i + 1,
                    'Left Idx': p['left_idx'].astype(int),
                    'Left X': f"{p['left_x']:.3f}",
                    'Left Y': f"{p['left_y']:.3f}",
                    '→': '→',
                    'Right Idx': p['right_idx'],
                    'Right X': f"{p['right_x']:.3f}",
                    'Right Y': f"{p['right_y']:.3f}"
                }
                for i, p in enumerate(pairs)
            ])

            display(df.style.set_properties(**{'text-align': 'center'}))

    def export_pairs(b=None):
        """Export pair data to DataFrame"""
        with out:
            out.clear_output()

            if not pairs:
                print("No pair data to export")
                return

            df = pd.DataFrame([
                {
                    'left_idx': p['left_idx'],
                    'left_x': p['left_x'],
                    'left_y': p['left_y'],
                    'right_idx': p['right_idx'],
                    'right_x': p['right_x'],
                    'right_y': p['right_y']
                }
                for p in pairs
            ])

            with pairs_out:
                pairs_out.clear_output()
                display(df)

            print("Pair data exported to table below")

    # fig.data[0].on_click(select_point_l)
    fig.data[0].on_selection(select_point_l)
    # fig.data[1].on_click(select_point_r)
    fig.data[1].on_selection(select_point_r)

    btn_add_pair.on_click(add_pair)
    btn_clear_selection.on_click(clear_selection)
    btn_remove_last.on_click(remove_last_pair)
    btn_recovery_last.on_click(recovery_selection)
    btn_clear_all.on_click(clear_all)
    btn_export.on_click(export_pairs)

    def axis_and_domain(which):
        if which == "left":
            ax, ay = fig.layout.xaxis, fig.layout.yaxis
        else:
            ax, ay = fig.layout.xaxis2, fig.layout.yaxis2
        return ax.domain, ax.range, ay.domain, ay.range

    def data_to_paper(x, y, which):
        (xd0, xd1), (xr0, xr1), (yd0, yd1), (yr0, yr1) = axis_and_domain(which)
        xp = xd0 + (x - xr0) / (xr1 - xr0) * (xd1 - xd0)
        yp = yd0 + (y - yr0) / (yr1 - yr0) * (yd1 - yd0)
        return float(xp), float(yp)

    # ---------------- 5. 画线 ----------------
    def add_line():
        l, r= temp_selection["l"], temp_selection["r"]
        xL, yL = x1[l], y1[l]
        xR, yR = x2[r], y2[r]
        xll, yll = data_to_paper(xL, yL, "left")
        xrr, yrr = data_to_paper(xR, yR, "right")

        shapes = list(fig.layout.shapes) if fig.layout.shapes else []
        shapes.append(dict(type="line", xref="paper", yref="paper",
                        x0=xll, y0=yll, x1=xrr, y1=yrr,
                        line=dict(width=2, color="black")))
        fig.layout.shapes = tuple(shapes)

    # ===== Display UI =====
    pairs_box = widgets.VBox([
        widgets.Label("Pair List:"),
        pairs_out
    ])
    dashboard = widgets.VBox([
        fig,

        button_box,
        counter_label,
        add_label,
        out,
        pairs_box,
    ])
    display(dashboard)

    # ===== Initial Instructions =====
    with out:
        print("Instructions:")
        print("1. Select a point on the left plot (click)")
        print("2. Select a point on the right plot (click)")
        print("3. Click 'Add Pair' to create a connection")
        print("4. Use other buttons to manage pairs:")
        print("   - 'Clear Selection': Remove current point selection")
        print("   - 'Clear Last': Remove the last added pair and line")
        print("   - 'Clear All': Remove all pairs and lines")
        print("   - 'Export Pairs': Show all pair data in the table")

    return dashboard, pairs_out, pairs

def manual_select_points1(pts1, pts2, size=1.0, same_scale=True, ssize=3, ksize=5, samples=None, pairs=None,
                         showlegend=False,
                          seed=0, width=500, height=500):
    import ipywidgets as widgets
    from IPython.display import display

    ncells = 2
    if not samples is None:
        if isinstance(samples, (float, int)):
            samples = [samples for i in  range(ncells) ]
        elif isinstance(samples, list):
            samples = samples[:ncells]
        else:
            raise ValueError('samples should be a number or a list of numbers')

        idxs = []
        np.random.seed(seed)
        for idx, data in enumerate([pts1, pts2]):
            isize = data.shape[0]
            sample = samples[idx]
            if (sample is None) or sample == 1:
                idx = np.arange(isize)
            elif (sample>1):
                if sample  <10: print('warning: sample size is smaller than 10')
                idx = np.random.choice(isize, min(int(sample), isize), replace=False)
            elif (sample<1):
                idx = np.random.choice(isize, int(isize*sample), replace=False)
            idxs.append(idx)
        x1, y1 = pts1[idxs[0]][:,0], pts1[idxs[0]][:,1]
        x2, y2 = pts2[idxs[1]][:,0], pts2[idxs[1]][:,1]
    else:
        x1, y1 = pts1[:,0], pts1[:,1]
        x2, y2 = pts2[:,0], pts2[:,1]
        idxs = [np.arange(pts1.shape[0]), np.arange(pts2.shape[0])]

    # ===== FigureWidget =====
    fig1 = go.FigureWidget()
    fig1.add_trace(
        go.Scattergl(
        x=x1,
        y=y1,
        mode='markers',
        name='Left',
        marker=dict(color='blue', size=size),
        selected=dict(marker=dict(color='red', size=ssize)),
        unselected=dict(marker=dict(opacity=0.5)),
        customdata=idxs[0]
    ))

    fig1.update_layout(title='Left', width=width, height=height, 
                    #    plot_bgcolor='#FFFFFF',
                    showlegend=showlegend,
                       hovermode='closest',
                        clickmode='event+select',
                       margin=dict(l=1, r=1, t=40, b=5), )

    fig2 = go.FigureWidget()
    fig2.add_trace(
        go.Scattergl(
        x=x2,
        y=y2,
        mode='markers',
        name='Right',
        marker=dict(color='green', size=size),
        selected=dict(marker=dict(color='orange', size=ssize)),
        unselected=dict(marker=dict(opacity=0.5)),
        customdata=idxs[1],
    ))
    fig2.update_layout(title='Right', width=width, height=height,
                    #    plot_bgcolor='#FFFFFF',
                    showlegend=showlegend,
                       hovermode='closest',
                        clickmode='event+select',
                       margin=dict(l=1, r=1, t=40, b=5), )

    if same_scale:
        fig1.update_yaxes( scaleanchor="x",scaleratio=1,)
        fig2.update_yaxes( scaleanchor="x",scaleratio=1,)

    container = widgets.HBox([fig1, fig2], 
         layout=widgets.Layout( 
        justify_content='space-around', 
        padding='1px', 
        gap='1px',
        margin='0px 0px 0px 0px',
        width='99%',
        height='99%'))

    # ===== State Tracking =====
    if pairs is None:
        pairs = []
    drop_paris = []
    temp_selection = {'l': None, 'r': None, 'org_l':None, 'org_r':None}  # Temporarily selected point indices
    lines = []  # Store all line objects

    # ===== UI Components =====
    out = widgets.Output()  # Output info box
    pairs_out = widgets.Output()  # Pair info output box

    # Control buttons
    btn_add_pair = widgets.Button(description="Add Pair", button_style='success', icon='plus')
    btn_remove_last = widgets.Button(description="Clear Last", button_style='warning', icon='step-backward')
    btn_recovery_last = widgets.Button(description="Undo Last", button_style='warning', icon='step-forward')
    btn_clear_selection = widgets.Button(description="Clear Selection", button_style='warning', icon='eraser')
    btn_clear_all = widgets.Button(description="Clear All", button_style='danger', icon='trash')
    btn_export = widgets.Button(description="Export Pairs", button_style='info', icon='download')

    pts_size = widgets.Combobox(
        placeholder='0.5',
        description='selpts size',
        ensure_option=True,
        disabled=False
    )

    button_box = widgets.HBox([
        btn_add_pair, btn_remove_last, btn_recovery_last, btn_clear_selection,
        btn_clear_all, btn_export
    ])

    # Status display
    counter_label = widgets.Label(value=f"Selected pairs: {len(pairs)}")

    # ===== Click Callback =====
    def select_point(trace, points, selector):
        """Handle scatter point selection"""
        with out:
            out.clear_output()
            if not points.point_inds:
                if trace.name == 'Left':
                    temp_selection['l'] = None
                    print("Cleared left selection")
                elif trace.name == 'Right':
                    temp_selection['r'] = None
                    print("Cleared right selection")
                return

            idx = points.point_inds[0]
            # idx = points.customdata[idx]
            org_idx = trace.customdata[idx]
            
            if trace.name == 'Left':
                temp_selection['l'] = idx
                temp_selection['org_l'] = org_idx
                print(f"Selected left: point {org_idx} @ ({x1[idx]:.3f}, {y1[idx]:.3f})")
            elif trace.name == 'Right':
                temp_selection['r'] = idx
                temp_selection['org_r'] = org_idx
                print(f"Selected right: point {org_idx} @ ({x2[idx]:.3f}, {y2[idx]:.3f})")

            counter_label.value = f"All pairs: {len(pairs)}"
            trace.selectedpoints = [org_idx]

    # ===== Add Pair =====
    def add_pair(b=None):
        """Add a new pair"""
        with out:
            out.clear_output()
            if temp_selection['l'] is None:
                print("Please select a left point first")
                return
            else:
                lidx, org_lidx = temp_selection['l'], temp_selection['org_l']
                print(f"Add left: point {org_lidx} @ ({x1[lidx]:.3f}, {y1[lidx]:.3f})")

            if temp_selection['r'] is None:
                print("Please select a right point first")
                return
            else:
                ridx, org_ridx = temp_selection['r'], temp_selection['org_r']
                print(f"Add right: point {org_ridx} @ ({x2[ridx]:.3f}, {y2[ridx]:.3f})")

            # Save pair
            pair_info = pd.Series({
                'left_idx': org_lidx,
                'right_idx': org_ridx,
                'left_x': x1[lidx],
                'left_y': y1[lidx],
                'right_x': x2[ridx],
                'right_y': y2[ridx],
                # 'line_id': f'line_{len(lines)}'
            })
            pairs.append(pair_info)

            # Create line
            line = {
                'type': 'line',
                'x0': x1[lidx],
                'y0': y1[lidx],
                'x1': x2[ridx],
                'y1': y2[ridx],
                'line': dict(color='red', width=2, dash='dot'),
                'xref': 'x1',
                'yref': 'y1',
                'layer': 'above'
            }

            # Reset temporary selection
            for i in ['l', 'r', 'org_l', 'org_r']:
                temp_selection[i] = None
            fig1.data[0].selectedpoints = None
            fig2.data[0].selectedpoints = None

            print(f"Added pair #{len(pairs)}: Left[{org_lidx}] → Right[{org_ridx}]")

            # Update pair display
            update_pairs_display()
            counter_label.value = f"Created pairs: {len(pairs)}"
            add_scatter_points()

    def add_scatter_points(b = None):
        if len(pairs) == 0:
            return
        points_df = pd.concat(pairs, axis=1).T
        #left_idx  right_idx       left_x       left_y      right_x      right_y
        points_df['color'] =  random_colors(len(points_df), seed=seed)
        points_df['r_text'] = points_df.index.astype(str) + '_' + points_df['right_idx'].astype(int).astype(str)
        points_df['l_text'] = points_df.index.astype(str) + '_' + points_df['left_idx'].astype(int).astype(str)

        al1 = dict(
                x=points_df['left_x'],
                y=points_df['left_y'],
                mode='markers',
                name ='selected',
                # name=points_df.index.astype(str),
                marker=dict(color=points_df['color'], size=ksize),
                selected=dict(marker=dict(size=ksize, opacity=1.0)),
                unselected=dict(marker=dict(size=ksize,opacity=1.0)),
                text=points_df['l_text'],
                hoverinfo='text+x+y'
        )
        ar1 = dict(
            x=points_df['right_x'],
            y=points_df['right_y'],
            mode='markers',
            name ='selected',
            # name=points_df.index.astype(str),
            marker=dict(color=points_df['color'], size=ksize),
            selected=dict(marker=dict(size=ksize, opacity=1.0)),
            unselected=dict(marker=dict(size=ksize,opacity=1.0)),
            text=points_df['r_text'], 
            hoverinfo='text+x+y'
        )

        if len(fig1.data) == 1:
            fig1.add_trace(go.Scattergl(**al1))
        else:
            fig1.data[1].update(al1)

        if len(fig2.data) == 1:
            fig2.add_trace(go.Scattergl(**ar1))
        else:
            fig2.data[1].update(ar1)

    # ===== Clear Current Selection =====
    def clear_selection(b=None):
        """Clear current selection but keep pairs"""
        with out:
            out.clear_output()

            for i in ['l', 'r', 'org_l', 'org_r']:
                temp_selection[i] = None

            fig1.data[0].selectedpoints = None
            fig2.data[0].selectedpoints = None

            print("Cleared current selection")

    def recovery_selection(b=None):
        """Recovery current selection but keep pairs"""
        with out:
            out.clear_output()

            for i in ['l', 'r', 'org_l', 'org_r']:
                temp_selection[i] = None
            fig1.data[0].selectedpoints = None
            fig2.data[0].selectedpoints = None
    
            if not drop_paris:
                print("No pairs to recovery")
                return
            ipair =drop_paris.pop(-1)
            pairs.append(ipair)
            print(f"Recovery last pair: Left[{ipair['left_idx']}] → Right[{ipair['right_idx']}]")

            update_pairs_display()
            counter_label.value = f"Created pairs: {len(pairs)}"
            add_scatter_points()

    # ===== Remove Last Pair =====
    def remove_last_pair(b=None):
        """Remove the last added pair and its line"""
        with out:
            out.clear_output()

            if not pairs:
                print("No pairs to remove")
                return

            last_pair = pairs.pop()
            drop_paris.append(last_pair)
            # if lines:
            #     current_shapes = list(fig.layout.shapes)
            #     if current_shapes:
            #         fig.layout.shapes = tuple(current_shapes[:-1])
            #         lines.pop()

            print(f"Removed last pair: Left[{last_pair['left_idx']}] → Right[{last_pair['right_idx']}]")

            update_pairs_display()
            counter_label.value = f"Created pairs: {len(pairs)}"
            add_scatter_points()

    # ===== Clear All Pairs =====
    def clear_all(b=None):
        """Clear all pairs and lines"""
        with out:
            out.clear_output()

            nonlocal pairs, drop_paris
            if len(pairs):
                drop_paris = [i for i in pairs ]
            pairs = []
            lines = []

            for i in ['l', 'r', 'org_l', 'org_r']:
                temp_selection[i] = None
            fig1.update_layout(shapes=[])
            fig2.update_layout(shapes=[])

            fig1.data[0].selectedpoints = None
            fig2.data[0].selectedpoints = None

            print("Cleared all pairs and lines")

            update_pairs_display()
            counter_label.value = f"Created pairs: {len(pairs)}"

    # ===== Export Pairs =====
    def export_pairs(b=None):
        """Export pair data to DataFrame"""
        with out:
            out.clear_output()

            if not pairs:
                print("No pair data to export")
                return

            df = pd.DataFrame([
                {
                    'left_idx': p['left_idx'],
                    'left_x': p['left_x'],
                    'left_y': p['left_y'],
                    'right_idx': p['right_idx'],
                    'right_x': p['right_x'],
                    'right_y': p['right_y']
                }
                for p in pairs
            ])

            with pairs_out:
                pairs_out.clear_output()
                display(df)

            print("Pair data exported to table below")

    # ===== Update Pair Display =====
    def update_pairs_display():
        """Update the displayed pair information"""
        with pairs_out:
            pairs_out.clear_output()

            if not pairs:
                return

            df = pd.DataFrame([
                {
                    'Pair #': i + 1,
                    'Left Idx': p['left_idx'],
                    'Left X': f"{p['left_x']:.3f}",
                    'Left Y': f"{p['left_y']:.3f}",
                    '→': '→',
                    'Right Idx': p['right_idx'],
                    'Right X': f"{p['right_x']:.3f}",
                    'Right Y': f"{p['right_y']:.3f}"
                }
                for i, p in enumerate(pairs)
            ])

            display(df.style.set_properties(**{'text-align': 'center'}))

    # ===== Bind Events =====
    fig1.data[0].on_click(select_point)
    fig1.data[0].on_selection(select_point)
    fig2.data[0].on_click(select_point)
    fig2.data[0].on_selection(select_point)
    btn_add_pair.on_click(add_pair)
    btn_clear_selection.on_click(clear_selection)
    btn_remove_last.on_click(remove_last_pair)
    btn_recovery_last.on_click(recovery_selection)
    btn_clear_all.on_click(clear_all)
    btn_export.on_click(export_pairs)

    # ===== Display UI =====
    pairs_box = widgets.VBox([
        widgets.Label("Pair List:"),
        pairs_out
    ])

    dashboard = widgets.VBox([
        container,
        button_box,
        counter_label,
        out,
        pairs_box
    ])

    display(dashboard)

    # ===== Initial Instructions =====
    with out:
        print("Instructions:")
        print("1. Select a point on the left plot (click)")
        print("2. Select a point on the right plot (click)")
        print("3. Click 'Add Pair' to create a connection")
        print("4. Use other buttons to manage pairs:")
        print("   - 'Clear Selection': Remove current point selection")
        print("   - 'Clear Last': Remove the last added pair and line")
        print("   - 'Clear All': Remove all pairs and lines")
        print("   - 'Export Pairs': Show all pair data in the table")

    return dashboard, pairs_out, pairs