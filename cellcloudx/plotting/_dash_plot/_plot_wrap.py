import numpy as np
import pandas as pd
import numpy as np
import plotly.express as px 

from ...utilis._arrays import list_iter, vartype
from .._colors import *

def scatter_wrap(Meta: dict, Paras : dict, idx : int, 
               scale=1,  same_scale=True, 
               size=None, opacity=None, size_max=20,
               save=None,
               figscale=800, werror=20, 
                render_mode='auto', 
                width = None, height=None, 
                reverse_y=False, reverse_x=False,  reverse_z=False,
                showlegend=True,
                aspectmode='data',  
                aspectratio=dict(x=1, y=1, z=1), 
                clip=None, sample=None,
                random_state = 200504, order ='guess',
                show_grid =True, 
                showticklabels = False,
                category_color = [],
                colormap=None, 
                template='none',
                legend_dict={ 
                    'itemsizing':'constant',
                    'itemwidth': 30},
                colorbar_dict = dict(
                    thicknessmode="pixels", 
                    thickness=25,
                    lenmode="pixels", 
                    len=300,
                    yanchor="middle",
                    y=0.5,
                    ypad=10,
                    ticks="outside", ),
                **kargs):
    
    def NONEtoNone(x): return (None if x in ['NONE', None, 'None'] else x)

    group = Paras['window_labels'][idx]
    split = NONEtoNone(Paras['window_splits'][idx])
    split_iterm = Paras['split_iterms'][idx]
    animation_frame = NONEtoNone(Paras['window_slices'][idx])
    basis = Paras['window_basis'][idx]
    colormap = colormap or Paras['colormaps'][idx]
    category_color = category_color or Paras['colors'].get(group, [])
    category_order = { i: Paras['colors_order'].get(i, None)
                      for i in [group, animation_frame]  }
    category_order = {k:v for k,v in category_order.items() 
                      if v is not None}

    bData = np.array(Meta['locs'][basis])
    nbasis =  min(bData.shape[1], 3)
    hbasis = list('xyz')[:nbasis]

    Data = pd.DataFrame({ i:Meta['metadata'][i] 
                          for i in [group, split, animation_frame, size] 
                          if i is not None }, index=Meta['index'])
    Data['index'] = Meta['index']
    Data[hbasis] = bData[:,:nbasis]
    Data['custom_index'] = np.arange(len(Data)).astype(str)

    if (not sample is None) and (sample <1):
        if (type(sample) == int) and (sample>1):
            Data = Data.sample(n=sample, 
                                    replace=False,
                                    random_state=random_state)
        if (type(sample) == float) and (sample<=1):
            Data = Data.sample(frac=sample, 
                                    replace=False,
                                    random_state=random_state)

    if bool(split_iterm) and (not split_iterm in ['ALL_', 'NONE', ['ALL_'], ['NONE']]):
        if type(split_iterm) == str:
            Data = Data[(Data[split]==split_iterm)]
        elif (type(split_iterm) == list):
            Data = Data[(Data[split].isin(split_iterm))]

    dimdict = dict(zip(hbasis, hbasis))
    dimdict['custom_data'] = [ 'custom_index' ]
    dimdict['animation_frame'] = animation_frame

    xyzrange = {
        f'{i}axis': dict(visible=show_grid, showticklabels=showticklabels, 
                          range= [float(Data[i].min()), float(Data[i].max())])
        for i in hbasis
    }
    # if (animation_frame is not None):
    #     if category_order.get(animation_frame, None) is not None:
    #         Data[animation_frame] = pd.Categorical(Data[animation_frame], 
    #                                         categories=category_order[animation_frame])

    #     Data.sort_values(by=animation_frame, inplace=True)
    #dimdict['animation_group'] = animation_frame
    # for i in hbasis:
    #     xyzrange[f'{i}axis']['range'] = [float(Data[i].min()), float(Data[i].max())]

    if nbasis == 2:
        dimdict['render_mode'] = render_mode

    if size is not None:
        dimdict['size_max'] = size_max
        dimdict['size'] = size

    if not group is None:
        ctype   = vartype(Data[group])
        dimdict['color']=group
        if ctype == 'discrete':
            colors_order = category_order.get(group, [])
            try:
                order = Data[group].cat.categories.tolist()
            except:
                order = Data[group].unique().tolist()
            for i in order:
                if i not in colors_order:
                    colors_order.append(i)

            IColor = []
            if (category_color is None) or (len(category_color)==0):
                IColor = color_palette(len(colors_order))
            for icor in category_color:
                if (type(icor)== str) and (icor.startswith('#')):
                    IColor.append(icor[:7])
                elif icor is None:
                    IColor.append(random_colors[1][0])
            IColor = IColor + random_colors(len(colors_order) - len(IColor))

            Paras['colors'][group] = IColor # update colors
            Paras['colors_order'][group] = colors_order # update colors orders
            category_order[group] = colors_order

            dimdict.update({'category_orders': category_order, 
                            'color_discrete_sequence':IColor}) #'animation_frame': group
        elif ctype == 'continuous':
            cmin = Data[group].min()
            if not clip is None:
                Data[group] = np.clip(Data[group], clip[0], clip[1])
            if order =='guess' or  order == True:
                Data = Data.sort_values(by = group, ascending=True)
            IColor = get_px_colormap(colormap)

            cmax = Data[group].max()
            dimdict.update({'color_continuous_scale': IColor, 
                            'range_color': [float(cmin), float(cmax)]})

    if figscale:
        werror = werror or 20
        # width = figscale+werror if width is None else width
        height= figscale if height is None else height

    # Data = Data.reset_index(drop=True) # not work
    pxscatter = px.scatter_3d if nbasis == 3 else px.scatter
    fig = pxscatter(Data, hover_name="custom_index",
                     hover_data=["index"],
                     width=width, height=height,
                    **dimdict, **kargs)

    if reverse_y:
        fig.update_yaxes(autorange="reversed")
    if reverse_x:
        fig.update_xaxes(autorange="reversed")
    if (nbasis == 3) and reverse_z:
        fig.update_scenes(zaxis_autorange="reversed")

    fig.update_layout(#legend=dict(itemsizing = 'constant'),
                        legend=legend_dict,
                        coloraxis_colorbar=colorbar_dict,
                        showlegend=showlegend,
                        scene=dict(
                            # aspectratio=aspectratio, 
                            aspectmode=aspectmode,
                            **xyzrange,
                        ),
                        # plot_bgcolor='#FFFFFF',
                        # title=f"current type: {group}",
                        # xaxis_title="X Axis",
                        # yaxis_title="Y Axis",
                        dragmode='select' , # select tools
                        margin=dict(l=5, r=5, t=30, b=5),
                        template=template,
                        annotations=[],
                        ) #

                        #paper_bgcolor='#000000',
                        #plot_or='#000000'

    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )
    # fig.update_traces(text=[{"index": Data.index}], 
    #                     # selected=dict(marker=dict(color='red', size=12)),
    #                     # unselected=dict(marker=dict(opacity=0.5)) 
    #                 )

    if not size is None:
        fig.update_traces(marker=dict(opacity=opacity, line=dict(width=0,color='DarkSlateGrey')),
                            selector=dict(mode='markers'))
    else:
        fig.update_traces(marker=dict(opacity=opacity, size=scale, line=dict(width=0,color='DarkSlateGrey')),
                            selector=dict(mode='markers'))
        
    if save:
        if save.endswith('.html'):
            fig.write_html(save)
        else:
            fig.write_image(save)

    return fig
