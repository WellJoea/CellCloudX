import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec
from scipy.spatial import distance
import scipy.cluster.hierarchy as sch
import numpy as np
import pandas as pd
from ._colors import random_colors, cmaptolist
plt.rcdefaults()

def todigitize(X, bins=256):
    low = np.min(X)
    high = np.max(X)
    margin = max(1e-10, 1e-10 * (high - low))
    expanded_low = low - margin
    expanded_high = high + margin

    bins = np.linspace(expanded_low, expanded_high, bins + 1)

    indices = np.digitize(X, bins, right=False) - 1
    return indices

def value2color(X, cmap='viridis', color_dict={},  bins=256, seed=None):
    if np_dtype(X) == "num":
        X = pd.DataFrame(todigitize(X, bins=bins))
        corlist = cmaptolist(cmap, n_color=bins)
        X = X.map(lambda x: corlist[x])
    elif np_dtype(X) == "str":
        X = pd.DataFrame(X).copy()
        if len(color_dict) == 0:
            uX = np.unique(X)
            corlist = random_colors(len(uX), seed = seed)
            color_dict = dict(zip(uX, corlist))
        X = X.map(lambda x: color_dict[x])
    else:
        raise ValueError("X must be numeric or string")
    return X

def color2value(X):
    X = pd.DataFrame(X).copy()

    colors = np.unique(X)
    color_l = dict(zip(colors, np.arange(len(colors)).astype(str).tolist()))

    X = (X.map(lambda x: color_l[x])).astype(np.int64)
    return (X, colors)

def check_cmap( imaps):
    if isinstance(imaps, str):
        icmap = plt.get_cmap(imaps)
    elif isinstance(imaps, list):
        icmap = mpl.colors.ListedColormap(imaps)
    else:
        icmap = imaps
    return icmap


def trans_cmap_tolist(imaps):
    icmap = check_cmap(imaps)
    icmap = list(map(mpl.colors.rgb2hex, icmap.colors))
    return (icmap)

def np_dtype(arr):
    arr = np.array(arr)
    if arr.size == 0:
        return None
    if arr.dtype.kind in 'iufc':
        return "num"
    elif arr.dtype.kind in 'SU':
        return "str"
    elif arr.dtype.kind == 'O': 
        ielem = arr[(0,) * arr.ndim]
        if isinstance(ielem, str):
            return "str"
        elif isinstance(ielem, (int, float, complex)):
            return "num"
        else:
            return f"{type(ielem)}"
    else:
        return arr.dtype.kind

def dendrogram_plot( mtrx,
                    dist_mtx=None,
                    rownames=None,
                    cor_method='pearson',
                    method='complete',
                    metric='euclidean',
                    fastclust=True,
                    color_threshold=None,
                    leaf_rotation=None,
                    link_colors=list(map(mpl.colors.rgb2hex, plt.get_cmap('tab20').colors)),
                    optimal_ordering=False,
                    grid=False, 
                    ticks_off=True,
                    axis_off=True,
                    ax=None,
                    **kargs):

    if rownames is None:
        try:
            rownames = mtrx.index.tolist()
        except:
            rownames = np.arange(mtrx.shape[0])
    mtrx = pd.DataFrame(mtrx)
    if not dist_mtx is None:
        corr_condensed = dist_mtx
    elif cor_method in ['pearson', 'kendall', 'spearman']:
        corr_matrix = mtrx.T.corr(method=cor_method)
        corr_condensed = distance.squareform(1 - corr_matrix)
    elif cor_method in ['sknormal']:
        from sklearn.preprocessing import normalize
        corr_condensed = normalize(mtrx)
    else:
        corr_condensed = mtrx

    if fastclust:
        import fastcluster
        z_var = fastcluster.linkage(corr_condensed, method=method,
                                    metric=metric)
    else:
        z_var = sch.linkage(corr_condensed,
                            method=method,
                            metric=metric,
                            optimal_ordering=optimal_ordering)

    sch.set_link_color_palette(link_colors)
    dendro_info = sch.dendrogram(z_var,
                                    labels=rownames,
                                    leaf_rotation=leaf_rotation,
                                    color_threshold=color_threshold,
                                    ax=ax,
                                    **kargs)

    if axis_off:
        ax.set_axis_off()
    if ticks_off:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    ax.grid(grid)

    return (dendro_info)

def heat_plot(mtrx, ax=None, cmap=None,  color_dict = None,
              rownames=None, colnames=None, grid=False, 
              ticks_off=True, axis_off=False,
              edgecolors='k', linewidths=0,
              set_frame_on=True, seed =None ,**kargs):
    mtrx = pd.DataFrame(mtrx).iloc[::-1,:].copy()
    if mtrx.ndim == 1:
        mtrx = mtrx[:, np.newaxis]
    assert mtrx.ndim == 2

    rownames = mtrx.index if rownames is None else rownames
    colnames = mtrx.columns if colnames is None else colnames

    dtype = np_dtype(mtrx.values)

    if dtype == "num":
        mtrx_num = mtrx
        cmap_ = 'viridis'
    elif dtype == "str":
        all_label = np.unique(mtrx.values)
    
        mtrx_num = pd.DataFrame(mtrx.values).replace(dict(zip(all_label, range(all_label.shape[0]))))
        mtrx_num = mtrx_num.astype(np.int64)
        if color_dict is None:
            colors = random_colors(all_label.shape[0], seed=seed)
        else:
            colors = [ color_dict[i] for i in all_label]
        cmap_ = mpl.colors.ListedColormap(colors)
    else:
        raise ValueError("Unsupported data type {dtype}")
    cmap = cmap_ if cmap is None else cmap

    ax.pcolormesh(mtrx_num, cmap=cmap, 
                  edgecolors=edgecolors, linewidths=linewidths,**kargs)

    x_ticks = np.arange(mtrx_num.shape[1]) + 0.5
    y_ticks = np.arange(mtrx_num.shape[0]) + 0.5
    ax.set_yticks(y_ticks)
    ax.set_xticks(x_ticks)
    ax.set_frame_on(set_frame_on)
    ax.set_yticklabels(rownames,
                        # rotation=90,
                        minor=False)
    ax.set_xticklabels(colnames,
                        # rotation=90,
                        # ha='center',
                        minor=False)
    if axis_off:
        ax.set_axis_off()
    if ticks_off:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    ax.grid(grid)

def dot_plot(size_df,
                color_df=None,
                colnames = None,
                rownames = None,
                ax=None,
                # fig=None,
                grid=True,
                color_on='dot',
                set_aspect='auto',
                axis_off=False,
                ticks_off=False,

                show_yticks=False,
                ticklabels_fs=None,
                xticklabels_rotation=90,
                legend_fs=8,
                yticks_pos='right',
                show_xticks=True,
                size_scale=None,
                color_scale=None,
                cmap=None,
                cax=None,
                scax=None,
                facecolor='none',
                grid_color='lightgrey',
                grid_linestyle='--',
                grid_linewidth=0.25,
                size_title='scale_size',
                color_title='color_scale',
                size_min=None, size_max=None, col_max=None, col_min=None,
                col_ticks=5,
                max_size=150,
                rigth_dist=None,
                fmt="%.1f",
                show_legend=True,
                pad=0.05, pwidth=3, cwidth=0.03, cheight=0.2,
                swidth=0.06, sheight=0.4):
    '''
    fig, ax= plt.subplots(1,1)
    sig_mean = np.random.randint(0,19, size=(20, 15))
    sig_padj = np.random.rand(20, 15)
    dot_plot(sig_mean, color_df=sig_padj, ax=ax, col_ticks=8)
    '''
 
    size_df = pd.DataFrame(size_df).copy()
    col_df = size_df.copy() if color_df is None else pd.DataFrame(color_df).copy()
    assert size_df.shape == col_df.shape

    colnames = size_df.columns.values if colnames is None else colnames
    rownames = size_df.index.values if rownames is None else rownames

    smin =size_df.min().min() if size_min is None else size_min
    smax = size_df.max().max() if size_max is None else size_max
    cmin = col_df.min().min() if col_min is None else col_min
    cmax = col_df.max().max() if col_max is None else col_max

    if size_scale == 'max':
        size_df = size_df / size_df.max().max()
    elif size_scale == 'log1p':
        size_df = np.log1p(size_df)
    elif size_scale == 'row':
        size_df = size_df.divide(size_df.max(1), axis=0)

    slmin = size_df.min().min() if size_min is None else size_min
    slmax = size_df.max().max() if size_max is None else size_max
    size_df = size_df.clip(slmin, slmax)

    if color_scale == 'row':
        col_df = col_df.divide(col_df.max(1), axis=0)

    # size_df = size_df.iloc[yorder[::-1], xorder].copy()
    # col_df= col_df.iloc[yorder[::-1], xorder].copy()

    col_df = col_df / col_df.max().max()
    col_df = plt.get_cmap(cmap)(col_df)
    col_df = np.apply_along_axis(mpl.colors.rgb2hex, -1, col_df).flatten()

    # xx,yy=np.meshgrid(xorder, yorder)
    yy, xx = np.indices(size_df.shape)
    yy = yy.flatten() + 0.5
    xx = xx.flatten() + 0.5
    # scat = ax.pcolor(size_df, cmap=None, shading='auto')
    msize_df = (size_df.copy() / size_df.max().max()) * max_size
    scat = ax.scatter(xx, yy, s=msize_df.values, c=col_df,
                    #    cmap=cmap, 
                       edgecolor=None)

    y_ticks = np.arange(size_df.shape[0]) + 0.5
    ticklabels_fs = plt.rcParams['axes.titlesize'] if ticklabels_fs is None else ticklabels_fs
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([rownames[idx] for idx, _ in enumerate(y_ticks)],
                        minor=False,
                        fontdict={'fontsize': ticklabels_fs}
                        )

    x_ticks = np.arange(size_df.shape[1]) + 0.5
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(
        [colnames[idx] for idx, _ in enumerate(x_ticks)],
        rotation=xticklabels_rotation,
        ha='center',
        minor=False,
        fontdict={'fontsize': ticklabels_fs}
    )
    # ax.tick_params(axis='both', labelsize='small')
    # ax.set_ylabel(y_label)

    if yticks_pos == 'right':
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
    # Invert the y axis to show the plot in matrix form
    # ax.invert_yaxis()
    # ax.xaxis.set_label_position("top")
    # ax.xaxis.tick_top()

    ax.set_ylim(size_df.shape[0], 0)
    ax.set_xlim(0, size_df.shape[1])
    if color_on == 'dot':
        x_padding = 0.5
        y_padding = 0.5
        x_padding = x_padding - 0.5
        y_padding = y_padding - 0.5
        ax.set_ylim(size_df.shape[0] + y_padding, -y_padding)
        ax.set_xlim(-x_padding, size_df.shape[1] + x_padding)

    ax.set_aspect(set_aspect)
    if axis_off:
        ax.set_axis_off()
    if ticks_off:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    if grid:
        ax.set_axisbelow(True)
        ax.grid(color=grid_color, linestyle=grid_linestyle, linewidth=grid_linewidth)
    else:
        # ax.margins(x=2/size_df.shape[1], y=2/size_df.shape[0])
        ax.grid(False)
    if not show_yticks:
        # ax.set_yticks([])
        ax.yaxis.set_ticklabels([])
    if not show_xticks:
        # ax.set_xticks([])
        ax.xaxis.set_ticklabels([])

    # if not xlim is None:
    #     ax.set_xlim(*xlim)
    # if not ylim is None:
    #     ax.set_ylim(*ylim)

    ax.set_facecolor(facecolor)
    
    if show_legend:
        if rigth_dist is None:
            if not show_yticks:
                rigth_dist = 1.01
            else:
                rigth_dist = 1.21
        if cax == 'full':
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size=f"{pwidth}%", pad=pad)
        elif (not ax in [plt]) and (cax in [None, 'part']):
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            cax = inset_axes(ax,
                                width=f"{cwidth * 100}%",
                                height=f"{cheight * 100}%",  # height : 50%
                                loc='lower left',
                                bbox_to_anchor=(rigth_dist, 0., 1, 1),
                                bbox_transform=ax.transAxes,
                                borderpad=0)
        elif cax is None:
            cax = [1.01, 0.5, 0.05, 0.3]
            cax = plt.axes(cax)
        else:
            cax = plt.axes(cax)

        # cbar = plt.colorbar(scat, cax=cax, ticks=np.linspace(0,1,5))
        cbar = plt.colorbar(
            mpl.cm.ScalarMappable(cmap=cmap),
            cax=cax, 
            ax=ax,
            # ticks=np.linspace(0, 1, col_ticks)
            )
        labels = [fmt % (i) for i in np.linspace(cmin, cmax, col_ticks)]
        cbar.set_ticks(ticks=np.linspace(0, 1, col_ticks), labels=labels)
        # cbar.set_ticklabels(np.linspace(cmin, cmax, col_ticks), format=fmt,)
        cbar.ax.set_title(color_title, size=legend_fs, loc='left', rotation=0)
        cbar.ax.tick_params(labelsize=legend_fs )


        pos = ax.get_position()
        bbpos = [rigth_dist, pos.y0 + 0.5, swidth, sheight] if scax is None else scax
        kw = dict(color=scat.cmap(0.7)) if color_df is None else dict(alpha=0.8)
        handles, labels = scat.legend_elements(prop="sizes", num=col_ticks, **kw)
        # labels = ['20', '40', '60', '80' ,'100']
        labels = [fmt % (i) for i in np.linspace(smin, smax, col_ticks+1)[-len(handles):]]

        legend2 = ax.legend(handles, labels, loc='center left',
                            alignment='left',
                            title=f'{size_title}',
                            title_fontsize=legend_fs,
                            bbox_to_anchor=bbpos,
                            fontsize=legend_fs, )


def clust_dot(
        size_df, color_df=None,
        colnames=None, rownames=None,
        figsize=None,

        width_ratios=None,
        height_ratios=None,
        sharex=False,
        sharey=False,
        wspace=0,
        hspace=0,
        layout='constrained',
        fig_kw = {},


        rowcolor =  'tab20' ,
        row_cluster=False,
        row_annot=None,
        row_annot_pos= 'left',

        colcolor =  'tab20b',
        col_cluster=False,
        col_annot=None,
        col_annot_pos= 'top',

        gskws=None,

        rowdendkws={},
        coldendkws={},
        row_threshold=None,
        col_threshold=None,

        rowannotkws = {},
        rowannot_colors = {},
        colannotkws = {},
        colannot_colors = {},
        annot_linewidths=0.2,
        annot_edgecolor='white',
        annot_cmap = 'viridis',

        dot_cmap='viridis',
        dotkws={},
        max_size=150,
        show_yticks=True,
        show_xticks=True,
        grid_color='lightgrey',
        grid_linestyle='--',
        grid_linewidth=0.25,

        save=None,
        dpi=800, transparent=True,
        seed =None,
        show=True):

    '''
    sig_mean = np.random.randint(0,19, size=(20, 15))
    sig_padj = np.random.rand(20, 15)


    add_col_annot = np.random.choice(list('ABCD'), size=(15, 3), replace=True)
    add_row_annot = np.random.rand(20, 3)

    clust_dot(sig_mean, color_df=sig_padj,
                # row_threshold=0,
                # col_threshold=0,
                add_row_annot=add_row_annot,
                add_col_annot=add_col_annot,
                row_cluster=True,
                col_cluster=True,
                figsize=(15, 10),
                # save='significant.mean.dotplot.size.mean.color.pvalue.cluster.pdf',
                dotkws={"size_title": 'omega weight',
                        'ticklabels_fs': 8,
                        'ticklabels_fs': 8,
                        'sheight': 0.3,
                        'cheight': 0.2,
                        "color_title": 'omega weight', },
                #     width_ratios=[1,1, 15],
                #     height_ratios=[1.4,1.5,15]
                )
    '''

    size_df = pd.DataFrame(size_df).copy()
    color_df = size_df.copy() if color_df is None else pd.DataFrame(color_df).copy()
    assert size_df.shape == color_df.shape

    colnames = size_df.columns.values if colnames is None else colnames
    rownames = size_df.index.values if rownames is None else rownames

    # row annot
    row_axidx = ['scatter']
    if not row_annot is None:
        assert row_annot.shape[0] == size_df.shape[0]
        row_axidx.append('heatx')
    if row_cluster: row_axidx.append('clustx')
    ncols = len(row_axidx)
    width_ratio_ = [8] + [1] * (ncols - 1) 

    # col annot
    col_axidx = []
    if col_cluster: col_axidx.append(['clusty'] + list('.' * (ncols - 1)))
    if not col_annot is None:
        assert col_annot.shape[0] == size_df.shape[1]
        col_axidx.append(['heaty'] + list('.' * (ncols - 1)))
    nrows = len(col_axidx) + 1
    height_ratio_ = [1] * (nrows - 1) + [8] 

    # mosaic
    ax_mosaic = col_axidx + [row_axidx]
    if row_annot_pos == 'left':
        ax_mosaic = [ i[::-1] for i in ax_mosaic ]
        width_ratio_ = width_ratio_[::-1]
    if col_annot_pos == 'bottom':
        ax_mosaic = ax_mosaic[::-1]
        height_ratio_ = height_ratio_[::-1]
    width_ratios = width_ratio_ if width_ratios is None else width_ratios
    height_ratios = height_ratio_ if height_ratios is None else height_ratios

    gskws = {} if gskws is None else gskws
    gskws.update(dict(wspace=wspace,
                     hspace=hspace,
                        # width_ratios=width_ratios,
                        # height_ratios=height_ratios,
                        ))
    gskws.update(fig_kw.get('gridspec_kw', {}))
    fig_kw.update(gridspec_kw=gskws)

    fig, axs = plt.subplot_mosaic(
                            ax_mosaic,
                            figsize=(10, 10) if figsize is None else figsize,
                            width_ratios=width_ratios, height_ratios=height_ratios,
                            sharex=sharex, sharey=sharey, 
                            layout=layout, 
                            **fig_kw)

    ## plot figure
    annotkws=dict(
        linewidths=annot_linewidths,
        edgecolor=annot_edgecolor,
        cmap=annot_cmap,  color_dict = None,
        edgecolors='k', 
        set_frame_on=True, seed =seed)
    
    ## row annot
    row_index = range(size_df.shape[0])
    row_heat = pd.DataFrame(index=rownames)
    if row_cluster:
        # rowdendkws = {}
        # axs['clustx'].sharey(axs['scatter'])
        rowdendkws.update(dict(color_threshold=row_threshold,
                            no_plot=True,
                            link_colors=trans_cmap_tolist(rowcolor)))
        rowdendkws.update(dict(
                    leaf_rotation=0,
                    no_plot=False,
                    ax=axs['clustx'],
                    rownames=rownames,
                    orientation=row_annot_pos))

        row_dendrog = dendrogram_plot(size_df, **rowdendkws)
        axs['clustx'].invert_yaxis()
        y_ticks = np.arange(size_df.shape[0]) + 0.5
        axs['clustx'].set_yticks(y_ticks)

        row_index = np.array(row_dendrog['leaves'], dtype=np.int64)
        row_heat = pd.DataFrame(
                        {'cluster': row_dendrog['leaves_color_list']},
                            index=rownames[row_index])

    if not row_annot is None:
        axs['heatx'].sharey(axs['scatter'])
        assert row_annot.shape[0] == size_df.shape[0]
        row_annot = pd.DataFrame(row_annot).iloc[row_index]
        row_annot.index = rownames[row_index]
        rowannotkws = {**annotkws, **rowannotkws, 
                       'ax': axs['heatx'],}

        row_df = value2color(row_annot, bins=256, seed=seed,
                             color_dict=rowannot_colors,
                             cmap=rowannotkws['cmap'] )
        row_df.index=row_annot.index
        row_df.columns=row_annot.columns

        if row_annot_pos == 'left':
            row_df = pd.concat([row_heat, row_df], axis=1)
        else:
            row_df = pd.concat([row_df, row_heat], axis=1)
        row_df, colors = color2value(row_df)

        rowannotkws['cmap'] = mpl.colors.ListedColormap(colors)
        heat_plot(row_df, **rowannotkws)

    ## col annot
    col_index = range(size_df.shape[1])
    col_heat = pd.DataFrame(index=colnames)
    if col_cluster:
        # axs['clusty'].sharey(axs['scatter'])
        # coldendkws = {}
        coldendkws.update(dict(color_threshold=col_threshold,
                            no_plot=True,
                            link_colors=trans_cmap_tolist(colcolor)))
        coldendkws.update(dict(
                    leaf_rotation=90,
                    no_plot=False,
                    ax=axs['clusty'],
                    rownames=colnames,
                    orientation=col_annot_pos))

        coldendrog = dendrogram_plot(size_df.T, **coldendkws)
        x_ticks = np.arange(size_df.shape[1]) + 0.5
        axs['clusty'].set_xticks(x_ticks)

        col_index = np.array(coldendrog['leaves'], dtype=np.int64)
        col_heat = pd.DataFrame(
                        {'cluster': coldendrog['leaves_color_list']},
                            index=colnames[col_index])

    if not col_annot is None:
        axs['heaty'].sharex(axs['scatter'])
        assert col_annot.shape[0] == size_df.shape[1]
        col_annot = pd.DataFrame(col_annot).iloc[col_index]
        col_annot.index = colnames[col_index]
        colannotkws = {**annotkws, **colannotkws, 'ax': axs['heaty'],}

        col_df = value2color(col_annot, bins=256, seed=seed,
                             color_dict=colannot_colors,
                             cmap=colannotkws['cmap'] )
        col_df.index=col_annot.index
        col_df.columns=col_annot.columns

        if col_annot_pos == 'top':
            col_df = pd.concat([col_heat, col_df], axis=1)
        else:
            col_df = pd.concat([col_df, col_heat], axis=1)
        col_df, colors = color2value(col_df)

        colannotkws['cmap'] = mpl.colors.ListedColormap(colors)
        heat_plot(col_df.T, **colannotkws)

    ## dot annot
    dotkws.update(dict(show_yticks=show_yticks,
                        show_xticks=show_xticks,
                        max_size=max_size,
                        cmap=dot_cmap,
                        grid_color=grid_color,
                        grid_linestyle=grid_linestyle,
                        grid_linewidth=grid_linewidth,
                ))


    size_df = size_df.iloc[row_index, col_index]
    color_df = color_df.iloc[row_index, col_index]
    dot_plot(size_df, color_df=color_df, 
            #  fig=fig,
             ax=axs['scatter'], **dotkws)

    # axs['clusty'].set_xticks(axs['scatter'].get_xticks())
    # for ikey in ['clusty',  'heaty', 'scatter']:
    #     x_ticks = np.arange(size_df.shape[1])
    #     if ikey in axs.keys(): 
    #         axs[ikey].set_xticks(x_ticks)

    try:
        fig.tight_layout()
    except:
        pass

    if save:
        fig.savefig(save, bbox_inches='tight', 
                    dpi=dpi, transparent=transparent)
    if show:
        plt.show()
    else:
        return fig, axs