import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec
from scipy.spatial import distance
import scipy.cluster.hierarchy as sch
import numpy as np
import pandas as pd
from ._colors import random_colors, cmaptolist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory
from scipy import sparse
from matplotlib import rc_context

plt.rcdefaults()

def auto_pointsize(
    p_num=None, ps=None,
    fig=None,
    ax=None,
    target_coverage=0.20,
    min_size=1e-4,
    max_size=20.0,
):
    if ps is not None:
        return float(ps)
    if p_num is None:
        return 1.0

    n = max(int(p_num), 1)
    if fig is None and ax is not None:
        fig = ax.figure
    if ax is not None and fig is not None:
        inv = fig.dpi_scale_trans.inverted()
        bbox_inches = ax.get_window_extent().transformed(inv)
        area_px = bbox_inches.width * bbox_inches.height * (fig.dpi ** 2)
        dpi = fig.dpi
    elif fig is not None:
        w_inch, h_inch = fig.get_size_inches()
        area_px = w_inch * h_inch * (fig.dpi ** 2)
        dpi = fig.dpi
    else:
        w_inch, h_inch = mpl.rcParams["figure.figsize"]
        dpi = mpl.rcParams["figure.dpi"]
        area_px = w_inch * h_inch * (dpi ** 2)

    area_per_point_px = (target_coverage * area_px / n) ** 0.90
    s = area_per_point_px * (72.0 / dpi) ** 2
    return float(np.clip(s, min_size, max_size))


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
    
    

def scdotplot0(
    adata,
    marker_genes,
    groupby,
    *,
    layer=None,
    use_raw=False,
    categories=None,
    expression_cutoff=0,
    standard_scale="var",
    cmap="Reds",
    dot_size=(6, 90),
    box_color="#FFF3CD",
    box_edge="#444444",
    box_alpha=0.45,
    figsize=None,
    show=True,
):
    """Fast standalone dotplot; return ``fig, ax, data``."""
    if use_raw:
        if adata.raw is None:
            raise ValueError("adata.raw is None")
        var_names = adata.raw.var_names
    else:
        var_names = adata.var_names

    genes, gene_groups, positions = [], [], []
    for label, values in marker_genes.items():
        values = [values] if isinstance(values, str) else list(values)
        start = len(genes)
        genes.extend(values)
        gene_groups.extend([label] * len(values))
        positions.append((label, start, len(genes)))

    unique_genes = list(dict.fromkeys(genes))
    missing = [g for g in unique_genes if g not in var_names]
    if missing:
        raise ValueError(f"Genes not found in adata: {missing[:10]}")

    if use_raw:
        X = adata.raw[:, unique_genes].X
    elif layer is None:
        X = adata[:, unique_genes].X
    else:
        X = adata[:, unique_genes].layers[layer]

    groups = adata.obs[groupby]
    if categories is None:
        categories = (
            list(groups.cat.categories)
            if isinstance(groups.dtype, pd.CategoricalDtype)
            else list(pd.unique(groups.dropna()))
        )
    categories = list(categories)
    codes = pd.Categorical(groups, categories=categories).codes
    valid = codes >= 0
    counts = np.bincount(codes[valid], minlength=len(categories))
    if np.any(counts == 0):
        raise ValueError("categories contains an empty group")

    G = sparse.csr_matrix(
        (np.ones(valid.sum()), (codes[valid], np.flatnonzero(valid))),
        shape=(len(categories), adata.n_obs),
    )
    if sparse.issparse(X):
        X = X.tocsr()
        mean = (G @ X).toarray() / counts[:, None]
        fraction = (G @ (X > expression_cutoff)).toarray() / counts[:, None]
    else:
        X = np.asarray(X)
        mean = np.asarray(G @ X) / counts[:, None]
        fraction = np.asarray(G @ (X > expression_cutoff)) / counts[:, None]

    gene_index = {g: i for i, g in enumerate(unique_genes)}
    order = [gene_index[g] for g in genes]
    mean, fraction = mean[:, order], fraction[:, order]
    color = mean.copy()
    if standard_scale == "var":
        vmin, vmax = color.min(0), color.max(0)
        color = (color - vmin) / np.where(vmax > vmin, vmax - vmin, 1)
    elif standard_scale == "group":
        vmin, vmax = color.min(1)[:, None], color.max(1)[:, None]
        color = (color - vmin) / np.where(vmax > vmin, vmax - vmin, 1)
    elif standard_scale is not None:
        raise ValueError("standard_scale must be 'var', 'group', or None")

    n_groups, n_genes = mean.shape
    figsize = figsize or (max(8, n_genes * 0.24), max(3, n_groups * 0.35))
    fig, ax = plt.subplots(figsize=figsize)

    row_index = {label: i for i, label in enumerate(categories)}
    for label, start, end in positions:
        if label in row_index:
            ax.add_patch(
                Rectangle(
                    (start, row_index[label]),
                    end - start,
                    1,
                    facecolor=box_color,
                    edgecolor=box_edge,
                    linewidth=0.8,
                    alpha=box_alpha,
                    zorder=0,
                )
            )

    y, x = np.indices(mean.shape)
    sizes = dot_size[0] + np.clip(fraction, 0, 1) ** 1.5 * (
        dot_size[1] - dot_size[0]
    )
    dots = ax.scatter(
        x.ravel() + 0.5,
        y.ravel() + 0.5,
        s=sizes.ravel(),
        c=color.ravel(),
        cmap=cmap,
        edgecolors="#888888",
        linewidths=0.25,
        zorder=2,
    )

    ax.set(
        xlim=(0, n_genes),
        ylim=(n_groups, 0),
        xticks=np.arange(n_genes) + 0.5,
        yticks=np.arange(n_groups) + 0.5,
        xticklabels=genes,
        yticklabels=categories,
    )
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(False)

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for label, start, end in positions:
        ax.plot([start, end], [1.015, 1.015], color=box_edge, lw=0.7,
                transform=trans, clip_on=False)
        ax.text((start + end) / 2, 1.025, str(label), rotation=90,
                ha="center", va="bottom", fontsize=7,
                transform=trans, clip_on=False)

    fig.colorbar(dots, ax=ax, fraction=0.012, pad=0.01,
                 label="Mean expression" if standard_scale is None else "Scaled mean expression")
    handles = [
        ax.scatter([], [], s=dot_size[0] + p**1.5 * (dot_size[1] - dot_size[0]),
                   facecolor="white", edgecolor="#777777", label=f"{p:.0%}")
        for p in (0.25, 0.50, 0.75, 1.00)
    ]
    ax.legend(handles=handles, title="Fraction", frameon=False,
              bbox_to_anchor=(1.12, 0.5), loc="center left", fontsize=7)
    fig.tight_layout()

    data = pd.DataFrame(
        {
            "group": np.repeat(categories, n_genes),
            "gene": np.tile(genes, n_groups),
            "marker_group": np.tile(gene_groups, n_groups),
            "mean_expression": mean.ravel(),
            "fraction": fraction.ravel(),
            "plot_color": color.ravel(),
        }
    )
    if show:
        plt.show()
    return fig, ax, data

# fig, ax, dotplot_df = dotplot(DEG1.adata, marker_genes1,  groupby=clumethod,
#                   expression_cutoff = 0.001,

#                    standard_scale='var',
#                   use_raw=True,
#                   cmap=cc.pl.cmap1,
#                               dot_size=(1,250),
#                    # figsize=figsize
# )


def scdotplot(
    adata=None,
    marker_genes=None,
    groupby=None,
    *,
    data=None,
    layer=None,
    use_raw=False,
    categories=None,
    expression_cutoff=0,
    standard_scale="var",
    cmap="Reds",
    dot_size=(6, 90),
    box_color="#FFF3CD",
    box_edge="#444444",
    box_alpha=0.45,
    box_lw =1.0,
    swap_axes=False,
    gene_width=0.30,
    gene_fontsize=7,
    group_fontsize=8,
    marker_fontsize=7,
    edgecolors="#888888",
    edgelw = 0.25,
    marker='o',
    legend_fontsize=7,
    legend_title_fontsize=8,
    colorbar_fontsize=8,
    gene_pad=3,
    bbox_to_anchor = None,
    pad =None,
    lheight=0.03,
    marker_box_bottom=1.01,
    marker_box_lw=0.7,
    marker_gap=0.12,
    legend_width=2.2,
    legend_wspace=0.55,
    legend_height=2.4,
    figsize=None,
    show=True,
    save=None,
    dpi=2000, transparent=True,
):
    """Dotplot from AnnData or from this function's returned DataFrame.

    AnnData mode
    ------------
    ``dotplot(adata, marker_genes, groupby='cluster')``

    DataFrame mode
    --------------
    ``dotplot(data=dotplot_df, swap_axes=True)``

    Returns
    -------
    fig, ax, data
    """
    if data is not None:
        if adata is not None:
            raise ValueError("Pass either adata or data, not both")
        adata = data
    if adata is None:
        raise ValueError("adata or data must be provided")

    if isinstance(adata, pd.DataFrame):
        df = adata
        required = {
            "group",
            "gene",
            "marker_group",
            "mean_expression",
            "fraction",
        }
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing columns: {sorted(missing)}")

        if categories is None:
            categories = list(pd.unique(df["group"]))
        else:
            observed = set(df["group"].dropna())
            categories = [group for group in categories if group in observed]
            if not categories:
                raise ValueError("No categories are present in DataFrame group")

        blocks = []
        for group in categories:
            block = df.loc[df["group"] == group]
            blocks.append(block)

        genes = blocks[0]["gene"].tolist()
        gene_groups = blocks[0]["marker_group"].tolist()
        for group, block in zip(categories[1:], blocks[1:]):
            if (
                block["gene"].tolist() != genes
                or block["marker_group"].tolist() != gene_groups
            ):
                raise ValueError(
                    f"Gene order for group {group!r} differs from the first group"
                )

        mean = np.vstack(
            [block["mean_expression"].to_numpy(float) for block in blocks]
        )
        fraction = np.vstack(
            [block["fraction"].to_numpy(float) for block in blocks]
        )

        positions = []
        start = 0
        while start < len(genes):
            label = gene_groups[start]
            end = start + 1
            while end < len(genes) and gene_groups[end] == label:
                end += 1
            positions.append((label, start, end))
            start = end

    else:
        if marker_genes is None or groupby is None:
            raise ValueError("AnnData input requires marker_genes and groupby")

        if use_raw:
            if adata.raw is None:
                raise ValueError("adata.raw is None")
            var_names = adata.raw.var_names
        else:
            var_names = adata.var_names

        genes, gene_groups, positions = [], [], []
        for label, values in marker_genes.items():
            values = [values] if isinstance(values, str) else list(values)
            start = len(genes)
            genes.extend(values)
            gene_groups.extend([label] * len(values))
            positions.append((label, start, len(genes)))

        unique_genes = list(dict.fromkeys(genes))
        missing = [g for g in unique_genes if g not in var_names]
        if missing:
            raise ValueError(f"Genes not found in adata: {missing[:10]}")

        if use_raw:
            X = adata.raw[:, unique_genes].X
        elif layer is None:
            X = adata[:, unique_genes].X
        else:
            X = adata[:, unique_genes].layers[layer]

        groups = adata.obs[groupby]
        if categories is None:
            categories = (
                list(groups.cat.categories)
                if isinstance(groups.dtype, pd.CategoricalDtype)
                else list(pd.unique(groups.dropna()))
            )
        categories = list(categories)
        codes = pd.Categorical(groups, categories=categories).codes
        valid = codes >= 0
        counts = np.bincount(codes[valid], minlength=len(categories))
        if np.any(counts == 0):
            raise ValueError("categories contains an empty group")

        G = sparse.csr_matrix(
            (np.ones(valid.sum()), (codes[valid], np.flatnonzero(valid))),
            shape=(len(categories), adata.n_obs),
        )
        if sparse.issparse(X):
            X = X.tocsr()
            mean = (G @ X).toarray() / counts[:, None]
            fraction = (
                (G @ (X > expression_cutoff)).toarray() / counts[:, None]
            )
        else:
            X = np.asarray(X)
            mean = np.asarray(G @ X) / counts[:, None]
            fraction = np.asarray(G @ (X > expression_cutoff)) / counts[:, None]

        gene_index = {gene: i for i, gene in enumerate(unique_genes)}
        order = [gene_index[gene] for gene in genes]
        mean, fraction = mean[:, order], fraction[:, order]

    color = mean.copy()
    if standard_scale == "var":
        vmin, vmax = color.min(0), color.max(0)
        color = (color - vmin) / np.where(vmax > vmin, vmax - vmin, 1)
    elif standard_scale == "group":
        vmin, vmax = color.min(1)[:, None], color.max(1)[:, None]
        color = (color - vmin) / np.where(vmax > vmin, vmax - vmin, 1)
    elif standard_scale is not None:
        raise ValueError("standard_scale must be 'var', 'group', or None")

    n_groups, n_genes = mean.shape
    if figsize is None:
        plot_size = (
            (max(5, n_groups * 0.55), max(8, n_genes * gene_width))
            if swap_axes
            else (max(8, n_genes * gene_width), max(3, n_groups * 0.35))
        )
    else:
        plot_size = figsize

    fig = plt.figure(
        figsize=(plot_size[0] + legend_width, plot_size[1]),
    )
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=(plot_size[0], legend_width),
        wspace=(0.28 if swap_axes else 0.04) if pad is None else pad,
    )
    ax = fig.add_subplot(gs[0])
    legend_gs = gs[1].subgridspec(
        1,
        2,
        width_ratios=(1.25, 1),
        wspace=legend_wspace,
    )
    size_host = fig.add_subplot(legend_gs[0])
    cbar_host = fig.add_subplot(legend_gs[1])
    size_host.set_axis_off()
    cbar_host.set_axis_off()

    # Keep both legends compact when swap_axes=True makes the figure very tall.
    legend_height_ratio = min(0.90, legend_height / plot_size[1])
    legend_bottom = (1 - legend_height_ratio) / 2
    size_ax = size_host.inset_axes(
        [0, legend_bottom, 1, legend_height_ratio]
    )

    row_index = {label: i for i, label in enumerate(categories)}
    for label, start, end in positions:
        if label not in row_index:
            continue
        if swap_axes:
            xy, width, height = (row_index[label], start), 1, end - start
        else:
            xy, width, height = (start, row_index[label]), end - start, 1
        ax.add_patch(
            Rectangle(
                xy,
                width,
                height,
                facecolor=box_color,
                edgecolor=box_edge,
                linewidth= box_lw ,
                alpha=box_alpha,
                zorder=0,
            )
        )

    plot_fraction = fraction.T if swap_axes else fraction
    plot_color = color.T if swap_axes else color
    y, x = np.indices(plot_color.shape)
    sizes = dot_size[0] + np.clip(plot_fraction, 0, 1) ** 1.5 * (
        dot_size[1] - dot_size[0]
    )
    dots = ax.scatter(
        x.ravel() + 0.5,
        y.ravel() + 0.5,
        s=sizes.ravel(),
        c=plot_color.ravel(),
        cmap=cmap,
        edgecolors=edgecolors,
        marker=marker,
        linewidths=edgelw,
        zorder=2,
    )

    if swap_axes:
        ax.set(
            xlim=(0, n_groups),
            ylim=(n_genes, 0),
            xticks=np.arange(n_groups) + 0.5,
            yticks=np.arange(n_genes) + 0.5,
            xticklabels=categories,
            yticklabels=genes,
        )
        ax.tick_params(axis="x", labelrotation=90, labelsize=group_fontsize)
        ax.tick_params(axis="y", labelsize=gene_fontsize)

        # Marker-gene blocks are vertical after swapping axes. Draw a real
        # right-side bracket with two horizontal caps for every block.
        trans = blended_transform_factory(ax.transAxes, ax.transData)
        for label, start, end in positions:
            gap = min(marker_gap, max((end - start) / 2 - 1e-6, 0))
            marker_start = start + gap
            marker_end = end - gap
            marker_outer = marker_box_bottom + lheight
            ax.plot(
                [marker_box_bottom, marker_outer,
                 marker_outer, marker_box_bottom],
                [marker_start, marker_start, marker_end, marker_end],
                color=box_edge,
                lw=marker_box_lw,
                transform=trans,
                clip_on=False,
            )
            ax.text(
                marker_outer + 0.015,
                (start + end) / 2,
                str(label),
                ha="left",
                va="center",
                fontsize=marker_fontsize,
                transform=trans,
                clip_on=False,
            )
    else:
        ax.set(
            xlim=(0, n_genes),
            ylim=(n_groups, 0),
            xticks=np.arange(n_genes) + 0.5,
            yticks=np.arange(n_groups) + 0.5,
            xticklabels=genes,
            yticklabels=categories,
        )
        ax.tick_params(axis="x", pad=gene_pad)
        ax.tick_params(axis="y", labelsize=group_fontsize)
        plt.setp(
            ax.get_xticklabels(),
            rotation=90,
            ha="right",
            va="center",
            rotation_mode="anchor",
            fontsize=gene_fontsize,
        )

        trans = blended_transform_factory(ax.transData, ax.transAxes)
        for label, start, end in positions:
            gap = min(marker_gap, max((end - start) / 2 - 1e-6, 0))
            marker_start = start + gap
            marker_end = end - gap
            marker_top = marker_box_bottom + lheight
            ax.plot(
                [marker_start, marker_start, marker_end, marker_end],
                [marker_box_bottom, marker_top,
                 marker_top, marker_box_bottom],
                color=box_edge,
                lw=marker_box_lw,
                transform=trans,
                clip_on=False,
            )
            ax.text(
                (start + end) / 2,
                marker_box_bottom + lheight / 2,
                str(label),
                rotation=0,
                ha="center",
                va="top",
                fontsize=marker_fontsize,
                transform=trans,
                clip_on=False,
            )

    ax.grid(False)
    cax = cbar_host.inset_axes(
        [0.18, legend_bottom, 0.30, legend_height_ratio]
    )
    cbar = fig.colorbar(dots, cax=cax)
    cbar.set_label(
        "Mean expression"
        if standard_scale is None
        else "Scaled mean expression",
        fontsize=colorbar_fontsize,
    )
    cbar.ax.tick_params(labelsize=colorbar_fontsize)
    fraction_levels = np.array([0.25, 0.50, 0.75, 1.00])
    legend_sizes = dot_size[0] + fraction_levels**1.5 * (
        dot_size[1] - dot_size[0]
    )
    legend_y = np.arange(len(fraction_levels))
    size_ax.scatter(
        np.zeros_like(legend_y),
        legend_y,
        s=legend_sizes,
        facecolor="white",
        edgecolor="#777777",
        marker=marker,
        linewidth=0.8,
    )
    size_ax.set(
        xlim=(-0.8, 0.8),
        ylim=(len(fraction_levels) - 0.25, -0.75),
        xticks=[],
        yticks=legend_y,
        yticklabels=[f"{p:.0%}" for p in fraction_levels],
    )
    size_ax.yaxis.tick_right()
    size_ax.tick_params(
        axis="y",
        left=False,
        right=False,
        labelleft=False,
        labelright=True,
        pad=3,
        labelsize=legend_fontsize,
    )
    size_ax.set_title(
        "Fraction",
        fontsize=legend_title_fontsize,
        pad=6,
    )
    for spine in size_ax.spines.values():
        spine.set_visible(False)

    data = pd.DataFrame(
        {
            "group": np.repeat(categories, n_genes),
            "gene": np.tile(genes, n_groups),
            "marker_group": np.tile(gene_groups, n_groups),
            "mean_expression": mean.ravel(),
            "fraction": fraction.ravel(),
            "plot_color": color.ravel(),
        }
    )
    if save:
        with rc_context({'pdf.fonttype': 42, 'ps.fonttype': 42}):
            fig.savefig(save, bbox_inches='tight', 
                        dpi=dpi, transparent=transparent)
    if show:
        plt.show()
    return fig, ax, data