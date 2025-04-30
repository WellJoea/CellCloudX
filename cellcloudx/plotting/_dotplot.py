import pandas as pd
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ._colors import adata_color
from ._utilis import colrows
from ._colors import color_palette, adata_color, pxsetcolor, cmap1, cmaptolist
from ..utilis._arrays import list_iter, vartype

def scattersplit(adata, groupby, splitby=None, basis='X_umap',
                groups = None, splits=None,
                legend_text=False,
                legend_ax = 'merge',
                dpi=200,
                use_raw = False,
                # lloc="best", # lloc="center left",
                rasterized=False,
                show_background=False,
                bg_size=None, 
                show_images = False,
                img_basis = None,
                img_key='hires',
                origin = 'upper', 
                swap_xy = False,
                alpha_img=1, 
                alpha_bg=1,
                alpha=1,
                ncols=None, 
                nrows = None, fsize=5, werror=0, herror=0, 
                size=None, vmin = None, vmax = None, cmin = None,
                drop_vmin=False,
                size_scale=False, share_cbar_value=False,
                text_fontweight = "bold",
                text_fontsize = None,
                text_fontoutline = None,
                text_adjust = True,
                adjust_args = {},
                legend_pad = 0.025,
                legend_width = 0.02,
                legend_pos = 'right',
                legend_format = None,
                legend_nbins = 6,
                legend_grid_color = 'none',
                legend_fontsize=None,
                legend_shrink = 1,
                legend_loc = 'center left',
                legend_orientation = None,
                marker = 'o',
                markerscale=6,
                edgecolors='None', 
                bbox_to_anchor=(1, 0, 0.7, 1),
                lncol=None, lnrow = None, lncols = None, mode=None,
                set_aspect=1,
                grid=False,
                axis_off=False,
                ticks_off=False,
                invert_xaxis=False,
                invert_yaxis=False,
                axis_label=None,
                xlim = None, ylim = None,
                palette=None,
                cmap =None,
                sharex=True, sharey=True,
                layout = None,
                fontsize=None, 
                labelsize=None,bg_color='lightgray', 
                show=True, save=None,  
                left=None, bottom=None,
                right=None, top=None, 
                wspace=None, hspace=None,
                iargs = {},
                **kargs):
    adataw = adata.copy()
    if not groups is None:
        if isinstance(groups, (str, int, bool)):
            groups = [groups]
        adata = adata[adata.obs[groupby].isin(groups)].copy()
    if not splits is None:
        if isinstance(splits, (str, int, bool)):
            splits = [splits]
        adata = adata[adata.obs[splitby].isin(splits)].copy()
    
    adata = adata.raw if use_raw else adata
    try:
        G = adata.obs[groupby].cat.remove_unused_categories().cat.categories
    except:
        G = adata.obs[groupby].unique()

    locs = adata.obsm[basis][:,:2]
    if swap_xy:
        locs = locs[:,::-1]

    bg_size = bg_size or size
    lncol= 1 if (lncol is None) and (lnrow is None) else lncol
    tight = True

    _idxd = np.ones(adata.shape[0], dtype=bool)
    if splitby:
        if splitby in adata.obs.columns:
            splitdt = adata.obs[splitby]
        elif splitby in adata.var_names:
            try:
                splitdt = adata[:,splitby].X.toarray().flatten()
            except:
                splitdt = adata[:,splitby].X.flatten()
        else:
            raise ValueError(f'{splitby} isnot in adata.obs.columns and adata.var_names')

        dtype = vartype(splitdt)
        if dtype == 'discrete':
            adata.obs[splitby] = adata.obs[splitby].astype('category')
            _sccor = adata_color(adata, value=splitby, palette=palette, return_color=True)
            try:
                S = adata.obs[splitby].cat.remove_unused_categories().cat.categories
            except:
                S = adata.obs[splitby].unique()
        else: 
            splitdt = np.array(splitdt)

            if not vmin is None:
                if drop_vmin:
                    _idxd = splitdt>=vmin
                splitdt[splitdt<vmin] = cmin or 0
            splitdt = np.clip(splitdt, cmin or splitdt.min(), vmax or splitdt.max())
            _sccor = cmap1 if cmap is None else cmap
    else:
        S = G
        dtype = 'discrete'
        adata.obs[groupby] = adata.obs[groupby].astype('category')
        _sccor = adata_color(adata, value=groupby, palette=palette, return_color=True)

    ncells= len(G)
    nrows, ncols = colrows(ncells, nrows=nrows, ncols=ncols, soft=False)
    size = size or max(120000 * ncells / adata.shape[0], 0.01)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*(fsize+werror), nrows*(fsize+herror)), 
                             dpi=dpi,
                             sharex=sharex, sharey=sharey, layout=layout)
    fig.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    for n,sid in enumerate(G):
        if ncells ==1:
            AX = axes
        elif min(nrows, ncols)==1:
            AX = axes[n]
        else:
            AX = axes[n//ncols, n%ncols]

        # set axis
        if n == 0:
            if invert_xaxis:
                AX.invert_xaxis()
            if invert_yaxis:
                AX.invert_yaxis()

        if not axis_label is False:
            basis = basis if axis_label is None else axis_label
            AX.set_ylabel((basis+'1').upper(), fontsize = fontsize)
            AX.set_xlabel((basis+'2').upper(), fontsize = fontsize)

        AX.set_title(sid,size=fontsize)
        # AX.set_yticks([])
        AX.tick_params(axis='both', which='major', labelsize=labelsize)
        # ax.tick_params(axis='both', which='minor', labelsize=8)
        AX.set_aspect(set_aspect)
        if axis_off:
            AX.set_axis_off()
        if ticks_off:
            AX.get_xaxis().set_visible(False)
            AX.get_yaxis().set_visible(False)
        AX.grid(grid)

        if not xlim is None:
            AX.set_xlim(*xlim)
        if not ylim is None:
            AX.set_ylim(*ylim)
        
        # plot scatter
        ## plot images
        if show_images:
            img_basis = img_basis or basis
            try:
                image = adataw.uns[img_basis][sid]['images'][img_key]
            except:
                image = None
                print(f"cannot find image in adata.uns['{img_basis}']['{sid}']['images']['{img_key}']")
            try:
                scalef = adataw.uns[img_basis][sid]['scalefactors'][f'tissue_{img_key}_scalef']
            except:
                scalef = 1
                # print("cannot find scalefactor in adata.uns[{img_key}][{sid}]['scalefactors']['tissue_{img_basis}_scalef']")
            if not image is None:
                img = AX.imshow(image, origin=origin, alpha=alpha_img, **iargs)
        ## plot background
        if show_background != False:
            if show_background == 'groups':
                _idxr = adataw.obs[groupby]==sid
                locsb = adataw[_idxr,:].obsm[basis]
            else:
                _idxr = list(set(adataw.obs_names.values) - set(adata.obs_names.values[_idx]))
                locsb = adataw[_idxr,:].obsm[basis]
            AX.scatter( locsb[:, 0], locsb[:, 1], s=bg_size, marker=marker, 
                    #    facecolors=bg_color, edgecolors=bg_color, 
                    alpha=alpha_bg,
                    edgecolors=edgecolors, 
                    c=bg_color)
        ## plot sactters
        _idx = (adata.obs[groupby]==sid) & (_idxd)
        if np.sum(_idx)==0:
            continue

        
        if splitby is None:
            AX.scatter( locs[:, 0][_idx], locs[:, 1][_idx], s=size, marker=marker, label=sid, 
                    #    facecolors=_sccor[n], edgecolors=_sccor[n], 
                    edgecolors=edgecolors, 
                    alpha=alpha,
                    rasterized=rasterized,
                    c=_sccor[n], **kargs)
            x_pos, y_pos = pos_center(locs[_idx])
            texts, x_pos, y_pos = [sid], [x_pos], [y_pos]
        else:
            if dtype == 'discrete':
                texts = S
                x_pos, y_pos = [], []
                for icor, isid in zip(_sccor, S):
                    _iidx = ((adata.obs[groupby]==sid) & (adata.obs[splitby]==isid))
                    if np.sum(_iidx)>0:
                        AX.scatter(locs[_iidx][:, 0], locs[_iidx][:, 1], s=size, 
                                    marker=marker, c=icor, 
                                    # facecolors=icor, edgecolors=icor,
                                    alpha=alpha,
                                    edgecolors=edgecolors, 
                                      label=isid,  **kargs)
                        ixy_pos = pos_center(locs[_iidx])
                        x_pos.append(ixy_pos[0])
                        y_pos.append(ixy_pos[1])
            else:
                idata = splitdt[_idx]
                if idata.shape[0]>0:
                    imin, imax = (splitdt.min(), splitdt.max()) \
                                        if (share_cbar_value or legend_ax=='merge') \
                                        else (idata.min(), idata.max())
                    markers = (idata * size) if size_scale else size
                    im = AX.scatter(locs[_idx][:, 0], locs[_idx][:, 1],
                                    s=markers, 
                                    vmin=imin, vmax=imax, 
                                    # facecolors=idata, edgecolors=idata,
                                    rasterized=rasterized,
                                    alpha=alpha,
                                    edgecolors=edgecolors, 
                                    marker=marker, c=idata, label=sid, cmap=_sccor, **kargs)
                    x_pos, y_pos = pos_center(locs[_idx])
                    texts, x_pos, y_pos = [sid], [x_pos], [y_pos]

        # add text:
        if legend_text:
            axtexts = []
            for itx, ixpos, iypos in zip(texts, x_pos, y_pos):
                tx = AX.text(
                        ixpos,
                        iypos,
                        itx,
                        weight=text_fontweight,
                        verticalalignment="center",
                        horizontalalignment="center",
                        fontsize=text_fontsize,
                        path_effects=text_fontoutline,
                    )
                axtexts.append(tx)
            if text_adjust:
                try:
                    from adjustText import adjust_text
                except:
                    raise ValueError('adjust_text is not installed. `pip install adjustText`')
                adjust_text(axtexts, ax=AX, **adjust_args)
        # add legend:
        if (not legend_ax is None) and \
            ((type(legend_ax) == int and n==legend_ax) or 
                (type(legend_ax) == list and n in legend_ax) or
                (type(legend_ax) == str and legend_ax=='all')):

            if (not splitby is None) and (dtype == 'continuous'):
                divider = make_axes_locatable(AX) 
                # legend_format = matplotlib.ticker.FuncFormatter(lambda x, pos: '')  if legend_format is None else legend_format
                cax1 = divider.append_axes(legend_pos, size=f"{legend_width*100}%", pad=legend_pad)
                cbar = fig.colorbar(im, cax = cax1, shrink=legend_shrink, cmap=_sccor,
                                    orientation =legend_orientation, 
                                    format = legend_format)
                cbar.ax.tick_params(grid_color=legend_grid_color, labelsize=legend_fontsize)
                cbar.ax.locator_params(tight = None, nbins=legend_nbins)
            else:
                ihandle, ilabel = AX.get_legend_handles_labels()
                lrow, lcol = colrows(len(ihandle), nrows=lnrow, ncols=lncol, soft=False)
                lcol = lcol if lncols is None else lncols[n]
                AX.legend(loc=legend_loc, bbox_to_anchor=bbox_to_anchor, fontsize = legend_fontsize,
                            mode=mode, ncol = lcol, markerscale=markerscale)

    # add merge legend
    if (not legend_ax is None) and (type(legend_ax) == str and legend_ax=='merge'):
        if (not splitby is None) and (dtype == 'continuous'):
            tight = False
            cbar_ax = axes_cax(axes, pad=legend_pad, w = legend_width, h =legend_shrink, orient=legend_orientation)
            cbar_ax = fig.add_axes(cbar_ax)
            # from matplotlib.colors import Normalize
            # import matplotlib.cm as cm
            # normalizer = Normalize(0, 4)
            # im = cm.ScalarMappable(norm=normalizer)
            # fig.colorbar(im, ax=axes.ravel().tolist())
            
            cbar = fig.colorbar( im, 
                                cax = cbar_ax, 
                                ax=axes.flatten()[-1],
                                orientation =legend_orientation, 
                                shrink=1, cmap=_sccor,
                                format = legend_format,
                                location=legend_pos
                               )
            cbar.ax.tick_params(grid_color=legend_grid_color, labelsize=legend_fontsize)
            cbar.ax.locator_params(tight = None, nbins=legend_nbins)
        else:
            handles, labels = [], []
            axes_list = axes.flatten() if len(G) >1 else [axes]
            for iax in axes_list:
                ihandle, ilabel = iax.get_legend_handles_labels()
                handles.extend(ihandle)
                labels.extend(ilabel)
            handles = [dict(zip(labels, handles))[ilb] for ilb in S]
            lrow, lcol = colrows(len(handles), nrows=lnrow, ncols=lncol, soft=False)
            fig.legend(handles, S, loc=legend_loc, bbox_to_anchor=bbox_to_anchor, 
                        fontsize = legend_fontsize,
                        mode=mode, ncol = lcol, markerscale=markerscale)

    if nrows*ncols - len(G) >0:
        for j in range(nrows*ncols - len(G)):
            fig.delaxes(axes[-1][ -j-1])

    if tight:
        fig.tight_layout()
    if save:
        if tight:
            fig.savefig(save, bbox_inches='tight')
        else:
            fig.savefig(save)
    if show is None:
        return fig, axes
    elif show is True:
        plt.show()
    else:
        plt.close()

def scatterstack(adata, groupby, splitby, basis='spatial', 
                   use_raw= False,
                   ax=None, #fig=None,
                   size=None, 
                   figsize=(8,8), alphas=None, cmaps=None,
                                   set_aspect=1,
                    rasterized=False,
                    grid=False,
                    axis_off=False,
                    ticks_off=False,
                    invert_xaxis=False,
                    invert_yaxis=False,
                    axis_label=None,
                    edgecolors='None',
                    xlim = None, ylim = None,
                    fontsize=None, 
                    title=None, 
                    labelsize=None,
                    show_legend=True,
                    value_trim=None,
                    legend_loc='best', anchor=None,
                    legend_pos = 'right',
                    legend_shrink = 1,
                    legend_width=0.03, legend_pad = 0.03, pad_shift=0.2,
                    legend_nbins = None,
                    legend_grid_color = 'none',
                    legend_fontsize=None,
                    legend_format = None,
                    largs={},
                    save=None, show=True,
                    transparent =True, dpi=None, saveargs={},
                    **kargs):

    adata = adata.raw if use_raw else adata
    try:
        G = adata.obs[groupby].cat.remove_unused_categories().cat.categories
    except:
        G = adata.obs[groupby].unique()

    n_split= len(G)
    cmaps = list_iter(cmaps)
    alphas = list_iter(alphas)
    size = list_iter(size)

    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=figsize)
    else:
        fig = ax.figure

    divider = make_axes_locatable(ax)
    for i, g in enumerate(G):
        idata = adata[adata.obs[groupby] == g]
        iloc = idata.obsm[basis]
        try:
            ival = idata[:,splitby].X.toarray().flatten()
        except:
            ival = idata[:,splitby].X.flatten()
        if not value_trim is None:
            if not value_trim[0] is None:
                ival[ival < value_trim[0]] = np.nan

        isc = ax.scatter(iloc[:,0], iloc[:,1], c=ival, s=size[i], rasterized=rasterized,
                         cmap=cmaps[i], alpha=alphas[i], edgecolors=edgecolors,
                         **kargs)
        if show_legend:
            if anchor is None:
                icax = divider.append_axes(legend_pos, size=f"{legend_width*100}%", 
                        pad=legend_pad+i*pad_shift)
                icbar = fig.colorbar(isc, cax = icax, 
                                    shrink=legend_shrink, 
                                    ax=ax,
                                    format = legend_format, **largs)
            else:        
                # ax.legend(loc=legend_loc, bbox_to_anchor=bbox_to_anchor, fontsize = legend_fontsize,
                #             #mode=mode
                #             )
                icbar = fig.colorbar(isc, 
                                     location=legend_pos,
                                     anchor=anchor,
                                     pad=legend_pad,
                                    shrink=legend_shrink, 
                                    ax=ax,
                                    format = legend_format, **largs)


            icbar.ax.tick_params(grid_color=legend_grid_color, labelsize=legend_fontsize)

            if legend_nbins:
                icbar.ax.locator_params(tight = None, nbins=legend_nbins)
            # width=ax.get_position().width
            # height=ax.get_position().height

    ax.set_title(title,size=fontsize)
    # AX.set_yticks([])
    ax.tick_params(axis='both', which='major', labelsize=labelsize)
    # ax.tick_params(axis='both', which='minor', labelsize=8)
    ax.set_aspect(set_aspect)
    if axis_off:
        ax.set_axis_off()
    if ticks_off:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    ax.grid(grid)

    if invert_xaxis:
        ax.invert_xaxis()
    if invert_yaxis:
        ax.invert_yaxis()

    if not xlim is None:
        ax.set_xlim(*xlim)
    if not ylim is None:
        ax.set_ylim(*ylim)

    try:
        fig.tight_layout()
    except:
        pass

    if save:
        fig.savefig(save, bbox_inches='tight',  transparent=transparent , dpi=dpi, **saveargs)

    if show is None:
        return fig, ax
    elif show is True:
        plt.show()

def spatialscattersplit( adatalist, group,
                        group_order=None, 
                        cmap = None,
                        save=None, 
                        nrows = 4,
                        ncols = 4,
                        fsize = 7,

                        lloc='center left',
                        markerscale=4, 
                        lncol=1,
                        mode='expand',
                        frameon=False,
                        bbox_to_anchor=(1, 0, 0.5, 1), #(1, 0.5),
                        borderaxespad=0.5,
                        largs={},
                        titledict={},
                        legend_num = 0,
                        herror= 0,
                         werror =0, show=True, **kargs):
    ncells = len(adatalist)
    fig, axes = plt.subplots(nrows,ncols, figsize=((fsize+werror)*ncols, (fsize+herror)*nrows))

    for i in range(ncells):
        if ncells ==1:
            ax = axes
        elif min(nrows, ncols)==1:
            ax = axes[i]
        else:
            ax = axes[i//ncols,i%ncols]
        adata = adatalist[i].copy()        
        if not group_order is None:
            adata.obs[group] = pd.Categorical(adata.obs[group], categories=group_order)
        #sc.pl._tools.scatterplots._get_palette(adata, group)
            
        sc.pl.spatial(adata, 
                      color=group,
                      img_key='hires',
                      show=False,
                      ax=ax,
                      
                      **kargs)
        if i ==legend_num:
            handles, labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()
        ax.set_title(str(i), **titledict)
    fig.legend(handles, labels,
                ncol=lncol,
                loc=lloc, 
                frameon=frameon,
                mode=mode,
                markerscale=markerscale,
                bbox_to_anchor=bbox_to_anchor,
                borderaxespad =borderaxespad,
                **largs)

    if nrows*ncols - ncells >0:
        for j in range(nrows*ncols - ncells):
            fig.delaxes(axes.flatten()[-j-1])
    fig.tight_layout()
    if save:
        fig.savefig(save)
    if show is True:
        plt.show()
    elif show is False:
        plt.close()
    else:
        return fig, axes

def pos_center(pos, percent = [7, 93]):
    lower, upper = np.percentile(pos, percent, axis=0)
    kidx = np.all((pos>=lower) | (pos<=upper), axis=1)

    if np.sum(kidx) == 0:
        percent = [percent[0]-1, percent[1]+1]
        if (percent[0] >= 0) and (percent[1] <= 100):
            pos_center(pos, percent)
        else:
            return np.median(pos, axis=0)
    return np.median(pos[kidx], axis=0)

def axes_cax(axes, pad=0.03, w = 0.03, h =1, orient = 'vertical',):
    axpos  = np.float64([iax.get_position().get_points() for iax in axes.ravel()])
    #[[x0, y0], [x1, y1]]
    x0 = axpos.min(0)[0,0]
    x1 = axpos.max(0)[1,0]
    y0 = axpos.min(0)[0,1]
    y1 = axpos.max(0)[1,1]

    H = y1 - y0
    W = x1 - x0
    xm = (x1 + x0)/2
    ym = (y1 + y0)/2

    if orient is None or  orient =='vertical':
        xc = x1
        yc = ym - h*H/2
        # yc = y0
        l = xc + pad
        b = yc
        return [l, b, w, H*h]

    elif orient =='horizontal':
        xc = xm - h*W/2
        yc = y0
        l = xc
        b = yc - pad
        return [l, b, h*W, w]
