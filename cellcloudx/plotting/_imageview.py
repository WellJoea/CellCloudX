import pandas as pd
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker

from skimage import transform as skitf
# from itkwidgets import view
# from itkwidgets import view, compare, checkerboard
# from itkwidgets.widget_viewer import Viewer
# import ipywidgets
# from ipywidgets import Button, Label, HBox, VBox
import scanpy as sc

from ._utilis import colrows
from ..utilis._arrays import list_iter
from ._colors import cmap1

def qview(*arrays, layout=None, 
                dtypes = None,
                axes = None,
                fsize=5, 
                werror=0,
                herror=0,
                titles=None,
                suptitle = None,
                nrows=None,
                ncols=None, 
                show=True, 
                save=None,
                invert_xaxis=False,
                invert_yaxis=False,
                rescale = None,
                anti_aliasing=None,
                size = 1,
                edgecolor='None',
                color = 'black',
                aspect='equal',
                set_aspect=1.0,
                grid=False,
                axis_off=False,
                vmin=None, vmax=None,
                sample=None,
                ticks_off=False,
                cmaps=None, alpha=None,
                sharex=False, sharey=False,
                figkargs={},
                title_fontsize = None,
                show_legend=False,
                legend_pad = 0.03,
                legend_width = 0.04,
                legend_pos = 'right',
                legend_format = None,
                legend_grid_color = 'none',
                legend_nbins = 6,
                legend_fontsize=None,
                legend_shrink = 1,
                legend_loc = 'center left',
                legend_orientation = None,
                bbox_to_anchor=None,
                seed = 491001,
                **kargs
                ):
    ncells= len(arrays)
    nrows, ncols = colrows(ncells, nrows=nrows, ncols=ncols, soft=False)
    import matplotlib
    if ((cmaps is None) or 
            isinstance(cmaps, (str, matplotlib.colors.ListedColormap)) or
            hasattr(cmaps, 'colors')):
        cmaps = [cmaps]*ncells

    if (dtypes is None):
        dtypes = []
        for ii in arrays:
            assert ii.ndim >=2
            if not axes is None:
                assert len(axes) == 2
                dtypes.append('loc')
            else:
                if min(ii.shape) ==2 and (ii.ndim ==2):
                    dtypes.append('loc')
                elif (ii.ndim >=2) and ii.shape[1] >2:
                    dtypes.append('image')
    elif (isinstance(dtypes), str):
        dtypes = [dtypes] * ncells
    elif (isinstance(dtypes), list):
        assert len(dtypes) >= ncells
        dtypes = dtypes[:ncells]

    arrays1 = []
    for arr in arrays:
        if arr.__class__.__name__ == 'Tensor':
            arrays1.append(arr.detach().cpu().numpy())
        else:
            arrays1.append(arr)
    arrays = arrays1

    if not rescale is None:
        arrs = []
        for n in range(ncells):
            ida = arrays[n]
            itp = dtypes[n]
            if itp in ['image']:
                hw = ida.shape[:2]
                resize = [int(round(hw[0] * rescale ,0)), int(round( hw[1] *rescale ,0))]
                arrs.append(skitf.resize(ida, resize))
            elif itp in ['loc']:
                arrs.append(ida * rescale)
    else:
        arrs = arrays

    if not sample is None:
        np.random.seed(seed)
        arrs_spl = []
        for n in range(ncells):
            ida = arrs[n]
            itp = dtypes[n]
            if itp in ['loc']:
                n_len = ida.shape[0]
                idex = np.random.choice(np.arange(n_len), 
                                size=int(n_len*sample), 
                                replace=False, p=None)
                arrs_spl.append(ida[idex])
        arrs = arrs_spl

    fig, axs = plt.subplots(nrows, ncols,
                              figsize=((fsize+werror)*ncols,(fsize+herror)*nrows),
                              sharex=sharex, sharey=sharey,**figkargs)
    #fig.patch.set_facecolor('xkcd:mint green')
    fig.suptitle(suptitle, fontsize=title_fontsize)
    for i in range(ncells):
        if ncells ==1:
            ax = axs
        elif min(nrows, ncols)==1:
            ax = axs[i]
        else:
            ax = axs[i//ncols,i%ncols]
        ida = arrs[i]
        itp = dtypes[i]

        if itp in ['image']:
            if layout=='bgr':
                im = ax.imshow(ida[:,:,::-1], aspect=aspect, cmap=cmaps[i], 
                               vmin=vmin, vmax=vmax, alpha=alpha,**kargs)
            elif layout=='rbg':
                im = ax.imshow(ida[:,:,:3], aspect=aspect, cmap=cmaps[i], vmin=vmin, vmax=vmax, 
                               alpha=alpha, **kargs)
            else:
                im = ax.imshow(ida, aspect=aspect, cmap=cmaps[i], vmin=vmin, vmax=vmax, 
                               alpha=alpha, **kargs)

            ax.set_xlabel('y')
            ax.set_ylabel('x')

            # fig.colorbar(im, ax=ax)
        elif itp in ['loc']:
            if axes is None:
                axes = [0,1]
            im = ax.scatter(ida[:,axes[0]], ida[:,axes[1]], s=size, c=color, edgecolor=edgecolor,**kargs)
            ax.set_xlabel('x')
            ax.set_ylabel('y')

        if (not titles is None):
            if i < len(titles):
                ax.set_title(titles[i], fontsize=title_fontsize)
        if axis_off:
            ax.set_axis_off()
        if ticks_off:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        if not set_aspect is None:
            ax.set_aspect(set_aspect, adjustable='box')
        ax.grid(grid)
        
        # if not show_ticks:
        #     ax.tick_params(
        #     axis='x',         
        #     which='both',      
        #     bottom=False,      
        #     top=False,        
        #     labelbottom=False) 
        if show_legend:
            # legend_format = matplotlib.ticker.FuncFormatter(lambda x, pos: '')  if legend_format is None else legend_format
            # cax1 = divider.append_axes(legend_pos, size=f"{legend_width*100}%", pad=legend_pad)
            # cbar = fig.colorbar(im, cax = cax1, 
            #                     format = legend_format)
            # cbar.ax.tick_params(grid_color=legend_grid_color)

            if not bbox_to_anchor is None:
                cax1 = ax.inset_axes(bbox_to_anchor, transform=ax.transAxes)
            else:
                divider = make_axes_locatable(ax) 
                cax1 = divider.append_axes(legend_pos, size=f"{legend_width*100}%", pad=legend_pad)

            cbar = fig.colorbar(im, cax = cax1, shrink=legend_shrink, cmap=cmaps[i],
                                orientation =legend_orientation, 
                                format = legend_format)
            cbar.ax.tick_params(grid_color=legend_grid_color, labelsize=legend_fontsize)
            cbar.ax.locator_params(tight = None, nbins=legend_nbins)

        if invert_xaxis:
            ax.invert_xaxis()
        if invert_yaxis:
            ax.invert_yaxis()

    if nrows*ncols - ncells >0:
        for j in range(nrows*ncols - ncells):
            fig.delaxes(axs.flatten()[-j-1])
    fig.tight_layout()
    if save:
        plt.savefig(save)
    if show is None:
        return fig, axs
    elif show is True:
        plt.show()
    else:
        plt.close()

def drawMatches(pairlist, bgs=None,
                pairidx=None,
                matches=None,
                line_widths=None,
                line_color=None, size=5, line_width=1, color='b',
                line_alpha=None, axis_off=False, grid=False, show_line=True,
                bg_color='black', bg_size=3, titles=None, title_fontsize= None,
                aspect='auto',
                origin='upper',
                cmap=None, alpha=None, arrowstyle ='-',
                nrows = None, ncols=None, fsize = 7,
                werror =0, herror =0,
                sharex=True, sharey=True,
                figkargs={},
                linekargs={},
                line_sample = None,
                line_top = None,
                equal_aspect = False, 
                save=None, show=True, 
                invert_xaxis=False,
                invert_yaxis=False,
                seed = None,
                **kargs):
    
    np.random.seed(seed)
    ncells= len(pairlist)
    assert ncells >1, 'pairlist length muse be >1'
    if (not bgs is None) and (len(bgs)!=ncells ) :
        raise('the length of bgs and pairlist must be equal.')
    if isinstance(color, str):
        color = [color] * ncells

    line_sample = line_sample or 1

    nrows, ncols = colrows(ncells, nrows=nrows, ncols=ncols, soft=False)
    fig, axis = plt.subplots(nrows,ncols, 
                             figsize=((fsize+werror)*ncols, (fsize+herror)*nrows),
                             sharex=sharex, sharey=sharey,**figkargs)
    axes = []
    for i in range(ncells):
        if ncells ==1:
            ax = axis
        elif min(nrows, ncols)==1:
            ax = axis[i]
        else:

            ax = axis[i//ncols,i%ncols]
        axes.append(ax)

    for i in range(ncells):
        axa = axes[i]
        posa= pairlist[i]
        if (not bgs is None):
            bga = bgs[i]
            if (bga.ndim ==2) and (bga.shape[1]==2):
                axa.scatter(bga[:,0], bga[:,1], s=bg_size, c=bg_color)
            elif (bga.shape[1]>2):
                axa.imshow(bga, aspect=aspect, cmap=cmap, origin=origin, alpha=alpha )
        axa.scatter(posa[:,0], posa[:,1], s=size, c=color[i])

        if not titles is None:
            axa.set_title(titles[i], fontsize=title_fontsize)
        if axis_off:
            axa.set_axis_off()

        if invert_xaxis:
            axa.invert_xaxis()
        if invert_yaxis:
            axa.invert_yaxis()

        if equal_aspect:
            axa.set_aspect('equal', adjustable='box')
        axa.grid(grid)

    if show_line and (line_width> 0):
        if pairidx is None:
            pairidx = [ [i, i +1] for i in range(ncells-1) ]

        if isinstance(line_color, str):
            line_color = [line_color] * len(pairidx)

        for i,(r,q) in enumerate(pairidx):
            rpair = pairlist[r]
            qpair = pairlist[q]
            if line_widths is None:
                lws = np.ones(rpair.shape[0])
            else:
                lws = line_widths[i]

            rax = axes[r]
            qax = axes[q]
            if matches is None:
                ridx = qidx = range(min(len(rpair),len(qpair)))
            else:
                assert len(matches) == len(pairidx), 'the length of pairidx and matches must be equal.'
                ridx = matches[i][0]
                qidx = matches[i][1]
            if not line_top is None:
                smpidx = range(len(ridx))[: min(line_top, len(ridx))]
            elif line_sample <1:
                smpidx =  np.random.choice(len(ridx), size=int(line_sample*len(ridx)), replace=False)
            elif line_sample >1:
                smpidx =  np.random.choice(len(ridx), size=min( len(ridx), line_sample), replace=False)
            else:
                smpidx = range(len(ridx))
            for k in smpidx:
                xy1 = rpair[ridx[k]]
                xy2 = qpair[qidx[k]]
                ilw = lws[k]
    
                if (line_color is None) or (line_color[i] is None):
                    if seed is None:
                        rngc = np.random.default_rng(seed)
                    else:
                        rngc = np.random.default_rng([seed, k, i])
                    icolor = rngc.random(3)
                else: 
                    icolor = line_color[i]
                con = ConnectionPatch(xyA=xy1, xyB=xy2, coordsA="data", 
                                      coordsB="data", axesA=rax, axesB=qax, 
                                      alpha=line_alpha,
                                      color=icolor, 
                                      arrowstyle=arrowstyle, **linekargs)
                con.set_linewidth(line_width*ilw)
                fig.add_artist(con)

    if nrows*ncols - ncells >0:
        for j in range(nrows*ncols - ncells):
            fig.delaxes(axis.flatten()[-j-1])
    fig.tight_layout()
    if save:
        fig.savefig(save)
    if show is True:
        plt.show()
    elif show is False:
        plt.close()
    else:
        return fig, axis

def plt_fit(img, pos, figsize = (10,10), show=True):
    fig, ax = plt.subplots(1,1, figsize=figsize)

    ax.imshow(img,cmap = 'gray')
    ax.scatter(pos[:,0],pos[:,1], color='red', s=30)
    ax.scatter(pos[:,4],pos[:,5], color='blue',  s=30)
    ax.set_axis_off()

    if show:
        fig.show()
    else:
        return (fig, ax)

def parahist(model,  
             fsize=5, 
                werror=0,
                herror=0,
                nrows=None,
                ncols=2, 
                show=True, 
                save=None,
                bins=50,
                grid=False,
                axis_off=False,
                sharex=False, sharey=False,
                **kargs):

    ncells = len(list(model.named_parameters())) 
    nrows, ncols = colrows(ncells, ncols=ncols)
    fig, axes = plt.subplots(nrows, ncols*2,
                                figsize=((fsize+werror)*ncols*2, (fsize+herror)*nrows),
                                sharex=sharex, sharey=sharey,**kargs)
    i = 0
    for name, para in model.named_parameters():
        irow = i //ncols
        icol0 = (i % ncols) *2
        icol1 = icol0 + 1

        if min(nrows, ncols)==1:
            aw, ag = axes[icol0], axes[icol1] 
        else:
            aw, ag = axes[irow, icol0], axes[irow, icol1]
        i +=1

        try:
            W = para.data.detach().cpu().numpy().flatten()
            aw.hist(W, bins=bins, color='red', **kargs)
        except:
            pass
        try:
            G = para.grad.data.detach().cpu().numpy().flatten()
            ag.hist(G, bins=bins, color='blue', **kargs)
        except:
            pass

        aw.set_title(f'{name}_weigth')
        ag.set_title(f'{name}_grad')


        aw.grid(grid)
        ag.grid(grid)

        aw.set_xlabel(f'{name}_weigth')
        ag.set_xlabel(f'{name}_grad')

    if nrows*ncols - ncells >0:
        for j in range(nrows*ncols*2 - ncells*2):
            fig.delaxes(axes.flatten()[-j-1])
    fig.tight_layout()
    if save:
        plt.savefig(save)
    if show is None:
        return fig, axes
    elif show is True:
        fig.show()
    else:
        plt.close()

def imagemappoints(images, points, titles=None,
                   color_scales = None,
                   ncols=None, nrows=None,
                   fsize = 5, werror=0, herror=0, sharex=False, sharey=False, grid=True,
                   show=True, save=None, fkarg={}, **kargs):
    ncells = len(images) 
    nrows, ncols = colrows(ncells, nrows=nrows, ncols=ncols,)
    fig, axes = plt.subplots(nrows, ncols,
                                figsize=((fsize+werror)*ncols, (fsize+herror)*nrows),
                                sharex=sharex, sharey=sharey,**fkarg)    
    for i in range(ncells):
        if ncells ==1:
            ax = axes
        elif min(nrows, ncols)==1:
            ax = axes[i]
        else:
            ax = axes[i//ncols,i%ncols]
        title = None if  titles is None else titles[i]
        if not color_scales is None:
            color_scale = color_scales[i]
        else:
            color_scale = None
        iargs = kargs.copy()

        iargs.update(dict(ax=ax, fig=fig,color_scale=color_scale, title=title))
        for iar in ['legend_pad', 'legend_shift', 'legend_width', 'legend_height', 'size']:
            if iar in kargs:
                iargs[iar] = list_iter(kargs[iar])[i]

        if iargs.get('invert_xaxis', False):
            if i == 0:
                ax.invert_xaxis()
            iargs['invert_xaxis'] = False
        if iargs.get('invert_yaxis', False):
            if i == 0:
                ax.invert_yaxis()
            iargs['invert_yaxis'] = False

        imagemappoint(images[i], points[i], 
                      show=False, **iargs)

    if nrows*ncols - ncells >0:
        for j in range(nrows*ncols - ncells):
            fig.delaxes(axes[-1, -j-1])

    fig.tight_layout()
    if save:
        plt.savefig(save)
    if show is None:
        return fig, axes
    elif show is True:
        fig.show()
    else:
        plt.close()

def imagemappoint(image, points, figsize=(7,7), size=1, color='red',
                  rescale=None, edgecolor='None', marker='.',
                  color_scale=None,
                  center_color=False,
                  size_scale = None,
                  origin = 'upper', 
                  swap_xy = True,
                  equal_aspect=True,
                grid=False,
                axis_off=False,
                ticks_off=False,
                invert_xaxis=False,
                invert_yaxis=False,
                title = None,
                title_fontsize=None,
                legend=None,
                p_vmin=None, 
                p_vmax=None,
                set_clim = None,
                use_divider=False,
                legend_pad = 0.01,
                legend_shift = 0,
                legend_width = 0.02,
                legend_height = 0.5,
                legend_pos = 'right',
                legend_format = None,
                legend_grid_color = 'none',
                legend_size=None,
                legend_color='black',
                show=True,
                tick_nbins=5,
                alpha=1, ax=None, fig = None, iargs = {}, **kargs):
    #import matplotlib
    #Redb = matplotlib.colors.LinearSegmentedColormap.from_list("Redb", ['white', 'red'])
    if not color_scale is None:
        color = color_scale
        if center_color and (np.nanmin(color)<0) and (np.nanmax(color)>0):
            cval = min(-np.nanmin(color), np.nanmax(color))
            clip = (-cval, cval)
            color = np.clip(color, *clip)

    if not size_scale is None:
        size=size_scale

    creat_fig = False
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=figsize)
        creat_fig = True

    if not rescale is None:
        hw =image.shape[:2]
        resize = [int(round(hw[0] * rescale ,0)), int(round( hw[1] *rescale ,0))]
        image = skitf.resize(image, resize)
        points = points*rescale

    # ax.scatter(0,0, s=0, alpha=0)
    if not image is None:
        im = ax.imshow(image, origin=origin, **iargs)
    if not points is None:
        if swap_xy:
            pointxy =points[:,[1,0]]
        else:
            pointxy =points
        ip = ax.scatter(pointxy[:,0], pointxy[:,1], s=size, c=color, 
                         vmin= p_vmin, vmax=p_vmax,
                        edgecolor=edgecolor, alpha=alpha, marker=marker, **kargs)

    if not legend is None:
        if use_divider:
            ifig = im if legend=='image' else ip
            divider = make_axes_locatable(ax) 
            # legend_format = matplotlib.ticker.FuncFormatter(lambda x, pos: '')  if legend_format is None else legend_format
            cax1 = divider.append_axes(legend_pos, size=f"{legend_width*100}%", pad=legend_pad)
            cbar = fig.colorbar(ifig, cax = cax1, 
                                format = legend_format)

        else:
            pos = ax.get_position()
            ifig = im if legend=='image' else ip
            cax1 = ifig.add_axes([pos.xmax + legend_pad, 
                                pos.ymin + legend_shift, 
                                legend_width,
                                (pos.ymax-pos.ymin)*legend_height,
                                ])
            cbar = fig.colorbar(ifig, 
                                cax = cax1, 
                                # shrink=0.3,
                                format = legend_format)
        tick_locator = ticker.MaxNLocator(nbins=tick_nbins)
        cbar.locator = tick_locator
        cbar.update_ticks()
        cbar.ax.tick_params(grid_color=legend_grid_color, 
                            labelsize=legend_size,
                            colors=legend_color )
        # cbar.ax.locator_params(nbins=tick_nbins)
        if not set_clim is None:
            ifig.set_clim(*set_clim)

    if axis_off:
        ax.set_axis_off()
    if ticks_off:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    else:
        ax.set_xlabel('y')
        ax.set_ylabel('x')
    if equal_aspect:
        ax.set_aspect('equal', adjustable='box')
    ax.grid(grid)

    if invert_xaxis:
        ax.invert_xaxis()
    if invert_yaxis:
        ax.invert_yaxis()

    if not title is None:
        ax.set_title(title, fontsize=title_fontsize)

    if creat_fig:
        if show is True:
            fig.show()
        elif  show is False:
            plt.close()

def imagehist(img, layout='rgb', figsize=(20,5), 
                   logyscale=True,
                   bin = None, iterval=None, show=True,):

    iterval=(0, 1)
    bins=100
    xtick = np.round(np.linspace(0,1,bins+1, endpoint=True), 2)

    fig, ax = plt.subplots(1,1, figsize=figsize)
    for i in range(img.shape[0]):
        x = img[i].flatten()
        counts, values=np.histogram(x, bins=bins, range=iterval)
        max_value = int(values[np.argmax(counts)])

        xrange = np.array([values[:-1], values[1:]]).mean(0)
        ax.plot(xrange, counts) #, label=f"{i} {layout[i]} {max_value}", color=layout[i])

    ax.legend(loc="best")
    ax.set_xticks(xtick)
    ax.set_xticklabels(
        xtick,
        rotation=90, 
        ha='center',
        va='center_baseline',
        fontsize=10,
    )
    if logyscale:
        ax.set_yscale('log')
    #ax.set_axis_on()
    if show:
        fig.show()
    else:
        plt.close()

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
        fig.show()
    elif show is False:
        plt.close()
    else:
        return fig, axes

def scattersplit(adata, groupby='sampleid', splitby=None, basis='X_umap', method=None,
                legend_loc='on data',
                legend_ax = 'merge',
                show_background=True,
                ncols=5, fsize=5, werror=0, herror=0, size=None, markerscale=5,
                # lloc="best", 
                lloc="center left",
                # bbox_to_anchor=None,# 
                bbox_to_anchor=(1, 0, 0.5, 1),
                lncol=1, mode=None,
                set_aspect=1,
                invert_xaxis=False,
                invert_yaxis=False,
                axis_label=None,
                sharex=True, sharey=True,
                bg_size=4, fontsize=10, bg_color='lightgray', 
                legend_fontsize=10,
                show=True, save=None,  
                left=None, bottom=None,
                right=None, top=None, 
                wspace=None, hspace=None,
                **kargs):
    adata.obsm[f'X_{basis}'] = adata.obsm[basis]
    if method is None:
        if basis in ['X_umap','umap']:
            method = 'umap'
        elif basis in ['X_tsne','tsne']:
            method = 'tsne'
        else:
            method = 'scatter'

    import math
    try:
        G = adata.obs[groupby].cat.remove_unused_categories().cat.categories
    except:
        G = adata.obs[groupby].unique()


    if splitby:
        _data  = adata.obsm[basis]
        _sccor = sc.pl._tools.scatterplots._get_palette(adata, splitby)
        _datacor = adata.obs[splitby].map(_sccor)
        try:
            S = adata.obs[splitby].cat.remove_unused_categories().cat.categories
        except:
            S = adata.obs[splitby].unique()
    else:
        try:
            S = adata.obs[groupby].cat.remove_unused_categories().cat.categories
        except:
            S = adata.obs[groupby].unique()

    if len(G) < ncols: ncols=len(G)
    nrows = math.ceil(len(G)/ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*(fsize+werror), nrows*(fsize+herror)), sharex=sharex, sharey=sharey)
    fig.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    for n,i in enumerate(G):
        if nrows==1:
            AX = axes[n]
        else:
            AX = axes[n//ncols,n%ncols]

        if n == 0:
            if invert_xaxis:
                AX.invert_xaxis()
            if invert_yaxis:
                AX.invert_yaxis()

        if splitby is None:
            if method in ['scatter', 'embedding']:
                eval('sc.pl.%s'%method)(adata, basis=basis, color=groupby, groups =i, show=False,
                        size=size, title=i, legend_loc =legend_loc, ax=AX, **kargs)
            else:
                eval('sc.pl.%s'%method)(adata, color=groupby, groups =i, na_in_legend =False, show=False,
                                        size=size, title=i, legend_loc =legend_loc, ax=AX, **kargs)
        else:
            _idx = adata.obs[groupby]==i
            size = size or 5
            if show_background:
                AX.scatter( _data[:, 0][~_idx], _data[:, 1][~_idx], s=bg_size, marker=".", c=bg_color)
            _sid = [k for k in S if k in adata.obs.loc[_idx, splitby].unique()]
            if len(_sid)>0:
                for _s in _sid:
                    _iidx = ((adata.obs[groupby]==i) & (adata.obs[splitby]==_s))
                    AX.scatter(_data[:, 0][_iidx], _data[:, 1][_iidx], s=size,  marker=".", c=_datacor[_iidx], label=_s)
            AX.set_title(i,size=fontsize)
            if not axis_label is False:
                basis = basis if axis_label is None else axis_label
                AX.set_ylabel((basis+'1').upper(), fontsize = fontsize)
                AX.set_xlabel((basis+'2').upper(), fontsize = fontsize)

            AX.set_yticks([])
            AX.set_xticks([])
            AX.set_aspect(set_aspect)
            AX.grid(False)

            if (not legend_ax is None) and \
                ((type(legend_ax) == int and n==legend_ax) or 
                 (type(legend_ax) == list and n in legend_ax) or
                 (type(legend_ax) == str and legend_ax=='all')):
                AX.legend(loc=lloc, bbox_to_anchor=bbox_to_anchor, fontsize = legend_fontsize,
                            mode=mode, ncol = lncol, markerscale=markerscale)

    if nrows*ncols - len(G) >0:
        for j in range(nrows*ncols - len(G)):
            fig.delaxes(axes[-1][ -j-1])
            #fig.delaxes(axes.flatten()[-j-1])

    if (not legend_ax is None) and (type(legend_ax) == str and legend_ax=='merge'):
        handles, labels = [], []
        for iax in axes.flatten():
            ihandle, ilabel = iax.get_legend_handles_labels()
            handles.extend(ihandle)
            labels.extend(ilabel)
        handles = [dict(zip(labels, handles))[ilb] for ilb in S]
        fig.legend(handles, S, loc=lloc, bbox_to_anchor=bbox_to_anchor, 
                    fontsize = legend_fontsize,
                    mode=mode, ncol = lncol, markerscale=markerscale)

    fig.tight_layout()
    if save:
        fig.savefig(save, bbox_inches='tight')
    if show:
        plt.show()
    elif show is False:
        plt.close()
    else:
        return fig, axes

def field_vector(U, V, h, w , nvec = 50, figsize=(5,5), save=None, show=True):
    nvec = 50  # Number of vectors to be displayed along each image dimension
    step = max(h//nvec, w//nvec)

    y, x = np.mgrid[:h:step, :w:step]
    u_ = U[::step, ::step]
    v_ = V[::step, ::step]
    norm = np.sqrt(U ** 2 + V ** 2)

    fig, ax = plt.subplots(1,1, figsize=figsize)
    ax.imshow(norm)
    ax.quiver(x, y, u_, v_, color='r', units='dots',
            angles='xy', scale_units='xy', lw=3)
    ax.set_title("Optical flow magnitude and vector field")
    ax.set_axis_off()

    fig.tight_layout()
    if save:
        plt.savefig(save)
    if show is None:
        return fig, ax
    elif show is True:
        fig.show()
    else:
        plt.close()

def plot_grid(x,y, ax=None, **kwargs):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    
    ax = ax or plt.gca()
    segs1 = np.stack((x,y), axis=2)
    segs2 = segs1.transpose(1,0,2)
    ax.add_collection(LineCollection(segs1, **kwargs))
    ax.add_collection(LineCollection(segs2, **kwargs))
    ax.autoscale()

def grid_vector(Y, TY, nvec=100, 
                 interp_method = 'cubic',
                 add_scatter = True,
                 point_color = 'black',
                 point_size = 1,
                 grid = False, 
                 grid_color='red',
                 grid_lw = 1,
                 backgroud_color = 'lightgrey',
                 backgroud_lw = 1,
                 axis_off =True,
                 ax = None, figsize=(5,5),  save=None, show=True):
    from scipy.interpolate import griddata

    maxxy = np.ceil( np.c_[Y.max(0), TY.max(0)].max(1))
    minxy = np.floor(np.c_[Y.min(0), TY.min(0)].min(1))

    grid_x,grid_y = np.meshgrid(np.linspace(minxy[0], maxxy[0], nvec),
                                np.linspace(minxy[1], maxxy[1], nvec))
    grid_U = griddata(Y, TY[:,0], (grid_x, grid_y), method=interp_method)
    grid_V = griddata(Y, TY[:,1], (grid_x, grid_y), method=interp_method)

    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=figsize)
    else:
        fig = ax.figure

    plot_grid(grid_x, grid_y, ax=ax, color=backgroud_color, linewidth=backgroud_lw)
    if add_scatter:
        ax.scatter(TY[:,0],TY[:,1], color=point_color, s=point_size)

    plot_grid(grid_U, grid_V, ax=ax, color=grid_color, linewidth=grid_lw)
    ax.grid(grid)
    if axis_off:
        ax.set_axis_off()

    if save:
        fig.tight_layout()
        plt.savefig(save)
    if show is None:
        return fig, ax
    elif show is True:
        fig.tight_layout()
        fig.show()

def arrow_vector(Y, TY , figsize=(10,10), cmap=None,
                  show_points=True,mutation_scale=5, 
                 arrowstyle='->',
                 linewidth=0.1, color='#111111',
                 ax = None,
                 save=None, show=True):
    import matplotlib.patches as patches

    V =TY -Y
    W =  np.sqrt(np.sum(V**2, axis=1))
    W = (W-W.min())/(W.max()-W.min())

    if cmap is None:
        cmap = cmap1
    else:
        if type(cmap) is str:
            cmap = plt.get_cmap(cmap)
    C = cmap(W)


    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=figsize)
    else:
        fig = ax.figure

    if show_points:
        ax.scatter(Y[:,0], Y[:,1], s=0.1, c='r')
        ax.scatter(TY[:,0], TY[:,1], s=0.1, c='b')
    for i, (ix, iv) in enumerate(zip(Y, TY)):
        # ax.arrow(*ix, *iv, color='green', linewidth=0.1, head_width=2, head_length=4)
        # ax.annotate("", ix, iv, arrowprops=dict(arrowstyle="->", lw=0.1, color='grey'))
        p = patches.FancyArrowPatch( 
                ix, iv, 
                arrowstyle=arrowstyle,
                mutation_scale=mutation_scale,
                linewidth=linewidth, 
                color=C[i])
        ax.add_patch(p)

    # ax.set_title("deformble point drif")
    # ax.grid(False)
    # ax.set_axis_off()

    if save:
        fig.tight_layout()
        plt.savefig(save)
    if show is None:
        return fig, ax
    elif show is True:
        fig.tight_layout()
        fig.show()

def arrow_vector2d(Y, TY ,  ax=None, figsize=(5,5), 
                   cmap='hsv', width=1,  show_legend=True, show_points=True,
                   labelsize=6, size=1,edgecolors='None',
                   rasterized=False, set_aspect='equal',
                    dpi = None, save=None, show=True, **kargs):
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=figsize)
    else:
        fig = ax.figure

    T =TY -Y
    M =  np.sqrt(np.sum(T**2, axis=1))
    W = (M-M.min())/(M.max()-M.min())

    if cmap is None:
        C = cmap1(W)
    else:
        C = cmap
        if type(cmap) is str:
            if cmap in plt.colormaps():
                C = plt.get_cmap(cmap)(W)
    # if isinstance(linewidths, (int, float)):
    #     linewidths = np.ones(len(Y))*linewidths

    if show_points:        
        ax.scatter(Y[:,0], Y[:,1], s=size, edgecolors=edgecolors, rasterized=rasterized, c='r')
        ax.scatter(TY[:,0], TY[:,1], s=size, edgecolors=edgecolors, rasterized=rasterized, c='b')

    Q = ax.quiver(Y[:,0], Y[:,1],  T[:,0], T[:,1], color=C,
                  scale_units ='xy', scale = 1.0,
                  cmap=cmap, width=width, **kargs)
    if not set_aspect is None:
        ax.set_aspect(set_aspect, adjustable='box')
    if show_legend:
        cbar = fig.colorbar(Q,shrink=0.15, aspect=10)
        tick_locator = ticker.MaxNLocator(nbins=5)
        cbar.locator = tick_locator
        cbar.update_ticks()
        cbar.ax.tick_params(grid_color='white', 
                            labelsize=labelsize,
                            grid_linewidth=0.1,
                            colors='black' )

    if save:
        fig.tight_layout()
        plt.savefig(save, dpi)
    if show is None:
        return fig, ax
    elif show is True:
        fig.tight_layout()
        fig.show()
