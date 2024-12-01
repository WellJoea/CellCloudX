import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

def cellnet(adata, groupby=None, basis='X_umap', adj='connectivities', isscatter=True, isnet=True,markon=True,
            node_size=1, legendms=4, legendfs=10, font_size=10, figsize=(8,5),gncol=1,
            vmax=None, vmin=None, add_self=False, edge_color='k',
            edge_cmap='default', edge_width_scale=1, show=True, text_col = None,
            legend_text = True,
                text_fontweight = "normal",
                text_fontsize = None,
                text_fontoutline = None,
                family='Arial',
                text_adjust = True,
                adjust_args = {},

            edge_alpha=0.9, save=None):
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from scipy.sparse import issparse

    if edge_cmap=='default':
        import matplotlib.colors as mcolors
        colors=plt.cm.binary(np.linspace(0.9, 1, 128))
        edge_cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

    #plt.rcParams["figure.figsize"] = figsize

    grouplist = adata.obs[groupby].cat.categories
    groupcolor= adata.uns[groupby+'_colors']
    colormap = dict(zip(grouplist, groupcolor))
    map_cor = adata.obs[groupby].map(colormap)

    adjacency = adata.obsp[adj]
    if issparse(adjacency):
        adjacency = adjacency.toarray()
    if not add_self:
        np.fill_diagonal(adjacency, 0)

    adjacency = pd.DataFrame(adjacency, columns=adata.obs_names, index=adata.obs_names)
    if not vmax is None:
        adjacency = adjacency.clip(None, vmax)
    if not vmin is None:
        adjacency[adjacency<vmin] = 0
    
    map_data  = adata.obsm[basis]

    nx_g_solid= nx.Graph(adjacency)
    # dict(zip( adataI.obs_names, adataI.obsm['X_umap']))
    POS = dict(zip( adata.obs_names, map_data))


    widths = [x[-1]["weight"]* edge_width_scale for x in nx_g_solid.edges(data=True)] 
    print('max width:', max(widths))


    fig, ax = plt.subplots(1,1, figsize=figsize)
    ax.grid(False)

    if isnet:
        edges = nx.draw_networkx_edges(
                    nx_g_solid,
                    POS,
                    width=widths,
                    edge_color=edge_color,
                    edge_cmap=edge_cmap,
                    style="-",
                    alpha=edge_alpha,
                    ax=ax,
                )
        # node = nx.draw_networkx_nodes(nx_g_solid, POS, node_size=node_size, node_color=map_cor)
        #node = nx.draw_networkx(nx_g_tree, POS, node_size=0, node_color=groupcolor)
        if markon:
            label = nx.draw_networkx_labels(nx_g_solid,POS,font_size=font_size,font_color='black')

    if isscatter:
        ax.scatter(map_data[:,0], map_data[:,1], s=node_size, alpha=1, c=map_cor)
    if legend_text:
        if text_col is None:
            texts = adata.obs_names
        elif isinstance(text_col, str):
            texts = adata.obs[text_col]
        else:
            texts = text_col
            assert len(texts) == map_data.shape[0]
    
        axtexts = []
        for itx, ipos in zip(texts, map_data):
            if (not itx in ['', 'none', None]):
                tx = ax.text(
                        ipos[0],
                        ipos[1],
                        itx,
                        weight=text_fontweight,
                        verticalalignment="center",
                        horizontalalignment="center",
                        fontsize=text_fontsize,
                        path_effects=text_fontoutline,
                    family=family,
                    )
                axtexts.append(tx)
        if text_adjust:
            try:
                from adjustText import adjust_text
            except:
                raise ValueError('adjust_text is not installed. `pip install adjustText`')
            adjust_text(axtexts, ax=ax, **adjust_args)

    if isscatter:
        legend_elements = [ Patch(facecolor=groupcolor[i], edgecolor=None,label=c) 
                            for i,c in enumerate(grouplist)]
        legend_elements1 = [ Line2D([0], [0], marker='o', color=groupcolor[i], lw=0,
                                label=c, markerfacecolor=None, markersize=legendms)
                            for i,c in enumerate(grouplist)]

        plt.legend(handles=legend_elements1, ncol=gncol, prop={'size':legendfs}, loc='center left', 
                title=groupby,
                bbox_to_anchor=(1.0, 0.5))

    plt.tight_layout()
    if save:
        plt.savefig(save)
        #plt.savefig( self.out, bbox_extra_artists=(leg,), bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
