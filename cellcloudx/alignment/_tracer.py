import scipy.sparse as sp
import numpy as np
import pandas as pd

from ..tools._neighbors import Neighbors,  mtx_similarity
from ..tools._outlier import Invervals
from ..tools._spatial_edges import  coord_edges


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np

def adddynamic(fig, ax, angle=5, elev = 0, interval=50, fps=10, dpi=100, 
                vertical_axis='z',
                bitrate=1800, save=None, show=True):
    from matplotlib import animation
    def rotate(angle):
        ax.view_init(elev =elev, azim=angle, vertical_axis=vertical_axis)
    if show:
        fig.show()
    if save:
        ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=interval)
        if save.endswith('gif'):
            ani.save(save, writer=animation.PillowWriter(fps=fps), dpi=dpi)
        elif save.endswith('mp4'):
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=bitrate)
            ani.save(save, writer=writer, dpi=dpi)

def arrow_vector3d(Y, TY , PX=None, fig=None, ax=None, figsize=(5,5), elev=None, azim=None, roll=None,
                   cmap='hsv', linewidth=1, arrow_length_ratio=0.3,
                   frame_linewidth = 1,
                   frame_type =1,
                   labelsize=6, size=1, color='grey',edgecolors='none',
                   angle=5, interval=50, fps=10,
                    vertical_axis='z',
                    bitrate=1800, write=True,
                    xlims=None, ylims=None,zlims=None,
                     dpi = 300, save=None, show=True):
    if (ax is None):
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(projection='3d')
    ax.view_init(elev=elev, azim=azim, roll=roll)
    
    # Y = adataP.obsm['spatial']
    # Y = adataP.obsm['ccf_l']
    # TY = adataP.obsm['ccf_d']
    V =TY -Y
    W =  np.sqrt(np.sum(V**2, axis=1))
    W = (W-W.min())/(W.max()-W.min())
    C = plt.cm.hsv(W)
    C = cc.pl.cmap1(W)

    Q = ax.quiver(Y[:,0], Y[:,1], Y[:,2], V[:,0], V[:,1],V[:,2], color=C,
                  cmap=cmap, linewidth=0.2, arrow_length_ratio=arrow_length_ratio)
    # ax.scatter(Y[:,0], Y[:,1], Y[:,2], color='red', s=0.1)
    ax.grid(True)
    # ax.set_axis_off()

    ax.set_axis_off()
    cc.pl.add_frame(ax, frame_linewidth=frame_linewidth, frame_type=frame_type,
                   xlims=xlims, ylims=ylims,zlims=zlims)
    cbar = fig.colorbar(Q,shrink=0.15, aspect=10)
    tick_locator = ticker.MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.update_ticks()
    cbar.ax.tick_params(grid_color='white', 
                        labelsize=labelsize,
                        grid_linewidth=0.1,
                        colors='black' )
    ax.set_aspect('equal', 'box')

    if not PX is None:
        ax.scatter(PX[:,0], PX[:, 1], PX[:,2], s=size, edgecolors=edgecolors, color=color)

    if write:
        fig.tight_layout()
        if save:
            save_type = save.split('.')[-1]
            if save_type in ['gif', 'mp4']:
                adddynamic(fig, ax, angle=angle, elev = elev, interval=interval, fps=fps, dpi=dpi, 
                            vertical_axis=vertical_axis,
                            bitrate=bitrate, save=save, show=show)
            else:
                plt.savefig(save)

        if show is None:
            return fig, ax
        elif show is True:
            fig.show()
        else:
            plt.close()

def max_project(P, axis=0, outlier=0, score_thred=0):
    if sp.issparse(P):
        q_idx = sp.csr_matrix.argmax(P.tocsr(), axis=axis).flatten()
        p_score = P.max(axis).toarray().flatten()
    else:
        q_idx = np.argmax(P, axis=axis)
        p_score = P.max(axis)

    top_n = int((1- outlier) * P.shape[0])
    top_n = min(top_n, np.sum(p_score >= score_thred))

    r_idx = np.argpartition(p_score, -top_n)[-top_n:]
    p_score = p_score[r_idx]
    q_idx = q_idx[r_idx]

    sidx = np.argsort(p_score)[::-1]
    q_idx = q_idx[sidx]
    r_idx = r_idx[sidx]
    p_score = p_score[sidx]

    if axis == 0:
        return [r_idx, q_idx, p_score]
    elif axis == 1:
        return [q_idx, r_idx, p_score]

def agg_project(P, label, top_k, axis=1, score_thred=0, na_value='None', agg='mode'):
    if sp.issparse(P):
        P = P.toarray()
    label = np.array(label)

    idx = np.argpartition(P, -top_k, axis=axis)[:, -top_k:]
    V = np.take_along_axis(P, idx, axis=axis)
    M = label[idx]
    M[V<score_thred] = na_value

    if agg =='mode':
        return pd.DataFrame(M).mode(axis=axis).values[:,0]
    elif agg =='mean':
        return np.nanmean(M, axis=axis)
    elif agg =='sum':
        return np.nansum(M, axis=axis)

def Tracer( X=None, Y=None, X_feat=None, Y_feat=None, P = None, label=None,
           knn = 11, radius=None, simi_method='cosine',
            score_thred = 0, CI=0.95 , outlier=0, axis=None,  na_value='None', agg='mode', 
            kd_method='annoy', kernel='poi'): 
    if P is None:
        Q,D = Y.shape
        R = X.shape[0]
        [src, dst, dist] = coord_edges(X, Y, knn = knn, radius=radius, method=kd_method)

        if (not CI is None) and (CI < 1):
            radiu_trim = Invervals(dist, CI=CI,  kernel=kernel, tailed ='two')[2]
            idx = (dist < radiu_trim)
            src = src[idx]
            dst = dst[idx]
            dist = dist[idx]

        if (X_feat is None) or (Y_feat is None) :
            P = sp.csr_array((np.exp(-dist), (dst, src)), shape=(Q, R))
        else:
            esimi = mtx_similarity(X_feat, Y_feat, method=simi_method, pairidx=[src, dst])
            esimi[esimi<score_thred] = 0
            P = sp.csr_array((esimi, (dst, src)), shape=(Q, R))
        axis=1
    else:
        P = P
        axis=0 if axis is None else axis
    if label is None:
        return max_project(P, axis=axis, outlier=outlier, score_thred=score_thred)
    else:
        return agg_project(P, label, knn, axis=axis,
                    score_thred=score_thred, na_value=na_value, agg=agg)

def sankeypairs(adataI, ssnn_idx, groupby = 'sampleid', matchby='cell_region', alpha = 0.8,
                 min_count=20, save=None, 
                 font_size=10,
                 height=1000, width=800,):
    m_order = adataI.obs[matchby].cat.categories
    g_order = adataI.obs[groupby].cat.categories

    colordict = dict(zip(m_order, adataI.uns[f'{matchby}_colors']))

    def hex_to_rgb(value, alpha=None):
        value = value.strip("#") # removes hash symbol if present
        lv = len(value)
        rgbs = [ int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3) ]
        if not alpha is None:
            rgbs = [*rgbs, alpha] 
        return tuple(rgbs)

    colordict1 = { k: f'rgba{hex_to_rgb(v, alpha)}' for k, v in colordict.items() }

    src = ssnn_idx[0]
    dst = ssnn_idx[1]
    src_ct = adataI.obs[matchby].iloc[src].values
    dst_ct = adataI.obs[matchby].iloc[dst].values
    src_id  = adataI.obs[groupby].iloc[src].values
    dst_id  = adataI.obs[groupby].iloc[dst].values

    sankdf = pd.DataFrame({'src':src, 'dst':dst, 'src_ct' : src_ct, 
                        'dst_ct' :dst_ct, 'src_id':src_id, 'dst_id':dst_id})
    sankdf['src_id'] = pd.Categorical(sankdf['src_id'], categories=g_order)
    sankdf['dst_id'] = pd.Categorical(sankdf['dst_id'], categories=g_order)
    sankdf.sort_values(by=['dst_id', 'src_id'], inplace=True)
    sankdf['s1'] = sankdf['dst_id'].astype(str) + '-' +  sankdf['dst_ct'].astype(str)
    sankdf['s2'] = sankdf['src_id'].astype(str) + '-' +  sankdf['src_ct'].astype(str)

    m_dict = { f'{g}-{m}': m for m in m_order for g in g_order }
    g_dict = { f'{g}-{m}': g for m in m_order for g in g_order }

    KK = sankdf[['s1','s2'] ].value_counts().reset_index()
    KK = KK[(KK['count']>min_count)]

    sdname = KK[['s1','s2'] ].values.flatten()
    ages = [ int(i.split('-')[0][1:]) for i in sdname]
    ctype = [ i.split('-')[1] for i in sdname]
    dd = pd.DataFrame({'sdname':sdname, 'ages': ages, 'ctype':ctype})
    dd = dd.sort_values(by=['ages', 'ctype']).reset_index(drop=True)
    dddict = dict(zip(dd['sdname'], dd.index))

    dd['node_name'] =  dd['sdname'].str.split('-').str[1]
    dd['node_color'] = dd['node_name'].replace(colordict)

    KK['source_color'] =KK['s1'].str.split('-').str[1].map(colordict1)
    KK['s1m'] = pd.Categorical( KK['s1'].map(m_dict), categories=m_order)
    KK['s2m'] = pd.Categorical( KK['s2'].map(m_dict), categories=m_order)
    KK.sort_values(by=['s1m', 'count', 's2m'], ascending=[True, False, True], inplace=True)

    MM = KK.replace(dddict)

    import plotly.graph_objects as go
    fig = go.Figure(data=[go.Sankey(
        node = dict(
            pad = 20,
            thickness = 60,
            line = dict(color = "black", width = 0.5),
            label = dd['node_name'],
            color = dd["node_color"]
        ),
        link = dict(
            source = MM['s1'], # indices correspond to labels, eg A1, A2, A1, B1, ...
            target = MM['s2'],
            value = KK['count'],
            color = KK['source_color'].values,
    ))])

    fig.update_layout(title_text=matchby, font_size=font_size,
                            height=height, width=width,)
    
    if save:
        fig.write_html(save)
    fig.show()