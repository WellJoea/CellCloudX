import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def searchidx(n_node,
                root=None, 
                labels=None,
                step=1, 

                regist_pair=None,
                full_pair=False,
                show_tree=False, 
                figsize=None,
                self_pair = False,
                layout="spectral", 
                search_type='dfs',
                drop_align_leaves =False, 
                **kargs):
    if not full_pair:
        assert  (not root is None) or (not regist_pair is None), 'A least one of root and regist_pair is not None.'

    align_idx = None
    trans_idx = None

    if full_pair:
        align_idx =  [ (i,j) for i in range(n_node) for j in range(i+1, n_node) ]

    if not regist_pair is None:
        align_idx = buildidx(n_node=n_node, 
                                step=step, 
                                root=None, 
                                edges=regist_pair,
                                show_tree=False, 
                                self_pair=self_pair,
                                search_type=search_type,
                                layout=layout, **kargs)
    if not root is None:
        trans_idx = buildidx(n_node=n_node, 
                                    step=step, 
                                    root=root, 
                                    edges=None,
                                    show_tree=False, 
                                    self_pair=self_pair,
                                    search_type=search_type,
                                    layout=layout, **kargs)

    align_pair = (trans_idx if align_idx is None else align_idx).copy()
    trans_pairs = (align_idx if trans_idx is None else trans_idx).copy()

    if drop_align_leaves:
        align_pair = digraph( align_pair, root=root, show_tree=False, 
                             search_type=search_type, drop_leaves=True)

    if show_tree:
        if align_pair == trans_pairs:
            fig, axes = plt.subplots(1,1, figsize=(6,6) if figsize is None else figsize)
            draw_graph(align_pair, root=root, ax=axes, layout=layout, show=False, title='align pair', **kargs)
            plt.show()
        else:
            fig, axes = plt.subplots(1,2, figsize=(12,6) if figsize is None else figsize)
            draw_graph(align_pair, root=root, ax=axes[0], layout=layout, show=False, title='align pair', **kargs)
            draw_graph(trans_pairs, root=root, ax=axes[1], layout=layout,show=False, title='trans_pairs', **kargs)
            plt.show()

    if not labels is None:
        align_pair = [ (labels[k], labels[v]) for k,v in align_pair ]

    return [align_pair, trans_pairs]

def buildidx(n_node=None, 
                step=1, 
                root=None, 
                self_pair=False,
                edges=None,
                show_tree=False, 
                search_type='dfs', 
                layout="spectral", **kargs):
    '''
    edges = [(1,3), (1,2), (4,5), (5,6), (4,7)]
    KK =buildidx( edges=edges, step=3, layout="circular")
    list(KK.dfs_edges)
    '''
    if edges is None:
        source, target = treeidx(n_node,  step=step, root=root, self_pair=self_pair )
        edges = zip(source, target)

    return digraph(edges, 
                root=root,
                show_tree=show_tree, 
                layout=layout, 
                search_type=search_type, 
                **kargs)


def digraph( edges, root=None, drop_leaves=False, show_tree=False, search_type='dfs', **kargs):
    G = nx.DiGraph()
    G.add_edges_from(edges)

    if drop_leaves:
        D = [x for x in G.nodes() if G.out_degree(x)==0]
        if len(D)>0:
            G.remove_nodes_from(D)

    if show_tree:
        draw_graph( edges, root=root, show=show_tree, **kargs)

    if search_type == 'dfs':
        # dfs_edges = list(nx.dfs_edges(G, source=root, depth_limit=None))
        dfs_edges = list(nx.edge_dfs(G, source=root))
        return dfs_edges
    elif search_type == 'bfs':
        try:
            # bfs_edges = list(nx.bfs_edges(G, source=root, 
            #                                     depth_limit=None, 
            #                                     sort_neighbors=None))
            bfs_edges = list(nx.edge_bfs(G, source=root))
        except:
            bfs_edges = []
            roots = list(v for v, d in G.in_degree() if d == 0)
            for iroot in roots:
                bfs_edges += list(nx.edge_bfs(G, source=iroot, 
                                                    # depth_limit=None, 
                                                    # sort_neighbors=None
                                            ))
        return bfs_edges

def draw_graph(edges, root=None, linewidths=2, font_weight='normal', ax=None, title=None,
              figsize=(7,7), prog="twopi", layout="pydot", show =True, font_size=8, node_size=60):
    G = nx.DiGraph()
    G.add_edges_from(edges)

    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=figsize)

    if layout=="pydot":
        try:
            from networkx.drawing.nx_pydot import graphviz_layout
            pos = graphviz_layout(G, prog=prog, root=root)
            nx.draw(G, pos, node_size=node_size, alpha=1, font_size = font_size,
                    #connectionstyle="arc3,rad=-0.2",
                        font_weight=font_weight, ax=ax,
                    linewidths  = linewidths, node_color="red", with_labels=True)
            # plt.show()
        except:
            print('pydot not found.'
                  'try to `pip install pydot pygraphviz and conda install anaconda::graphviz`.'
                  'or  use layout with spectral, spring, circular, kawai')
    else:
        if layout == "spectral":
            nplt = nx.draw_spectral
        elif layout == 'spring':
            nplt = nx.draw_spring
        elif layout == 'circular':
            nplt = nx.draw_circular
        elif layout == 'kawai':
            nplt = nx.draw_kamada_kawai
        nplt(G,
                alpha=1, font_size = font_size, linewidths  = linewidths,
                font_weight=font_weight, ax=ax,
                node_color="red", with_labels=True)
    if not title is None:
        ax.set_title(title)
    if show:
        plt.show()

def treeidx0(n_node, step=1, root=None, self_pair=False ):
    root = n_node // 2 if root is None else root

    target1 = np.arange(0, root)[::-1]
    source1 = sum([ [i+1]+[i]*(step-1) for i in range(root-1, -1, -step) ], [])
    source1 = np.array(source1)[:root]

    target2 = np.arange(root, n_node, 1)
    source2 = sum([ [i-1]+[i]*(step-1)  for i in range(root, n_node, step) ], [])
    source2 = np.array(source2)[:(n_node-root)]
    source2[0] = root

    source = np.concatenate([source1, source2])
    target = np.concatenate([target1, target2])
    sortidx= np.argsort(target)
    source = np.int64(source[sortidx])
    target = np.int64(target[sortidx])
    return [source, target]

def treeidx(n_node, step=1, root=None, self_pair=False ):
    root = n_node // 2 if root is None else root

    target1 = np.arange(0, root)[::-1]
    source1 = np.repeat(np.arange(root, -step, -step), step)[:root]

    target2 = np.arange(root+1, n_node, 1)
    source2 = np.repeat(np.arange(root, n_node+step, step), step)[:(n_node-root-1)]

    if self_pair:
        target3 = source3 =  np.arange(0, n_node)
    else:
        # target3 = source3 = [root]
        target3 = source3 = []

    source = np.concatenate([source1, source3, source2]).astype(np.int64)
    target = np.concatenate([target1, target3, target2]).astype(np.int64)
    
    # sortidx= np.argsort(target)
    # source = np.int64(source[sortidx])
    # target = np.int64(target[sortidx])
    return [source, target]


def neighboridx(start, end, top=None, values=None, show_tree=False):
    top = 0 if top is None else top
    if not values is None:
        start = 0
        end = len(values)-1
        top = list(values).index(top)
    back = np.arange(top, start, -1)
    forw = np.arange(top,end)
    source= np.concatenate([back, [top], forw])
    target= np.concatenate([back-1, [top], forw+1])
    if not values is None:
        source = np.array(values)[source]
        target = np.array(values)[target]
    G = nx.DiGraph()
    G.add_edges_from(zip(source, target))
    if show_tree:
        nx.draw(G, with_labels=True, font_weight='bold', layout="spring")
    return G