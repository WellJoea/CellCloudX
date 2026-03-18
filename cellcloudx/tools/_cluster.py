import os
import numpy as np
import pandas as pd

import numpy as np
from anndata import AnnData

import leidenalg as la
import igraph as ig

def cutsne():
    import cudf
    import cuml
    import cudf, requests
    import cupy

def ftsne(data,
            use_rep='X_pca', 
            add_key='X_tsne',
            inplace=True, 
            perplexity = 30, 
            n_components=2,
            n_jobs=5,
            metric="euclidean", #cosine
            initialization='pca', #random
            learning_rate= 'auto', #'auto',
            early_exaggeration_iter=250,
            early_exaggeration='auto',
            n_iter=500,
            exaggeration=None,
            initial_momentum=0.8,
            final_momentum=0.8,
            verbose=True,
            random_state=19491001,
            **kargs):

    data_is_AnnData = isinstance(data, AnnData)
    X = data.obsm[use_rep].copy() if data_is_AnnData else data.copy()

    if type(perplexity) in [int, float]:
        import openTSNE
        tsne = openTSNE.TSNE(
            n_components=n_components,
            n_jobs=n_jobs,
            perplexity=perplexity,
            metric=metric,
            initialization=initialization,
            learning_rate=learning_rate,
            early_exaggeration_iter=early_exaggeration_iter,
            early_exaggeration=early_exaggeration,
            n_iter=n_iter,
            exaggeration=exaggeration,
            initial_momentum=initial_momentum,
            final_momentum=final_momentum,
            random_state=random_state,
            verbose=verbose,
            **kargs
        )
        embedding_train = tsne.fit(X)
    else:
        import openTSNE
        affinities_multiscale_mixture = openTSNE.affinity.Multiscale(
            X,
            perplexities=perplexity,
            metric=metric,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )
        if initialization=='pca':
            init = openTSNE.initialization.pca(X, n_components=n_components, random_state=random_state)
        else:
            init = openTSNE.initialization.random(X, n_components=n_components,  random_state=random_state)

        tsne = openTSNE.TSNE(n_jobs=n_jobs,
                            n_components=n_components,
                            metric=metric,
                            learning_rate=learning_rate,
                            early_exaggeration_iter=early_exaggeration_iter,
                            early_exaggeration=early_exaggeration,
                            n_iter=n_iter,
                            exaggeration=exaggeration,
                            initial_momentum=initial_momentum,
                            final_momentum=final_momentum,
                            random_state=random_state,
                            verbose=verbose,
                            **kargs
            )
        embedding_train = tsne.fit(
            affinities=affinities_multiscale_mixture,
            initialization=init,
        )

    if data_is_AnnData:
        data = data if inplace else data.copy()
        data.obsm[add_key] = np.array(embedding_train)
        if not inplace:
            return data
    else:
        return embedding_train

def rmclust(adata, nclust=None, use_rep='pca', X=None,  modelNames='EEE',
             add_key='mclust', copy=False,
             R_HOME = None, R_USER=None, verbose=False,
            random_seed=491001):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    https://mclust-org.github.io/mclust/reference/mclustModelNames.html
    modelNames: A vector of character strings indicating the models to be
          fitted in the EM phase of clustering. The default is:

            • for univariate data (d = 1): ‘c("E", "V")’

            • for multivariate data (n > d): all the models available
              in ‘mclust.options("emModelNames")’
                "EII" "VII" "EEI" "VEI" "EVI" "VVI" "EEE" "VEE" "EVE" 
                "VVE" "EEV" "VEV" "EVV" "VVV"

            • for multivariate data (n <= d): the spherical and
              diagonal models, i.e. ‘c("EII", "VII", "EEI", "EVI",
              "VEI", "VVI")’
    """
    if R_HOME:
        os.environ['R_HOME'] = R_HOME
    if R_USER:
        os.environ['R_USER'] = R_USER
    if copy:
        adata = adata.copy()
    if X is None:
        X = adata.obsm[use_rep]

    np.random.seed(random_seed)
    from rpy2 import robjects as robj
    robj.r.library("mclust")

    import rpy2.robjects.numpy2ri as npr
    npr.activate()
    r_random_seed = robj.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robj.r['Mclust']
    summary = robj.r['summary']
    nclust = robj.rinterface.NULL if nclust is None else nclust #robj.r.seq(1, nclust)

    #mclust::adjustedRandIndex(kowalczyk.integrated$seurat_clusters, cell_info$cell_type_label)
    res = rmclust(npr.numpy2rpy(X), nclust, modelNames=modelNames)
    if verbose:
        print(summary(res))
    try:
        clust = np.int32(res[-2])-1
        adata.obs[add_key] = pd.Categorical(clust.astype(str),
                                    categories=np.unique(clust).astype(str))
    except:
        adata.obs[add_key] = '0'

    if copy:
        return adata

def pandas2adjacency(df, gtype='igraph', drop_neg_weights=False ):
    import igraph as ig
    import networkx as nx
    import scipy as sci
    import numpy as np

    df = df.copy()
    col = df.columns 
    row = df.index
    if drop_neg_weights:
        df[df<0] = 0
    if (len(col) == len(row)) and (row != row).sum() ==0: # fast
        if gtype in ['nx', 'networkx']:
            G = nx.from_pandas_adjacency(df)
        elif gtype in ['nx2ig']:
            nG = nx.from_pandas_adjacency(df)
            G = ig.Graph.from_networkx(nG, vertex_attr_hashable="name")
        elif gtype in ['igwa']:
            G = ig.Graph.Weighted_Adjacency(df, 
                                            attr='weight',
                                            mode='undirected')
        elif gtype in ['iga']:
            G = ig.Graph.Adjacency(df, 
                                    mode='undirected')

        elif gtype in ['ig', 'igraph']:
            adj = sci.sparse.csr_matrix(df.values)
            sources, targets = adj.nonzero()
            weights = adj[sources, targets]
            if isinstance(weights, np.matrix):
                weights = weights.A1
            G = ig.Graph(directed=None)
            G.add_vertices(adj.shape[0])
            G.add_edges(list(zip(sources, targets)))
            G.vs["name"] = col
            G.es['weight'] = weights

    else:
        g_df = df.melt(ignore_index=False).reset_index()
        g_df.columns = ['source', 'target', 'value']

        if gtype in ['nx', 'networkx']:
            G = nx.Graph()
            G.add_weighted_edges_from(g_df.to_numpy())
            #H = nx.relabel_nodes(G, rename_label)
        elif gtype in ['nx2ig']:
            G = nx.Graph()
            G.add_weighted_edges_from(g_df.to_numpy())
            G = ig.Graph.from_networkx(G, vertex_attr_hashable="name")
        elif gtype in ['ig','igraph']:
            G = ig.Graph.DataFrame(g_df, directed=False, use_vids=False)
            G.es["weight"] = g_df['value']

        if G.is_bipartite():
            G.vs['type'] = np.where(np.isin(G.vs['name'], list(row)), 0, 1)
    return G

def merge_small_cluster(partition, optimiser=None, min_comm_size=100, max_comm_num=150):
    import leidenalg as la
    
    optimiser = la.Optimiser() if optimiser is None else optimiser
    optimiser.consider_empty_community = False

    cluster, sizes = np.unique(partition.membership, return_counts=True)
    if min_comm_size is None:
        min_comm_size = np.min(sizes)
    if max_comm_num is None:
        max_comm_num = len(sizes)

    cont = 1
    while min_comm_size > np.min(sizes) or max_comm_num < len(sizes):
        aggregate_partition = partition.aggregate_partition(partition)
        cluster, sizes = np.unique(partition.membership, return_counts=True)
        minidx = np.where(sizes == np.min(sizes))[0][-1]
        smallest = cluster[minidx]

        fixed = np.setdiff1d(cluster, smallest)
        max_diff = -np.inf
        cluster_idx = -1
        for fix in fixed:
            diff = aggregate_partition.diff_move(smallest, fix)
            if diff > max_diff:
                max_diff = diff
                cluster_idx = fix

        aggregate_partition.move_node(smallest, cluster_idx)
        if cont <11:
            print(f'merge cluster {smallest} -> {cluster_idx}')
        elif cont ==11:
            print(f'merge cluster ... -> ...')
        cont += 1

        if not (aggregate_partition.sizes()[-1] == 0):
            optimiser.optimise_partition(aggregate_partition)
        partition.from_coarse_partition(aggregate_partition)
        sizes = partition.sizes()
    # optimiser.optimise_partition(partition)
    return partition

def leiden_graph(
    G: ig.Graph ,
    drop_neg_weights=False,
    graph_type=None,
    resolution: float = 1,
    random_state: int = 0,
    directed: bool = False,
    use_weights: bool = True,
    n_iterations: int = -1,
    max_comm_size: int = 0,
    initial_membership: np.array = None,
    max_comm_num: int = None,
    min_comm_size: int = None,
    layer_weights=None,
    is_membership_fixed=None,
    partition_type: la.VertexPartition.MutableVertexPartition = None,
    **partition_kwargs,
) -> np.array:

    if graph_type == 'multiplex':
        drop_neg_weights = False
        G_pos = G.subgraph_edges(G.es.select(weight_gt = 0), delete_vertices=False)
        G_neg = G.subgraph_edges(G.es.select(weight_lt = 0), delete_vertices=False)
        G_neg.es['weight'] = [-w for w in G_neg.es['weight']]
        part_pos = la.RBConfigurationVertexPartition(G_pos, weights='weight', resolution_parameter=resolution)
        part_neg = la.RBConfigurationVertexPartition(G_neg, weights='weight', resolution_parameter=resolution)

        optimiser = la.Optimiser()
        optimiser.set_rng_seed(random_state)
        optimiser.consider_empty_community = False
        diff = optimiser.optimise_partition_multiplex([part_pos, part_neg],
                                                      layer_weights=[1,-1],
                                                      n_iterations=n_iterations,
                                                      is_membership_fixed=is_membership_fixed)
        part = part_pos #[part_pos, part_neg]
    elif graph_type == 'bipartite':
        p_01, p_0, p_1 = la.CPMVertexPartition.Bipartite(G,
                                                 initial_membership=initial_membership,
                                                 weights = np.array(G.es['weight']).astype(np.float64),
                                                 degree_as_node_size=False,
                                                 resolution_parameter_01=resolution,
                                                 resolution_parameter_0=0,
                                                 resolution_parameter_1=0)
        optimiser = la.Optimiser()
        optimiser.set_rng_seed(random_state)
        optimiser.consider_empty_community = False
        diff = optimiser.optimise_partition_multiplex([p_01, p_0, p_1],
                                                      layer_weights=[1, -1, -1],
                                                      n_iterations=n_iterations,
                                                      is_membership_fixed=is_membership_fixed)
        part = p_01 #[part_pos, part_neg]

    else:
        partition_kwargs = dict(partition_kwargs)
        if partition_type is None:
            partition_type = la.RBConfigurationVertexPartition

        partition_kwargs['n_iterations'] = n_iterations
        partition_kwargs['max_comm_size'] = max_comm_size
        partition_kwargs['initial_membership'] = initial_membership

        if resolution is not None:
            if partition_type==la.CPMVertexPartition.Bipartite:
                partition_kwargs['resolution_parameter_01'] = resolution
                partition_kwargs['degree_as_node_size'] = False
            else:
                partition_kwargs['resolution_parameter'] = resolution
        # clustering proper
    
        if drop_neg_weights:
            G = G.subgraph_edges(G.es.select(weight_gt = 0), delete_vertices=False)

        if use_weights:
            partition_kwargs['weights'] = np.array(G.es['weight']).astype(np.float64)

        partition_kwargs['seed'] = random_state
        part = la.find_partition(G, partition_type, **partition_kwargs)
        if not((min_comm_size is None) and (max_comm_num is None)):
            part = merge_small_cluster(part, 
                                        optimiser=la.Optimiser(),
                                        min_comm_size=min_comm_size,
                                        max_comm_num=max_comm_num)
    groups = np.array(part.membership)
    print('partition number: ', len(set(groups)))
    return groups

