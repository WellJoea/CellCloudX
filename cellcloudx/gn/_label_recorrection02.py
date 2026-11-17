import torch as th
import torch.nn as nn
import torch.nn.functional as Fun

from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, TransformerConv
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering, DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import connected_components
from scipy.spatial import Delaunay, cKDTree
from umap import UMAP
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import pykeops
    pykeops.set_verbose(False)
    from pykeops.torch import LazyTensor
except:
    pass
from ..tools._neighbors import Neighbors



def _local_bandwidth(Z, knn=15):
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=knn+1).fit(Z)
    d, _ = nn.kneighbors(Z)
    # 去掉自身距离后取中位数
    return np.median(d[:,1:], axis=1) + 1e-12  # [M]

def fused_knnGraph(X, F, k_small=10, k_large=30,
                   method='sknn', beta=0.5, alpha=[0.3,1.0],
                   adapt_bandwidth=True, dtype=th.float32, verbose=1):
    # 归一化
    Xn = centerlize(X)[0]
    Fn = center_normalize(F)
    Z  = np.hstack([alpha[0]*Xn, alpha[1]*Fn])

    # 两个尺度邻接并集
    s1 = coord_edges(X, knn=k_small, method=method)
    s2 = coord_edges(X, knn=k_large, method=method)
    src = np.concatenate([s1[0], s2[0]]); dst = np.concatenate([s1[1], s2[1]])

    # 去重
    E = np.vstack([dst, src]).T
    E = np.unique(E, axis=0)
    dst, src = E[:,0], E[:,1]

    # 距离（空间与表达）
    dx = np.linalg.norm(Xn[src] - Xn[dst], axis=1)**2
    df = np.linalg.norm(Fn[src] - Fn[dst], axis=1)**2

    if adapt_bandwidth:
        sigX = _local_bandwidth(Xn, knn=15)
        sigF = _local_bandwidth(Fn, knn=15)
        sig_x = np.sqrt(sigX[src]*sigX[dst])
        sig_f = np.sqrt(sigF[src]*sigF[dst])
    else:
        sig_x = sig_f = 1.0

    kx = np.exp(- dx / (sig_x**2))
    kf = np.exp(- df / (sig_f**2))
    W  = beta * kx + (1.0 - beta) * kf   # 也可用乘法核： (kx**beta) * (kf**(1-beta))

    edge_index = th.tensor(np.array([dst, src]), dtype=th.long)
    data = Data(x=th.tensor(Z, dtype=dtype),
                edge_index=edge_index,
                edge_attr=th.tensor(W, dtype=dtype))
    return data

class BAGCN(th.nn.Module):
    def __init__(self, in_dim, hidden=[128,64,64], num_classes=5, slope=0.2, dropout=0.5):
        super().__init__()
        if isinstance(hidden, int): hidden=[hidden]
        dims=[in_dim]+hidden
        self.convs=th.nn.ModuleList([GCNConv(dims[i], dims[i+1]) for i in range(len(dims)-1)])
        self.cls  = GCNConv(dims[-1], num_classes)
        self.act  = th.nn.LeakyReLU(slope)
        self.dropout=dropout

        H=dims[-1]
        self.edge_mlp=th.nn.Sequential(
            th.nn.Linear(3*H, H), th.nn.LeakyReLU(0.2),
            th.nn.Linear(H,1)
        )

    def node_forward(self, x, edge_index, edge_weight):
        h=x
        for conv in self.convs:
            h=conv(h, edge_index, edge_weight)
            h=self.act(h)
            h=Fun.dropout(h, p=self.dropout, training=self.training)
        logits=self.cls(h, edge_index, edge_weight)
        return logits, h

    def forward(self, x, edge_index, edge_weight, return_embedding=False, edge_feat=False):
        logits, h = self.node_forward(x, edge_index, edge_weight)
        if not edge_feat and not return_embedding:
            return logits
        row, col = edge_index
        hij = th.cat([h[row], h[col], (h[row]-h[col]).abs()], dim=1)
        bij = th.sigmoid(self.edge_mlp(hij)).squeeze(-1)
        if return_embedding and edge_feat:
            return logits, h, bij
        elif return_embedding:
            return logits, h
        else:
            return logits, bij

def class_weights_from_mask(y, mask, K):
    y = np.asarray(y); mask=np.asarray(mask)
    cnt = np.bincount(y[mask], minlength=K) + 1e-6
    w = (cnt.sum() / cnt)  # 反频率
    w = w / w.mean()
    return th.tensor(w, dtype=th.float32)

def temperature_scale(logits, labels, mask, init_T=1.0, steps=100, lr=0.01):
    # 简单温度拟合（只在已标注上）
    T = th.tensor([init_T], requires_grad=True)
    opt = th.optim.LBFGS([T], lr=lr, max_iter=steps)
    y = th.tensor(labels, dtype=th.long, device=logits.device)
    m = th.tensor(mask, dtype=th.bool, device=logits.device)
    def closure():
        opt.zero_grad()
        z = logits[m] / T.clamp_min(1e-2)
        loss = Fun.cross_entropy(z, y[m])
        loss.backward()
        return loss
    opt.step(closure)
    return float(T.detach().cpu().item())

@th.no_grad()
def update_ema(student, teacher, ema=0.99):
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(ema).add_(ps.data, alpha=(1-ema))

def train_bagcn_full(data, labels, label_mask,
                     n_epochs=200, lr=1e-2, weight_decay=5e-4,
                     hidden=[128,64,64], smooth_lambda=1.0, ent_lambda=0.1,
                     contrast_lambda=0.1, margin=1.0, proto_lambda=0.1,
                     cons_lambda=0.1, dropedge_rate=0.1,
                     use_ema=True, ema_decay=0.99,
                     label_smoothing=0.05,
                     dtype=th.float32, device=None, mc_dropout=False):
    device = th.device('cuda' if th.cuda.is_available() else 'cpu') if device is None else device
    data = data.to(device)
    X=data.x; Ei=data.edge_index; Ew=data.edge_attr.squeeze()
    M, indim = X.shape

    Y, Y_orders = label_binary(labels, mask=label_mask)
    Yt = th.tensor(Y, dtype=th.long, device=device)
    K = int(np.max(Y) + 1)

    model = BAGCN(indim, hidden, K, dropout=0.5).to(device)
    teacher = BAGCN(indim, hidden, K, dropout=0.0).to(device)
    teacher.load_state_dict(model.state_dict())
    for p in teacher.parameters(): p.requires_grad=False

    # 类权重 + label smoothing
    cls_w = class_weights_from_mask(Y, label_mask, K).to(device)
    opt = th.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    lid = th.where(th.tensor(label_mask, device=device))[0]
    uid = th.where(~th.tensor(label_mask, device=device))[0]

    from torch_geometric.utils import dropout_edge
    from tqdm import tqdm
    pbar = tqdm(range(n_epochs), colour='red', desc='BA-GCN++')

    for _ in pbar:
        model.train(); opt.zero_grad()

        # DropEdge一致性视图
        Ei1, Ew1 = dropout_edge(Ei, p=dropedge_rate, force_undirected=True), Ew
        Ei2, Ew2 = dropout_edge(Ei, p=dropedge_rate, force_undirected=True), Ew

        logits, h, bij = model(X, Ei1[0], Ew1, return_embedding=True, edge_feat=True)
        probs = Fun.softmax(logits, dim=1)

        # 监督交叉熵（label smoothing）
        with th.no_grad():
            y_smooth = th.zeros(len(lid), K, device=device) + label_smoothing / K
            y_smooth.scatter_(1, Yt[lid][:,None], 1.0 - label_smoothing + label_smoothing/K)
        loss_sup = Fun.kl_div((logits[lid]).log_softmax(dim=1), y_smooth, reduction='batchmean')
        loss_sup = (loss_sup * 1.0)  # 可乘 cls_w 均值，但LS已缓解不均衡

        # 边界感知平滑
        row, col = Ei1[0]
        diff_p = (probs[row]-probs[col]).pow(2).sum(dim=1)
        loss_gl = (Ew1.squeeze()* (1.0-bij) * diff_p).sum() * 0.5 / M

        # 边界对比
        dij = (h[row]-h[col]).pow(2).sum(dim=1).sqrt()+1e-9
        pull = (Ew1*(1.0-bij)*dij).mean()
        push = (Ew1*bij*Fun.relu(margin-dij)).mean()
        loss_contrast = pull + push

        # 原型一致（labeled + 高置信 unlabeled）
        with th.no_grad():
            conf = probs.max(dim=1).values
            mask_high = conf > 0.9
            pseudo = probs.argmax(dim=1)
            use_idx = th.cat([lid, th.where(mask_high & ~th.tensor(label_mask, device=device))[0]])
            use_lab = th.cat([Yt[lid], pseudo[mask_high & ~th.tensor(label_mask, device=device)]])
            # 类原型
            protos = []
            for c in range(K):
                idxc = use_idx[use_lab==c]
                if len(idxc)>0:
                    protos.append(h[idxc].mean(0, keepdim=True))
                else:
                    protos.append(th.zeros_like(h[:1]))
            P = th.cat(protos, dim=0) # [K,H]
        d2p = ((h[:,None,:]-P[None,:,:])**2).sum(-1)  # [M,K]
        loss_proto = d2p.gather(1, probs.argmax(1)[:,None]).mean()

        # 未标注熵
        loss_ent = -(probs[uid] * (probs[uid]+1e-12).log()).sum(dim=1).mean()

        # 一致性（两个视图）
        with th.no_grad():
            logits_t, _ = teacher.node_forward(X, Ei2[0], Ew2)
            q = Fun.softmax(logits_t, dim=1)
        p = Fun.softmax(logits, dim=1)
        loss_cons = Fun.mse_loss(p, q)

        loss = loss_sup + smooth_lambda*loss_gl + ent_lambda*loss_ent \
               + contrast_lambda*loss_contrast + proto_lambda*loss_proto \
               + cons_lambda*loss_cons

        loss.backward(); opt.step()
        if use_ema: update_ema(model, teacher, ema_decay)

        with th.no_grad():
            acc = (probs[lid].argmax(1)==Yt[lid]).float().mean()
        pbar.set_postfix({'Loss': f'{loss.item():.4f}','LAcc': f'{acc.item():.4f}'})
    pbar.close()

    # 推理与校准温度
    model.eval()
    with th.no_grad():
        logits, h, bij = model(X, Ei, Ew, return_embedding=True, edge_feat=True)
    T = temperature_scale(logits, Y, label_mask, init_T=1.0)
    probs = Fun.softmax(logits/ T, dim=1).cpu().numpy()
    emb   = h.cpu().numpy()
    bij   = bij.cpu().numpy()

    # 可选：MC-dropout不与温度冲突
    var_prob = None

    return probs, emb, bij, T, Y_orders

def crf_refine_graph(probs, edge_index, edge_w, boundary_ij, gamma=3.0, n_iter=5):
    row, col = edge_index.cpu().numpy()
    w_eff = edge_w.cpu().numpy() * (1.0 - boundary_ij)
    M, C = probs.shape
    Q = probs.copy() + 1e-12
    deg = np.zeros(M); np.add.at(deg, row, w_eff); deg=np.maximum(deg,1e-12)
    for _ in range(n_iter):
        msg = np.zeros_like(Q)
        np.add.at(msg, row, (w_eff[:,None]*Q[col]))
        msg = msg/deg[:,None]
        Q = Q * np.exp(gamma * msg)
        Q = Q / (Q.sum(axis=1, keepdims=True)+1e-12)
    return Q

def split_nonadjacent(preds, edge_index, min_size=20):
    import networkx as nx
    G = nx.Graph(); row, col=edge_index.cpu().numpy(); G.add_edges_from(zip(row,col))
    labels = preds.copy(); max_lab = labels.max()
    for k in np.unique(preds):
        nodes = np.where(preds==k)[0]
        if len(nodes)==0: continue
        H = G.subgraph(nodes)
        comps = [list(c) for c in nx.connected_components(H)]
        if len(comps)<=1: continue
        for ci, comp in enumerate(comps):
            if len(comp)<min_size: labels[comp]=-1
            else:
                if ci==0: continue
                max_lab+=1; labels[comp]=max_lab
    return labels

def energy_score(logits, T=1.0):
    z = logits.cpu().numpy() / (T+1e-12)
    m = z.max(axis=1, keepdims=True)
    lse = m + np.log(np.exp(z-m).sum(axis=1, keepdims=True))
    return - (T * lse.squeeze())  # 能量越大越不确定

def mahalanobis_to_prototypes(emb, probs, K):
    # 简化：各类协方差对角（或共享标量），仅用均值
    means=[]
    for c in range(K):
        idx = np.where(probs.argmax(1)==c)[0]
        if len(idx)==0: means.append(np.zeros((emb.shape[1],)))
        else: means.append(emb[idx].mean(0))
    M = np.stack(means,0)
    # L2 代替马氏
    D = ((emb[:,None,:]-M[None,:,:])**2).sum(-1)**0.5  # [M,K]
    return D.min(axis=1)  # 距最近原型距离

def discover_novel(emb, logits, probs, T, min_cluster_size=30, tau_prob=0.5,
                   tau_energy=None, tau_proto=None, start_label=None):
    K = probs.shape[1]
    E = energy_score(logits, T)
    Dmin = mahalanobis_to_prototypes(emb, probs, K)
    maxp = probs.max(1)
    if tau_energy is None: tau_energy = np.percentile(E, 80)
    if tau_proto  is None: tau_proto  = np.percentile(Dmin, 80)
    unknown = (maxp < tau_prob) | (E > tau_energy) | (Dmin > tau_proto)
    idxU = np.where(unknown)[0]

    new_labels = np.full(probs.shape[0], -1, dtype=int)
    if len(idxU)==0: return new_labels

    try:
        import hdbscan
        clu = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(emb[idxU])
    except Exception:
        from sklearn.cluster import AgglomerativeClustering
        Kc = max(2, min(10, len(idxU)//min_cluster_size))
        clu = AgglomerativeClustering(n_clusters=Kc).fit_predict(emb[idxU])

    start = (start_label if start_label is not None else (probs.argmax(1).max()+1))
    cur = start
    for c in np.unique(clu):
        if c<0: continue
        ii = idxU[clu==c]
        if len(ii) >= min_cluster_size:
            new_labels[ii] = cur; cur+=1
    return new_labels

def self_training_once(build_data_fn, X, F, labels, mask,
                       knn_kwargs, train_kwargs, crf_kwargs,
                       split_kwargs, novel_kwargs):
    data = build_data_fn(X, F, **knn_kwargs)
    probs, emb, bij, T, Y_orders = train_bagcn_full(data, labels, mask, **train_kwargs)
    # CRF
    probs_ref = crf_refine_graph(probs, data.edge_index, data.edge_attr, bij, **crf_kwargs)
    preds_ref = probs_ref.argmax(1)
    # 拆分
    preds_split = split_nonadjacent(preds_ref, data.edge_index, **split_kwargs)
    # 新类
    logits = th.log(th.tensor(probs_ref+1e-12))
    novel = discover_novel(emb, logits, probs_ref, T, start_label=preds_split.max()+1, **novel_kwargs)
    preds_final = preds_split.copy()
    m = (novel>=0); preds_final[m]=novel[m]
    return preds_final, probs_ref, emb, bij, T, Y_orders

def self_training_loop(X, F, C, label_mask,
                       rounds=1, tau_high=0.9,
                       knn_kwargs=None, train_kwargs=None,
                       crf_kwargs=None, split_kwargs=None, novel_kwargs=None):
    labels = C.copy(); mask = label_mask.copy()
    for _ in range(max(1, rounds)):
        preds, probs, emb, bij, T, _ = self_training_once(
            fused_knnGraph, X, F, labels, mask,
            knn_kwargs, train_kwargs, crf_kwargs, split_kwargs, novel_kwargs
        )
        maxp = probs.max(1)
        # 高置信 + 低能量 + 近原型（下界）
        logits = th.log(th.tensor(probs+1e-12))
        E = energy_score(logits, T); e_thr = np.percentile(E, 60)
        Dmin = mahalanobis_to_prototypes(emb, probs, probs.shape[1]); d_thr = np.percentile(Dmin, 60)
        good = (maxp >= tau_high) & (E <= e_thr) & (Dmin <= d_thr) & (~mask)
        labels[good] = preds[good]; mask[good]=True
    return preds, probs, emb, bij, labels, mask
