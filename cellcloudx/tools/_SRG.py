# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import random
from typing import List, Optional, Literal, Dict, Tuple, Union, Any

import numpy as np
import pandas as pd
import scipy.sparse as ssp

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm


ModeType = Literal["poisson", "nb", "gaussian", "t"]
UGenType  = Literal["gnn", "residual"]
BType     = Literal["full", "lowrank"]


# =========================
# base
# =========================
class thbase():
    def __init__(self, device=None, dtype=None, eps: float = 1e-8, seed: int = 200504):
        self.device = (th.device("cuda" if th.cuda.is_available() else "cpu")
                       if device is None else th.device(device))
        self.dtype = th.float32 if dtype is None else dtype
        self.eps = float(eps)
        self.seed = int(seed)
        self.seed_torch(self.seed)

    def is_sparse(self, X):
        if ssp.issparse(X):
            return True, "scipy"
        if th.is_tensor(X):
            return bool(X.is_sparse), "torch"
        return False, "numpy"

    def to_tensor(self, X, dtype=None, device=None, todense: bool = False, requires_grad: bool = False):
        dtype = self.dtype if dtype is None else dtype
        device = self.device if device is None else th.device(device)

        if th.is_tensor(X):
            T = X.clone()
        elif ssp.issparse(X):
            T = self.spsparse_to_thsparse(X)
            if todense:
                T = T.to_dense()
        else:
            try:
                T = th.as_tensor(X, dtype=dtype)
            except Exception as e:
                raise ValueError(f"{type(X)} cannot be converted to tensor: {e}")

        T = T.to(device=device, dtype=dtype).clone()
        T.requires_grad_(requires_grad)
        return T

    def spsparse_to_thsparse(self, X: ssp.spmatrix) -> th.Tensor:
        XX = X.tocoo()
        indices = np.vstack((XX.row, XX.col)).astype(np.int64)
        values = XX.data
        i = th.from_numpy(indices)
        v = th.as_tensor(values, dtype=th.float32)
        shape = th.Size(XX.shape)
        return th.sparse_coo_tensor(i, v, shape)

    def thsparse_to_spsparse(self, X):
        """Convert a torch sparse COO/CSR tensor to a SciPy CSR array."""
        XX = X.to_sparse_coo().coalesce()
        values = XX.values().detach().cpu().numpy()
        indices = XX.indices().detach().cpu().numpy()
        row, col = indices[0], indices[1]
        shape = XX.shape
        return ssp.csr_array((values, (row, col)), shape=shape)

    def thsparse_slice(self, X, idx, axis):
        if th.is_tensor(idx):
            idx = idx.cpu().numpy()
 
        A = self.thsparse_to_spsparse(X)
        if axis == 0:
            B = A[idx,:]
        elif axis == 1:
            B = A[:,idx]
        else:
            raise ValueError(f"axis={axis} is not supported")
        B = self.spsparse_to_thsparse(B)
        return B.to(X.device)

    def seed_torch(self, seed: int = 200504):
        if seed is None:
            return
        seed = int(seed)
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        th.manual_seed(seed)
        if th.cuda.is_available():
            th.cuda.manual_seed(seed)
            th.cuda.manual_seed_all(seed)
            th.backends.cudnn.deterministic = True
            th.backends.cudnn.benchmark = False
        if hasattr(th.backends, "mps") and th.backends.mps.is_available():
            th.mps.manual_seed(seed)

# =========================
# main SRG
# =========================

class SRG(thbase):
    def __init__(self,
                 Exp: Union[np.ndarray, th.Tensor],
                 Cor: Union[np.ndarray, th.Tensor],
                 BI: Union[np.ndarray, th.Tensor],
                 BI_gene: Dict[str, List[str]],
                 C_gene: Union[List[str], np.ndarray, th.Tensor],

                 mode: Union[ModeType, str] = "poisson",
                 log_link: str = "log1p",
                 t_df: float = 4.0,
                 device: str = "cuda" if th.cuda.is_available() else "cpu",
                 normal_Cor = True,
                 dtype=th.float32,
                 verbose: int = 1,
                 eps: float = 1e-8,
                 seed: int = 0):
        super().__init__(device=device, dtype=dtype, eps=eps, seed=seed)

        if mode not in ("poisson", "nb", "gaussian", "t"):
            raise ValueError(f"Unknown mode={mode}")

        self.Exp = self.to_tensor(Exp, dtype=self.dtype, device="cpu")
        self.Cor = self.to_tensor(Cor, dtype=self.dtype, device="cpu")
        if normal_Cor:
            self.Cor = normal_scaler(self.Cor)

        self.BI = BI
        # self.BI_order = np.unique(BI)
        # self.BIv = self.to_tensor(label_to_order(self.BI, order=self.BI_order), dtype=th.int64, device=self.device)
        self.BI_order, self.BIv = np.unique(self.BI, return_inverse=True)
        self.BIv = self.to_tensor(self.BIv, dtype=th.int64, device=self.device)

        self.nB = int(self.BIv.max().item() + 1)
        self.N = int(self.Exp.shape[0])
        self.G = int(self.Exp.shape[1])

        self.C_gene = C_gene
        self.geneid = th.arange(self.G, device=self.device, dtype=th.int64)
        self.genedict = {g: i for i, g in enumerate(self.C_gene)}
        assert len(set(C_gene)) == self.G

        self.BI_gene = BI_gene
        self.BI_mask = th.zeros((self.nB, self.G), dtype=self.dtype, device=self.device)
        for i, ib in enumerate(self.BI_order):
            ibg = self.BI_gene[ib]
            for g in ibg:
                self.BI_mask[i, self.genedict[g]] = 1.0

        self.Lib = self.Exp.sum(dim=1)
        if self.is_sparse(self.Exp)[0]:
            self.Lib = self.Lib.to_dense()
        self.Lib = self.Lib.to(self.device, dtype=self.dtype) + 1.0

        self.mode: ModeType = mode  # type: ignore
        self.log_link = str(log_link)
        self.t_df = float(t_df)

        self.device = device
        self.dtype = dtype
        self.verbose = verbose
        self.seed = int(seed)

        self.src: Optional[th.Tensor] = None
        self.dst: Optional[th.Tensor] = None
        self.stagea: Dict[str, Any] = {}
        self.stageb: Dict[str, Any] = {}

    # ---------- graph builder ----------
    def KnnGraph(self, knn=None, radius=None, knn_method='global', BIv=None,  ks=1, 
                 keep_loops=False,  kd_method='annoy', CI=0.98, BI_order=None,):
        assert knn_method in ['ovo', 'ovr', 'global', 'pair']
        Cor = self.Cor.detach().cpu().numpy()

        if knn_method == 'global': # not symmetric
            src, dst, dist = coord_edges(Cor, Cor, knn = knn, radius=radius, method=kd_method)
        elif knn_method in ['ovo']: # not symmetric
            BIv = self.BIv.detach().cpu().numpy() if BIv is None else BIv
            bis = np.unique(BIv)
            nB = len(bis)
            knn = scalar2array(knn, (nB, nB))
            radius = scalar2array(radius, (nB, nB))
            src, dst, dist = [], [], []
    
            pbar = tqdm(range(nB), total=nB, desc=f'KnnGraph ovo')
            for i in pbar:
                ib1 = bis[i]
                for j, ib2 in enumerate(bis):
                    bid1 = np.where(BIv == ib1)[0]
                    bid2 = np.where(BIv == ib2)[0]
                    iknn = int(knn[i, j]) or None
                    iradius = radius[i, j] or None

                    if iknn is None and iradius is None:
                        continue

                    isrc, idst, idist = coord_edges(Cor[bid1], Cor[bid2], knn = iknn, 
                                                radius=iradius, method=kd_method)
                    isrc = bid1[isrc]
                    idst = bid2[idst]
                    src.append(isrc)
                    dst.append(idst)
                    dist.append(idist)
            src = np.concatenate(src)
            dst = np.concatenate(dst)
            dist = np.concatenate(dist)

        elif knn_method in ['ovr']:
            BIv = self.BIv.detach().cpu().numpy() if BIv is None else BIv
            bis = np.unique(BIv)
            nB = len(bis)
            knn = scalar2array(knn, (nB, 2))
            radius = scalar2array(radius, (nB, 2))
            src, dst, dist = [], [], []
    
            pbar = tqdm(range(nB), total=nB, desc=f'KnnGraph ovr')
            for i in pbar:
                ib1 = bis[i]
                bid1 = np.where(BIv == ib1)[0]
                bid2 = np.where(BIv != ib1)[0]
                iknn0 = int(knn[i, 0]) or None
                iradius0 = radius[i, 0] or None
                iknn1 = int(knn[i, 1]) or None
                iradius1 = radius[i, 1] or None
                
                isrc0, idst0, idist0 = coord_edges(Cor[bid1], knn = iknn0, 
                                            radius=iradius0, method=kd_method)
                isrc1, idst1, idist1 = coord_edges(Cor[bid1], Cor[bid2], knn = iknn1,
                                            radius=iradius1, method=kd_method)
                isrc0 = bid1[isrc0]
                idst0 = bid1[idst0]
                isrc1 = bid1[isrc1]
                idst1 = bid2[idst1]
                src += [isrc0, isrc1]
                dst += [idst0, idst1]
                dist += [idist0, idist1]
                
            src = np.concatenate(src)
            dst = np.concatenate(dst)
            dist = np.concatenate(dist)

        elif knn_method in ['pair']:
            assert BI_order is not None, "BI_order is required for 'pair' knn_method."
            BI_order = np.array(BI_order)
            BI = self.BI
            assert len( set(BI_order) - set(BI) ) ==0, "BI_order must match BI labels."
            nB = self.nB
            assert len(BI_order) == nB, f"BI_order length {len(BI_order)} != nB {nB}"
    
            knn = scalar2array(knn, (nB, 2))
            radius = scalar2array(radius, (nB, 2))
            src, dst, dist = [], [], []
            pbar = tqdm(range(nB), total=nB, desc=f'KnnGraph pair')
            for i in pbar:
                for j in range(i, min(i+ks+1, nB)):
                    bid1 = np.where(BI == BI_order[i])[0]
                    bid2 = np.where(BI == BI_order[j])[0]
                    if i == j:
                        iknn = int(knn[i, 0]) or None
                        iradius = radius[i, 0] or None
                        isrc, idst, idist = coord_edges(Cor[bid1], Cor[bid2], knn = iknn, 
                                                    radius=iradius, method=kd_method)
                        isrc = bid1[isrc]
                        idst = bid2[idst]
                    else:
                        iknn = int(knn[i, 1]) or None
                        iradius = radius[i, 1] or None
                        if iknn is None and iradius is None:
                            continue

                        isrc1, idst1, idist1 = coord_edges(Cor[bid1], Cor[bid2], knn = iknn, 
                                                    radius=iradius, method=kd_method)

                        isrc2, idst2, idist2 = coord_edges(Cor[bid2], Cor[bid1], knn = iknn, 
                                                    radius=iradius, method=kd_method)
                        isrc = np.concatenate([bid1[isrc1], bid2[isrc2]])
                        idst = np.concatenate([bid2[idst1], bid1[idst2]])
                        idist = np.concatenate([idist1, idist2])
                    src.append(isrc)
                    dst.append(idst)
                    dist.append(idist)
            src = np.concatenate(src)
            dst = np.concatenate(dst)
            dist = np.concatenate(dist)

        if CI <1.0:
            K  = int(len(dist)*CI)
            kidx = np.argpartition(dist, K,)[:K]
            src = src[kidx]
            dst = dst[kidx]
            dist = dist[kidx]
        
        if not keep_loops:
            mask = src != dst
            src = src[mask]
            dst = dst[mask]
            dist = dist[mask]

        if self.verbose:
            print(f'KnnGraph: {Cor.shape[0]} nodes, {len(src)} edges')
        src = self.to_tensor(src, dtype=th.int64, device=self.device) # src->dst
        dst = self.to_tensor(dst, dtype=th.int64, device=self.device) # dst->src
        self.src = dst
        self.dst = src
        return dst, src

    # def edge_weight(self, dist: Optional[np.ndarray] = None, sigma: Optional[float] = None):

    # ---------- training ----------

    def train(self, 
                u_enhence: bool = False,
                use_expw: bool = False,
                sigma: Optional[float] = None,
                tau: Optional[float] = None,
                B_mode: BType = "lowrank",

                u_gen: UGenType = "gnn",
                K: int = 64, R: int = 32,
                h_dim: int = 64,
                gnn_layers: int = 2,
                lambda_graph: float = 1e-2,
                use_adaptive_lambda: bool = False,
                use_edge_gate: bool = True,
                edge_gate_chunk: Optional[int] = None,
                use_node_reliability: bool = False,

                device = None,
                dtype = None,

                epochs: int = 500,
                lr: float = 2e-2,
                weight_decay: float = 5e-3,
                reg_l2: float = 0) -> Dict[str, Any]:
        

        if self.src is None or self.dst is None:
            raise RuntimeError("Call KnnGraph first to set self.src/self.dst.")

        device = self.device if device is None else device
        dtype = self.dtype if dtype is None else dtype
        th.manual_seed(self.seed)
        np.random.seed(self.seed)

        Exp = self.Exp.to_dense() if self.is_sparse(self.Exp)[0] else self.Exp
        Exp = Exp.to(device, dtype=dtype)
        Cor = self.Cor.to(device, dtype=dtype)
        Lib = self.Lib.to(device, dtype=dtype)
        BIv = self.BIv.to(device).long()
        BI_mask  = self.BI_mask.to(device, dtype=dtype)
        BIv_mask = BI_mask[BIv]

        nB  = self.nB
        mode = self.mode
        n, c = Exp.shape


        src, dst = self.src.to(device), self.dst.to(device)
        dist = th.norm(Cor[src] - Cor[dst], dim=1).to(device, dtype=dtype)
        sigma = dist.mean().item() if sigma is None else sigma
        print(sigma)
        dist /= sigma

        import matplotlib.pyplot as plt
        plt.hist(dist.cpu().numpy(), bins=100)
        plt.show()

        if use_expw:
            Expn = normalize_l2(Exp, eps=self.eps)
            diste = th.norm(Expn[src] - Expn[dst], dim=1).to(device, dtype=dtype)
            del Expn
            tau = diste.mean().item() if tau is None else tau
            diste /= tau

            base_w = th.exp(- (dist ** 2) - (diste ** 2))
            if self.verbose:
                print(f"[StageA] n={n} g={c} slices={self.nB} common={c} sigma={sigma:.3e} tau={tau:.3e}")
        else:
            base_w = th.exp(- (dist ** 2) )
            if self.verbose:
                print(f"[StageA] n={n} g={c} slices={self.nB} common={c} sigma={sigma:.3e}")

        model = StageA_GNN(
            n=n, c=c, n_slices=nB, K=K, R=R,
            mode=mode, u_gen=u_gen, gnn_layers=gnn_layers,
            h_dim=h_dim, use_edge_gate=use_edge_gate,
            use_node_reliability=use_node_reliability,
            edge_gate_chunk=edge_gate_chunk
        ).to(device)

        if weight_decay >0:
            opt = th.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay) 
        else:
            opt = th.optim.Adam(model.parameters(), lr=lr)
        # scheduler = th.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)

        pbar = tqdm(range(epochs), total=epochs,
                    desc=("stageA_enhance 1111" if u_enhence else "stageA_simple"),
                    colour="red", disable=(self.verbose < 1))

        for ep in pbar:
            U, h, w = model.compute_U(Exp, src, dst, base_w)    
            m, extra = model.forward_common(U, Lib, BIv, batch_correct=False,
                                                    log_link=self.log_link, eps=self.eps)
            loss_graph = self.charbonnier_graph_loss(
                U, src, dst, w, eps=1e-3,
            )
            loss_data = self.masked_mode_loss(Exp, m, BIv_mask, sigma=extra, 
                                             df=self.t_df, mode=mode, eps=self.eps)

            if use_adaptive_lambda:
                effective_lambda = self.adaptive_loss_weight(loss_data, loss_graph, 
                                                             target_ratio=lambda_graph)
            else:
                effective_lambda = lambda_graph
            loss_graph = float(effective_lambda) * loss_graph


            loss_reg = th.tensor(0.0, device=device)
            if reg_l2 > 0:
                for p in model.parameters():
                    loss_reg = loss_reg + p.pow(2).mean()
                loss_reg = reg_l2 * loss_reg

            loss = loss_data + loss_graph + loss_reg

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            msg = {"loss_data": f'{loss_data.item():.3e}', 
                   "loss_graph": f'{loss_graph.item():.3e}'}
            if reg_l2 > 0:
                msg["loss_reg"] = f'{loss_reg.item():.3e}'
            pbar.set_postfix(msg)

        pbar.close()

        with th.no_grad():
            if u_enhence:
                U, h, w = model.compute_U(Exp, src, dst, base_w)
                Wc = model.W_pos()
                stagea = {
                    "U": U.detach().cpu(),
                    "Wc": Wc.detach().cpu(),
                    "alphac": model.alpha.detach().cpu(),
                    "P": model.P.detach().cpu(),
                    "Qc": model.Q.detach().cpu(),
                }
                if mode == "nb":
                    stagea["thetac"] = model.theta_pos().detach().cpu()
                if mode in ("gaussian", "t"):
                    stagea["sigmac"] = model.sigma_pos().detach().cpu()
                if use_node_reliability:
                    stagea["rho"] = model.rho().detach().cpu()
            else:
                if B_mode == "full":
                    stagea = {
                        "U": model.U_pos().detach().cpu(),
                        "Wc": model.W_pos().detach().cpu(),
                        "alphac": model.alpha.detach().cpu(),
                        "B": model.B.detach().cpu(),
                    }
                else:
                    stagea = {
                        "U": model.U_pos().detach().cpu(),
                        "Wc": model.W_pos().detach().cpu(),
                        "alphac": model.alpha.detach().cpu(),
                        "P": model.P.weight.detach().cpu(),
                        "Qc": model.Q.weight.detach().cpu(),
                    }
                if mode == "nb":
                    stagea["thetac"] = model.theta_pos().detach().cpu()
                if mode in ("gaussian", "t"):
                    stagea["sigmac"] = model.sigma_pos().detach().cpu()
                if use_node_reliability:
                    stagea["rho"] = model.rho().detach().cpu()

        self.stagea = stagea
        return stagea
    
    @th.no_grad()
    def prediction_common(self, block_size: Optional[int] = None, device: str = "cpu"):
        """
        Returns:
          pred_corr: batch-removed mean in OBS space
          pred_raw : slice-specific mean in OBS space
        OBS space:
          - poisson/nb: counts mean mu
          - gaussian/t: mean in log space (log1p(mu) or log(mu))
        """
        if not self.stagea:
            raise RuntimeError("Run rbe_train first.")
        dev = th.device(device)

        Lib = self.Lib.to(dev, dtype=self.dtype)
        BIv = self.BIv.to(dev).long()

        U = self.stagea["U"].to(dev, dtype=self.dtype)
        Wc = self.stagea["Wc"].to(dev, dtype=self.dtype)
        alphac = self.stagea["alphac"].to(dev, dtype=self.dtype)

        if "B" in self.stagea:
            B = self.stagea["B"].to(dev, dtype=self.dtype)
            Bs = B - B.mean(dim=0, keepdim=True)
        else:
            P = self.stagea["P"].to(dev, dtype=self.dtype)
            Qc = self.stagea["Qc"].to(dev, dtype=self.dtype)
            Pm = P.mean(dim=0, keepdim=True)
            Bs = (P - Pm) @ Qc.t()

        def _eta_from(Ub, libb, sidb):
            dot = (Ub @ Wc.t()).clamp_min(self.eps)
            eta = th.log(libb.clamp_min(self.eps))[:, None] + alphac[None, :] + th.log(dot)
            eta_raw = eta + Bs[sidb]
            return eta, eta_raw

        if block_size is None:
            eta_corr, eta_raw = _eta_from(U, Lib, BIv)
            pred_corr = self._eta_to_obs_mean(eta_corr).detach().cpu()
            pred_raw = self._eta_to_obs_mean(eta_raw).detach().cpu()
            return pred_corr, pred_raw
        else:
            n, c = U.shape[0], Wc.shape[0]
            pred_corr = th.empty((n, c), device="cpu", dtype=th.float32)
            pred_raw = th.empty((n, c), device="cpu", dtype=th.float32)
            for st in range(0, n, int(block_size)):
                ed = min(st + int(block_size), n)
                eta_corr, eta_raw = _eta_from(U[st:ed], Lib[st:ed], BIv[st:ed])
                pred_corr[st:ed] = self._eta_to_obs_mean(eta_corr).detach().cpu()
                pred_raw[st:ed] = self._eta_to_obs_mean(eta_raw).detach().cpu()
            return pred_corr, pred_raw
        
    def mode_loss(self,  y: th.Tensor, m: th.Tensor,
                   sigma: th.Tensor = None, df: float = 4.0, mode: str = "poisson",
                   eps: float = 1e-8):
            
        if mode == "poisson":
            loss_data = self.poisson_log_null(y, m, eps=self.eps).mean()
        elif mode == "nb":
            loss_data = (self.nb_log_prob(y, m, sigma, eps=self.eps)).mean()
        elif mode == "gaussian":

            loss_data = self.gaussian_nll(y, m, sigma, eps=self.eps).mean()
        elif mode == "t":
            loss_data = self.studentt_nll(y, m, sigma, df=self.t_df, eps=self.eps).mean()
        else:
            raise ValueError(mode)
        return loss_data

    def masked_mode_loss(self,  y: th.Tensor, m: th.Tensor, mask: th.Tensor,
                   sigma: th.Tensor = None, df: float = 4.0, mode: str = "poisson",
                   eps: float = 1e-8):
            
        if mode == "poisson":
            loss_data = self.masked_poisson_nll(y, m, mask, eps=self.eps)
        elif mode == "nb":
            loss_data = (self.masked_nb_nll(y, m, sigma, mask, eps=self.eps))
        elif mode == "gaussian":
            loss_data = self.gaussian_nll(y, m, sigma, eps=self.eps)
            loss_data = (loss_data * mask).sum() / (mask.sum() + self.eps)
        elif mode == "t":
            loss_data = self.studentt_nll(y, m, sigma, df=self.t_df, eps=self.eps)
            loss_data = (loss_data * mask).sum() / (mask.sum() + self.eps)
        else:
            raise ValueError(mode)
        return loss_data

    # ---------- distributions ----------
    def poisson_log_null(self, y: th.Tensor, mu: th.Tensor, eps: float = 1e-8) -> th.Tensor:
        # return th.exp(mu) - y * mu
        return mu - y * th.log(mu.clamp_min(eps)) # log

    def nb_log_prob(self, y: th.Tensor, mu: th.Tensor, theta: th.Tensor, eps: float = 1e-8) -> th.Tensor:
        t1 = th.lgamma(y + theta) - th.lgamma(theta) - th.lgamma(y + 1.0)
        t2 = theta * (th.log(theta + eps) - th.log(theta + mu + eps))
        t3 = y * (th.log(mu + eps) - th.log(theta + mu + eps))
        return -(t1 + t2 + t3)

    def gaussian_nll(self, y: th.Tensor, m: th.Tensor, sigma: th.Tensor, eps: float = 1e-8) -> th.Tensor:
        s = sigma.clamp_min(eps)
        z = (y - m) / s
        return th.log(s) + 0.5 * z * z

    def studentt_nll(self, y: th.Tensor, m: th.Tensor, sigma: th.Tensor, df: float, eps: float = 1e-8) -> th.Tensor:
        v = th.tensor(float(df), device=y.device, dtype=y.dtype)
        s = sigma.clamp_min(eps)
        z2 = ((y - m) / s).pow(2)
        logC = th.lgamma((v + 1.0) / 2.0) - th.lgamma(v / 2.0) - 0.5 * th.log(v * th.tensor(np.pi, device=y.device, dtype=y.dtype)) - th.log(s)
        logK = - (v + 1.0) / 2.0 * th.log1p(z2 / v)
        return -(logC + logK)

    def masked_poisson_nll(self,
        y: th.Tensor,
        mu: th.Tensor,
        mask: th.Tensor,
        neg_rate: float = 1.0,
        eps: float = 1e-8,
    ) -> th.Tensor:
        """
        NLL (ignore const): mu - y*log(mu)
        mask indicates observed entries.
        Optional negative sampling for (y==0) terms:
        - keep all nonzero observed
        - sample zeros with prob=neg_rate, scale mu term by 1/neg_rate
        """
        mask = mask.to(mu.dtype)

        # nonzero observed
        nz = (y > 0) & (mask > 0)
        z0 = (y <= 0) & (mask > 0)

        loss = th.tensor(0.0, device=mu.device, dtype=mu.dtype)
        denom = th.tensor(0.0, device=mu.device, dtype=mu.dtype)

        if nz.any():
            y_nz = y[nz]
            mu_nz = mu[nz]
            # mu - y log mu
            loss = loss + (mu_nz - y_nz * th.log(mu_nz.clamp_min(eps))).sum()
            denom = denom + nz.sum().to(mu.dtype)

        if z0.any():
            mu0 = mu[z0]
            if neg_rate >= 1.0:
                loss = loss + mu0.sum()
                denom = denom + z0.sum().to(mu.dtype)
            else:
                # sample zeros
                keep = (th.rand_like(mu0) < float(neg_rate))
                if keep.any():
                    loss = loss + (mu0[keep].sum() / float(neg_rate))
                    denom = denom + (keep.sum().to(mu.dtype) / float(neg_rate))

        return loss / denom.clamp_min(1.0)

    def masked_nb_nll(self,
        y: th.Tensor,
        mu: th.Tensor,
        theta: th.Tensor,
        mask: th.Tensor,
        neg_rate: float = 1.0,
        eps: float = 1e-8,
    ) -> th.Tensor:
        mask = mask.to(mu.dtype)
        
        nz = (y > 0) & (mask > 0) 
        z0 = (y <= 0) & (mask > 0)
        
        loss = th.tensor(0.0, device=mu.device, dtype=mu.dtype)
        denom = th.tensor(0.0, device=mu.device, dtype=mu.dtype)
        
        if nz.any():
            y_nz = y[nz]
            mu_nz = mu[nz]
            theta_nz = theta[nz]
            
            log_theta_mu = th.log(theta_nz + mu_nz + eps)
            t1 = th.lgamma(y_nz + theta_nz) - th.lgamma(theta_nz) - th.lgamma(y_nz + 1.0)
            t2 = theta_nz * (th.log(theta_nz + eps) - log_theta_mu)
            t3 = y_nz * (th.log(mu_nz + eps) - log_theta_mu)
            loss_nz = -(t1 + t2 + t3)
            
            loss = loss + loss_nz.sum()
            denom = denom + nz.sum().to(mu.dtype)
        
        if z0.any():
            mu0 = mu[z0]
            theta0 = theta[z0]
            
            if neg_rate >= 1.0:
                log_term = th.log(theta0 + mu0 + eps) - th.log(theta0 + eps)
                loss_0 = theta0 * log_term
                
                loss = loss + loss_0.sum()
                denom = denom + z0.sum().to(mu.dtype)
            else:
                keep = (th.rand_like(mu0) < float(neg_rate))
                if keep.any():
                    mu0_sampled = mu0[keep]
                    theta0_sampled = theta0[keep]

                    log_term_s = th.log(theta0_sampled + mu0_sampled + eps) - th.log(theta0_sampled + eps)
                    sampled_loss = theta0_sampled * log_term_s

                    loss = loss + (sampled_loss.sum() / float(neg_rate))
                    denom = denom + (keep.sum().to(mu.dtype) / float(neg_rate))
        
        return loss / denom.clamp_min(1.0)

    def _eta_to_obs_mean(self, eta: th.Tensor) -> th.Tensor:
        """
        eta = log(mu_counts).
        Returns mean in observed space:
          - poisson/nb: mu_counts = exp(eta)
          - gaussian/t:
              log1p: mean = log(1+exp(eta)) = softplus(eta)
              log:   mean = eta
        """
        if self.mode in ("poisson", "nb"):
            return th.exp(eta.clamp(-20, 20))
        if self.log_link == "log1p":
            return F.softplus(eta)
        if self.log_link == "log":
            return eta
        raise ValueError(f"Unknown log_link={self.log_link}")

    # ---------- graph loss ----------
    def charbonnier_graph_loss0(self, U: th.Tensor, h: Optional[th.Tensor],
                              src: th.Tensor, dst: th.Tensor,
                              base_w: th.Tensor,
                              edge_gate: Optional[EdgeGate] = None,
                              rho: Optional[th.Tensor] = None,
                              eps: float = 1e-3,
                              edge_gate_chunk: Optional[int] = None) -> th.Tensor:
        if edge_gate is None or h is None:
            w = base_w
        else:
            h_src = h[src]
            h_dst = h[dst]
            gate = edge_gate(h_src, h_dst, chunk_size=edge_gate_chunk)
            # w = base_w * gate
            # w = base_w.detach() * gate  # base_w is constant; keep grad for gate
            w = base_w * gate.detach()  # stop gradient through base_w
        if rho is not None:
            w = w * rho[src] * rho[dst]
        diff2 = (U[src] - U[dst]).pow(2).sum(dim=1)
        # diff2 = (U[src] - U[dst]).pow(2).mean(dim=1)
        val = w * th.sqrt(diff2 + eps*eps) 
        # return val.mean()
        n = U.shape[0]
        deg = th.zeros((n,), device=U.device, dtype=U.dtype)
        deg.index_add_(0, dst, w)                          # weighted degree
        acc = th.zeros((n,), device=U.device, dtype=U.dtype)
        acc.index_add_(0, dst, val)
        return (acc / (deg + 1e-8)).mean()  

    def charbonnier_graph_loss(self, U: th.Tensor,
                            src: th.Tensor, dst: th.Tensor,
                            w: th.Tensor,
                            eps: float = 1e-3,
                            ) -> th.Tensor:

        U_norm = F.normalize(U, p=2, dim=1)  # L2 归一化
        diff2 = (U_norm[src] - U_norm[dst]).pow(2).sum(dim=1)
        
        val = w * th.sqrt(diff2 + eps*eps)
        
        n = U.shape[0]
        deg = th.zeros((n,), device=U.device, dtype=U.dtype)
        deg.index_add_(0, dst, w)
        acc = th.zeros((n,), device=U.device, dtype=U.dtype)
        acc.index_add_(0, dst, val)
        
        return (acc / (deg.clamp_min(1e-8))).mean()

    def adaptive_loss_weight(self, 
                              loss_data: th.Tensor, 
                              loss_graph: th.Tensor,
                              target_ratio: float = 0.1,
                              momentum: float = 0.9) -> th.Tensor:
        """
        自适应调整 graph loss 权重
        
        目标: 使 lambda * loss_graph ≈ target_ratio * loss_data
        """
        if not hasattr(self, '_running_ratio'):
            self._running_ratio = 1.0
        
        with th.no_grad():
            current_ratio = (loss_data / (loss_graph + 1e-8)).item()
            # EMA 平滑
            self._running_ratio = momentum * self._running_ratio + (1 - momentum) * current_ratio
            
            # 计算自适应 lambda
            adaptive_lambda = target_ratio * self._running_ratio
            adaptive_lambda = max(1e-4, min(adaptive_lambda, 10.0))  # 限制范围
        
        return adaptive_lambda
# =========================
# model pieces
# =========================

class EdgeGate(nn.Module):
    """
    [h_src, h_dst, dist] -> sigmoid gate in (0,1)
    Supports chunked forward to save memory.
    """
    def __init__(self, dim_h: int, hidden: int = 64,):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim_h * 2 , hidden), 
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias, 2.0)  # sigmoid(2) ≈ 0.88

    def forward(self, h_src: th.Tensor, h_dst: th.Tensor,
                chunk_size: Optional[int] = None) -> th.Tensor:
        E = h_src.shape[0]
        if chunk_size is None or E <= chunk_size:
            x = th.cat([h_src, h_dst], dim=1)
            return th.sigmoid(self.mlp(x)).squeeze(-1)

        outs = []
        for st in range(0, E, int(chunk_size)):
            ed = min(st + int(chunk_size), E)
            x = th.cat([h_src[st:ed], h_dst[st:ed]], dim=1)
            outs.append(th.sigmoid(self.mlp(x)).squeeze(-1))
        return th.cat(outs, dim=0)

class EdgeGate0(nn.Module):
    """
    [h_src, h_dst, dist] -> sigmoid gate in (0,1)
    Supports chunked forward to save memory.
    """
    def __init__(self, dim_h: int, hidden: int = 64, use_dist=False):
        super().__init__()

        self.use_dist = use_dist
        inhid = dim_h * 2 + 1 if use_dist else dim_h * 2
        self.mlp = nn.Sequential(
            nn.Linear(inhid, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, h_src: th.Tensor, h_dst: th.Tensor, dist: th.Tensor,
                chunk_size: Optional[int] = None) -> th.Tensor:
        E = h_src.shape[0]
        if chunk_size is None or E <= chunk_size:
            x = th.cat([h_src, h_dst, dist[:, None]], dim=1)
            return th.sigmoid(self.mlp(x)).squeeze(-1)

        outs = []
        for st in range(0, E, int(chunk_size)):
            ed = min(st + int(chunk_size), E)
            x = th.cat([h_src[st:ed], h_dst[st:ed], dist[st:ed, None]], dim=1)
            outs.append(th.sigmoid(self.mlp(x)).squeeze(-1))
        return th.cat(outs, dim=0)

class StageA_Simple(nn.Module):
    """
    StageA for common genes, with OPTIONAL learned edge-gate from expression.

    eta_{i,g} = log(lib_i) + log(U_i^T W_g) + alpha_g + b_{s(i),g}
    - poisson/nb: mu = exp(eta)
    - gaussian/t : mean in observed log space:
        log1p: m = softplus(eta)
        log  : m = eta

    Batch:
      - full:   b_{s,g} = B_{s,g} - mean(B[:,g])
      - lowrank: b_{s,g} = (P_s - mean(P))^T Q_g

    Graph:
      - Graph loss computed outside, but this module provides:
        encode(x_common)->h ; edge_gate(h_src,h_dst,dist)->gate ; rho()->node reliability
    """
    def __init__(self, n: int, c: int, n_slices: int, K: int = 32, R: int = 16,
                 mode: ModeType = "poisson", B_mode: BType = "lowrank",
                 # NEW: learn edge weights from expression
                 use_edge_gate: bool = False,
                 edge_h_dim: int = 64,
                 edge_hidden: int = 64,
                 edge_gate_chunk: Optional[int] = None,
                 use_node_reliability: bool = False):
        super().__init__()
        self.n, self.c = int(n), int(c)
        self.n_slices = int(n_slices)
        self.K, self.R = int(K), int(R)
        self.mode = mode
        self.B_mode = B_mode

        self.U_raw = nn.Parameter(th.randn(n, K) * 0.01)
        self.W_raw = nn.Parameter(th.randn(c, K) * 0.01)
        self.alpha = nn.Parameter(th.zeros(c))

        if B_mode == "full":
            self.B = nn.Parameter(th.zeros(n_slices, c))
            nn.init.zeros_(self.B)
            self.P = None
            self.Q = None
        elif B_mode == "lowrank":
            self.P = nn.Embedding(n_slices, R)
            self.Q = nn.Embedding(c, R)
            nn.init.normal_(self.P.weight, std=0.02)
            nn.init.normal_(self.Q.weight, std=0.02)
            self.B = None
        else:
            raise ValueError(f"Unknown B_mode={B_mode}")

        if mode == "nb":
            self.theta_raw = nn.Parameter(th.zeros(c))
        if mode in ("gaussian", "t"):
            self.sigma_raw = nn.Parameter(th.zeros(c))

        # NEW: edge gate from expression encoder
        self.use_edge_gate = bool(use_edge_gate)
        self.edge_gate_chunk = edge_gate_chunk
        if self.use_edge_gate:
            self.enc = nn.Sequential(
                nn.Linear(c, edge_h_dim), nn.ReLU(),
                nn.Linear(edge_h_dim, edge_h_dim), nn.ReLU(),
            )
            self.edge_gate = EdgeGate(dim_h=edge_h_dim, hidden=edge_hidden)
        else:
            self.enc = None
            self.edge_gate = None

        self.use_node_reliability = bool(use_node_reliability)
        if self.use_node_reliability:
            self.rho_raw = nn.Parameter(th.zeros(n))

    def U_pos(self): return F.softplus(self.U_raw)
    def W_pos(self): return F.softplus(self.W_raw)
    def theta_pos(self): return F.softplus(self.theta_raw) + 1e-4
    def sigma_pos(self): return F.softplus(self.sigma_raw) + 1e-4

    def rho(self) -> Optional[th.Tensor]:
        if not self.use_node_reliability:
            return None
        return th.sigmoid(self.rho_raw)

    def mp_aggregate(self, n: int, src: th.Tensor, dst: th.Tensor, w: th.Tensor, h: th.Tensor, eps: float = 1e-8):
        try:
            from torch_scatter import scatter_add  # type: ignore
            m = scatter_add(w[:, None] * h[src], dst, dim=0, dim_size=n)
            deg = scatter_add(w, dst, dim=0, dim_size=n)
            return m / (deg[:, None] + eps)
        except Exception:
            dim = h.shape[1]
            m = th.zeros((n, dim), device=h.device, dtype=h.dtype)
            deg = th.zeros((n,), device=h.device, dtype=h.dtype)
            m.index_add_(0, dst, w[:, None] * h[src])
            deg.index_add_(0, dst, w)
            return m / (deg[:, None] + eps)
    
    def gnn_forward(self, x_common: th.Tensor, src: th.Tensor, dst: th.Tensor, base_w: th.Tensor) -> th.Tensor:
        n = x_common.shape[0]
        h = self.enc(x_common)

        rho = self.rho()
        for l in range(self.gnn_layers):
            if self.use_edge_gate:
                h_src = h[src]
                h_dst = h[dst]
                gate = self.edge_gate(h_src, h_dst, chunk_size=self.edge_gate_chunk)
                w = base_w * gate
            else:
                w = base_w
            if rho is not None:
                w = w * rho[src] * rho[dst]

            h = self.mp_aggregate(n, src, dst, w, h)
            # h = self.ln[l](h + self.upd[l](neigh))
        return h

    def _batch_term(self, sid: th.Tensor) -> th.Tensor:
        if self.B_mode == "full":
            Bm = self.B.mean(dim=0, keepdim=True)
            return (self.B[sid] - Bm)
        else:
            Pm = self.P.weight.mean(dim=0, keepdim=True)
            return (self.P(sid) - Pm) @ self.Q.weight.t()


    # lambda_ij = base_w * (lambda0 + lambda1 * gate_ij) * rho_i * rho_j
    # self.lambda0_raw = nn.Parameter(th.tensor(0.0))  # global base precision
    # self.lambda1_raw = nn.Parameter(th.tensor(0.0))  # gate-controlled extra precision

    # def lambda0_pos(self):
    #     return F.softplus(self.lambda0_raw) + 1e-4

    # def lambda1_pos(self):
    #     return F.softplus(self.lambda1_raw) + 1e-4

    # def edge_lambda(self, gate: th.Tensor) -> th.Tensor:
    #     """
    #     gate: (E,) in (0,1)
    #     return lambda_edge: (E,) positive
    #     """
    #     return self.lambda0_pos() + self.lambda1_pos() * gate

    # def adaptive_gmrf_prior(self,
    #                         U: th.Tensor,
    #                         src: th.Tensor, dst: th.Tensor,
    #                         base_w: th.Tensor,
    #                         gate: th.Tensor,
    #                         lambda_edge: th.Tensor,
    #                         rho: Optional[th.Tensor] = None,
    #                         eps: float = 1e-8) -> th.Tensor:
    #     """
    #     Probabilistic prior:
    #     (U_i-U_j) ~ N(0, sigma_U^2 / (base_w * lambda_edge * rho_i*rho_j))
    #     Up to constants, NLL:
    #     0.5 * [ (base_w*lambda_edge)*||diff||^2 - K*log(lambda_edge) ]
    #     (base_w is constant so log(base_w) is dropped)
    #     """
    #     lam = (base_w * lambda_edge).clamp_min(eps)      # (E,)
    #     if rho is not None:
    #         lam = lam * rho[src] * rho[dst]

    #     diff2 = (U[src] - U[dst]).pow(2).sum(dim=1)      # (E,)
    #     Kdim = U.shape[1]

    #     # -K log(lambda_edge) prevents trivial lambda -> inf
    #     loss = 0.5 * (lam * diff2 - Kdim * th.log(lambda_edge.clamp_min(eps))).mean()
    #     return loss

    # U = model.U_pos()
    # h = model.encode(Exp)
    # m, extra, _eta = model.forward_obs_mean(...)
    # rho = model.rho() ...
    # loss_graph = self.charbonnier_graph_loss(U, h, src, dst, base_w, dist,
    #                                         edge_gate=model.edge_gate, rho=rho, ...)

    # U = model.U_pos()
    # h = model.encode(Exp)  # (n, edge_h_dim) or None

    # # compute gate on edges
    # if (model.use_edge_gate and model.edge_gate is not None and h is not None):
    #     gate = model.edge_gate(h[src], h[dst], dist, chunk_size=edge_gate_chunk)  # (E,)
    # else:
    #     # if no edge_gate, treat all edges equally
    #     gate = th.ones_like(base_w)

    # lambda_edge = model.edge_lambda(gate)  # (E,)

    # rho = model.rho() if use_node_reliability else None
    # loss_graph = self.adaptive_gmrf_prior(
    #     U, src, dst,
    #     base_w=base_w,
    #     gate=gate,
    #     lambda_edge=lambda_edge,
    #     rho=rho,
    #     eps=self.eps
    # )

    # # (optional) gate regularization to avoid collapse to all-ones or all-zeros
    # # e.g., encourage sparsity (fewer "strong smoothing" edges):
    # beta_gate = 0.0  # set >0 if needed
    # if beta_gate > 0:
    #     loss_graph = loss_graph + beta_gate * gate.mean()


    def forward_eta(self, lib: th.Tensor, sid: th.Tensor, eps: float = 1e-8) -> th.Tensor:
        U = self.U_pos()
        W = self.W_pos()
        dot = (U @ W.t()).clamp_min(eps)
        b = self._batch_term(sid)
        eta = th.log(lib.clamp_min(eps))[:, None] + th.log(dot) + self.alpha[None, :] + b
        return eta.clamp(-20, 20)

    def forward_obs_mean(self, lib: th.Tensor, sid: th.Tensor,
                         log_link: str = "log1p", eps: float = 1e-8):
        """
        Return mean in observed space (depends on mode):
          - poisson/nb: mu (n,c), extra=None or theta
          - gaussian/t: m  (n,c) mean in log space, extra=sigma
        """
        
        mu = self.forward_eta(lib, sid, eps=eps)
        if self.mode == "poisson":
            mu = th.exp(mu)
            return mu, None

        elif self.mode == "nb":
            mu = th.exp(mu)
            theta = self.theta_pos()[None, :].expand_as(mu) 
            return mu, theta
        else:
            if log_link == "log1p":
                mu = F.softplus(mu)
            elif log_link == "log":
                mu = mu
            else:
                raise ValueError(f"Unknown log_link={log_link}")
            sigma = self.sigma_pos()[None, :].expand_as(mu)
            return mu, sigma

class StageA_GNN(nn.Module):
    """
    GNN-enhanced StageA for common genes.
    """
    def __init__(self, n: int, c: int, n_slices: int,
                 K: int = 32, R: int = 16,
                 mode: ModeType = "poisson",
                 u_gen: UGenType = "gnn",
                 gnn_layers: int = 2,
                 h_dim: int = 64,
                 use_edge_gate: bool = True,
                 use_node_reliability: bool = True,
                 edge_gate_chunk: Optional[int] = None):
        super().__init__()
        self.n, self.c, self.n_slices = int(n), int(c), int(n_slices)
        self.K, self.R = int(K), int(R)
        self.mode = mode
        self.u_gen = u_gen
        self.gnn_layers = int(gnn_layers)
        self.use_edge_gate = bool(use_edge_gate)
        self.use_node_reliability = bool(use_node_reliability)
        self.edge_gate_chunk = edge_gate_chunk

        # self.enc = nn.Sequential(
        #     nn.Linear(c, h_dim), nn.ReLU(),
        #     nn.Linear(h_dim, h_dim), nn.ReLU(),
        # )
        self.enc = nn.Sequential(
            nn.Linear(c, h_dim),
            # nn.LayerNorm(h_dim),
            nn.ReLU(),

            nn.Linear(h_dim, h_dim),
            # nn.LayerNorm(h_dim),
            nn.ReLU(),
        )
        # self.h_norm = nn.LayerNorm(h_dim)

        self.upd = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h_dim, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, h_dim)
            )
            for _ in range(self.gnn_layers)
        ])
        self.ln = nn.ModuleList([nn.LayerNorm(h_dim) for _ in range(self.gnn_layers)])

        if self.use_edge_gate:
            self.edge_gate = EdgeGate(dim_h=h_dim, hidden=64)
        else:
            self.edge_gate = None

        if self.use_node_reliability:
            self.rho_raw = nn.Parameter(th.zeros(n))

        if self.u_gen == "gnn":
            self.toU = nn.Linear(h_dim, K)
            nn.init.xavier_uniform_(self.toU.weight, gain=0.1)
            nn.init.constant_(self.toU.bias, 1.0)
        else:
            self.U_base_raw = nn.Parameter(th.ones(n, K))
            self.toDelta = nn.Linear(h_dim, K)
            nn.init.zeros_(self.toDelta.weight)
            nn.init.zeros_(self.toDelta.bias)

        self.W_raw = nn.Parameter(th.randn(c, K) * 0.01)
        self.alpha = nn.Parameter(th.zeros(c))
        self.P = nn.Parameter(th.randn(n_slices, R) * 0.02)
        self.Q = nn.Parameter(th.randn(c, R) * 0.02)

        if self.mode == "nb":
            self.theta_raw = nn.Parameter(th.zeros(c))
        if self.mode in ("gaussian", "t"):
            self.sigma_raw = nn.Parameter(th.zeros(c))

    def rho(self) -> Optional[th.Tensor]:
        if not self.use_node_reliability:
            return None
        return th.sigmoid(self.rho_raw)

    def W_pos(self) -> th.Tensor:
        return F.softplus(self.W_raw)

    def theta_pos(self) -> th.Tensor:
        return F.softplus(self.theta_raw) + 1e-4

    def sigma_pos(self) -> th.Tensor:
        return F.softplus(self.sigma_raw) + 1e-4

    def mp_aggregate(self, n: int, src: th.Tensor, dst: th.Tensor,
                      w: th.Tensor, h: th.Tensor, eps: float = 1e-8):
        try:
            from torch_scatter import scatter_add  # type: ignore
            m = scatter_add(w[:, None] * h[src], dst, dim=0, dim_size=n)
            deg = scatter_add(w, dst, dim=0, dim_size=n)
            return m / (deg[:, None] + eps)
        except Exception:
            dim = h.shape[1]
            m = th.zeros((n, dim), device=h.device, dtype=h.dtype)
            deg = th.zeros((n,), device=h.device, dtype=h.dtype)
            m.index_add_(0, dst, w[:, None] * h[src])
            deg.index_add_(0, dst, w)
            return m / (deg[:, None] + eps)

    def gnn_forward(self, x_common: th.Tensor, src: th.Tensor, dst: th.Tensor, base_w: th.Tensor) -> th.Tensor:
        n = x_common.shape[0]
        h = self.enc(x_common)
        # h = self.h_norm(h)

        rho = self.rho()
        for l in range(self.gnn_layers):
            if self.use_edge_gate:
                h_src = h[src]
                h_dst = h[dst]
                gate = self.edge_gate(h_src, h_dst, chunk_size=self.edge_gate_chunk)
                w = base_w * gate
            else:
                w = base_w
            if rho is not None:
                w = w * rho[src] * rho[dst]

            neigh = self.mp_aggregate(n, src, dst, w, h)
            # h = neigh
            # ------------------------------------------------------------
            # 3) update: Pre-LN on messages (more stable than Post-LN)
            #    h <- h + MLP(LN(neigh))
            # ------------------------------------------------------------
            # h = h + self.upd[l](self.ln[l](neigh))
            h = self.ln[l](h + self.upd[l](neigh)) # check
        return h, w

    def compute_U(self, x_common: th.Tensor, src: th.Tensor, dst: th.Tensor, 
                  base_w: th.Tensor, dist =None) -> Tuple[th.Tensor, th.Tensor]:
        h, w = self.gnn_forward(x_common, src, dst, base_w)
        if self.u_gen == "gnn":
            U = F.softplus(self.toU(h))
        else:
            U = F.softplus(self.U_base_raw + self.toDelta(h))
        return U, h, w

    def forward_common(self, U: th.Tensor, lib: th.Tensor, sid: th.Tensor, batch_correct: bool,
                       log_link: str = "log1p", eps: float = 1e-8):
        W = self.W_pos()
        dot = (U @ W.t()).clamp_min(eps)
        if batch_correct:
            b = 0.0
        else:
            Pm = self.P.mean(dim=0, keepdim=True)
            b = (self.P[sid] - Pm) @ self.Q.t()

        mu = th.log(lib.clamp_min(eps))[:, None] + th.log(dot) + self.alpha[None, :] + b
        mu = mu.clamp(-20, 20)

        if self.mode == "poisson":
            mu = th.exp(mu)
            return mu, None
    
        elif self.mode == "nb":
            mu = th.exp(mu)
            theta = self.theta_pos()[None, :].expand_as(mu) 
            return mu, theta

        else:
            if log_link == "log1p":
                mu = F.softplus(mu)
            elif log_link == "log":
                mu = mu
            else:
                raise ValueError(f"Unknown log_link={log_link}")
            sigma = self.sigma_pos()[None, :].expand_as(mu)
            return mu, sigma


# =========================
# small utils
# =========================
def scalar2array(X, size):
    """
    Convert scalar / vector / matrix-like into ndarray of shape `size`.
    - If X is None -> zeros(size)
    - If scalar -> full(size, scalar)
    - If 1D and matches last dim -> broadcast to `size`
    - If 1D and matches first dim -> column-broadcast
    - Else must already match size
    """
    if X is None:
        return np.zeros(size, dtype=float)

    X = np.asarray(X)
    if X.ndim == 0:
        return np.full(size, float(X), dtype=float)
    if X.ndim == 1 and X.shape[0] == size[-1]:
        return np.tile(X.reshape(1, -1), (size[0], 1)).astype(float)
    if X.ndim == 1 and X.shape[0] == size[0]:
        return np.tile(X.reshape(-1, 1), (1, size[-1])).astype(float)
    assert X.shape == tuple(size), f"X.shape={X.shape} != size={size}"
    return X.astype(float)

def normal_scaler( X, Xm=None, Xs=None,  xp = th):
    X = X.clone()
    N,D = X.shape
    Xm = xp.mean(X, 0) if Xm is None else Xm

    X -= Xm 
    Xs = xp.sqrt(xp.sum(xp.square(X))/(N*D/2)) if Xs is None else Xs
    X /= Xs
    return X


def normalize_l2(X: th.Tensor, eps: float = 1e-8) -> th.Tensor:
    X = X.clone()
    l2x = th.norm(X, dim=1, keepdim=True)
    l2x[l2x == 0] = 1
    l2x = th.clamp(l2x, min=1e-8)
    return X / l2x


def label_to_order(x, order=None):
    uniq, inv = np.unique(x, return_inverse=True)
    if not order is None:
        order = np.array(order)
        inv = order[inv]
    return inv

# =========================
# graph edges (self-contained fallback)
# =========================
def check_knnpos(src, dst, Cor, size=50, scale=0.5):
    from ..plotting import pview
    np.random.seed(0)
    nsel = size
    # col = np.array(cc.pl.random_colors(nsel))

    src_np =  src.cpu().numpy() if th.is_tensor(src) else src
    dst_np =  dst.cpu().numpy() if th.is_tensor(dst) else dst

    selpos = np.random.choice(np.unique(src_np), size=nsel, replace=False)
    selpos = np.sort(selpos)
    selidx = np.isin(src_np, selpos)
    selsrc = src_np[selidx]
    seldst = dst_np[selidx]
    oidx  = np.unique(selsrc, return_inverse=True)[1]
    # selcol = col[oidx]
    sel_corr = pd.DataFrame( Cor[ np.r_[seldst, selpos]], columns=list('xyz'))
    sel_corr['group'] =  np.r_[selsrc, selpos].astype(str)
    # sel_corr['color'] =  np.r_[selcol, col]
    pview(sel_corr, group='group', scale=scale)

def coord_edges(coordx, coordy=None,
                knn=50,
                radius=None,
                
                max_neighbor = int(1e4),
                method='sknn' ,
                keep_loops= True,
                return_array = False,
                n_jobs = -1):
    from ._neighbors import Neighbors

    if coordy is None:
        coordy = coordx
    
    cknn = Neighbors( method=method ,metric='euclidean', n_jobs=n_jobs)
    cknn.fit(coordx, radius_max= None,max_neighbor=max_neighbor)
    distances, indices = cknn.transform(coordy, knn=knn, radius = radius)

    if return_array:
        return distances, indices

    src = np.concatenate(indices, axis=0).astype(np.int64)
    dst = np.repeat(np.arange(len(indices)), list(map(len, indices))).astype(np.int64)
    dist = np.concatenate(distances, axis=0)

    if (coordy is None) and (not keep_loops):
        mask = src != dst
        src = src[mask]
        dst = dst[mask]
        dist = dist[mask]
    # print(f'mean edges: {dist.shape[0]/coordy.shape[0]}')
    return [src, dst, dist]

def coord_edges0(coordx: np.ndarray, coordy: Optional[np.ndarray] = None,
                knn: int = 50,
                radius: Optional[float] = None,
                method: str = "sknn",
                keep_loops: bool = True,
                return_array: bool = False,
                n_jobs: int = -1):
    """
    Return directed edges from neighbors of each node in coordy to coordx.

    Outputs (src, dst, dist):
      src: neighbor indices in coordx (concat)
      dst: query indices in coordy (repeat)
      dist: euclidean distances
    """
    coordx = np.asarray(coordx, dtype=float)
    coordy = coordx if coordy is None else np.asarray(coordy, dtype=float)

    distances, indices = None, None
    try:
        # if you have your own neighbor backend
        from ._neighbors import Neighbors  # type: ignore
        cknn = Neighbors(method=method, metric="euclidean", n_jobs=n_jobs)
        cknn.fit(coordx, radius_max=None, max_neighbor=int(1e4))
        distances, indices = cknn.transform(coordy, knn=knn, radius=radius)
    except Exception:
        from sklearn.neighbors import NearestNeighbors
        k = int(knn) if knn is not None else 0
        if k <= 0 and radius is None:
            raise ValueError("coord_edges: need knn>0 or radius!=None")
        if k <= 0:
            k = min(50, coordx.shape[0])

        nnbrs = NearestNeighbors(n_neighbors=min(k, coordx.shape[0]), metric="euclidean", n_jobs=n_jobs)
        nnbrs.fit(coordx)
        dist_k, ind_k = nnbrs.kneighbors(coordy, return_distance=True)

        if radius is None:
            distances = [dist_k[i] for i in range(coordy.shape[0])]
            indices = [ind_k[i] for i in range(coordy.shape[0])]
        else:
            distances, indices = [], []
            rad = float(radius)
            for i in range(coordy.shape[0]):
                m = dist_k[i] <= rad
                distances.append(dist_k[i][m])
                indices.append(ind_k[i][m])

    if return_array:
        return distances, indices

    src = np.concatenate([np.asarray(ix, dtype=np.int64) for ix in indices], axis=0)
    dst = np.repeat(np.arange(len(indices), dtype=np.int64), [len(ix) for ix in indices])
    dist = np.concatenate([np.asarray(dx, dtype=float) for dx in distances], axis=0)

    if (coordy is coordx) and (not keep_loops):
        mask = src != dst
        src, dst, dist = src[mask], dst[mask], dist[mask]
    return [src, dst, dist]


def common_genes(G_gene: List[np.ndarray], rate=1.0) -> np.ndarray:
    A_gene  = np.concatenate(G_gene, axis=0)
    unique, counts = np.unique(A_gene, return_counts=True)
    thresh = rate * len(G_gene) 
    common = unique[counts >= thresh]
    return common