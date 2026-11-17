import numpy as np
import scipy.sparse as ssp

from typing import List, Optional, Literal, Dict, Tuple, Union
import random
import os
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

ModeType = Literal["poisson", "nb", "gussian", 't']
UGenType  = Literal["gnn", "residual"] 
BType    = Literal["full", "lowrank"]

class thbase():
    def __init__(self, device=None, dtype =None, eps=1e-8, seed=200504, ):
        self.device = (th.device('cuda' if th.cuda.is_available() else 'cpu') 
                       if device is None else device)
        self.dtype = (th.float32 if dtype is None else dtype)
        self.eps = eps
        self.seed = seed
        self.seed_torch(seed)

    def is_sparse(self, X):
        if ssp.issparse(X):
            return True, 'scipy'
        elif th.is_tensor(X):
            return X.is_sparse, 'torch'
        else:
            return False, 'numpy'

    def to_tensor(self, X, dtype=None, device=None, todense=False, requires_grad =False):
        if th.is_tensor(X):
            X = X.clone()
        elif ssp.issparse(X):
            X = self.spsparse_to_thsparse(X)
            if todense:
                X = X.to_dense()
        else:
            try:
                X = th.asarray(X, dtype=dtype)
            except:
                raise ValueError(f'{type(X)} cannot be converted to tensor')
        X = X.clone().to(device=device, dtype=dtype)
        X.requires_grad_(requires_grad)
        return X

    def spsparse_to_thsparse(self, X):
        XX = X.tocoo()
        values = XX.data
        indices = np.vstack((XX.row, XX.col))
        i = th.LongTensor(indices)
        v = th.tensor(values, dtype=th.float64)
        shape = th.Size(XX.shape)
        return th.sparse_coo_tensor(i, v, shape)

    def thsparse_to_spsparse(self, X):
        XX = X.to_sparse_coo().coalesce()
        values = XX.values().detach().cpu().numpy()
        indices = XX.indices().detach().cpu().numpy()
        shape = XX.shape
        return ssp.csr_array((values, indices), shape=shape)

    def seed_torch(self, seed=200504):
        if not seed is None:
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            th.manual_seed(seed)
            if th.cuda.is_available():
                th.cuda.manual_seed(seed)
                th.cuda.manual_seed_all(seed)
                th.backends.cudnn.deterministic = True
                th.backends.cudnn.benchmark = False
            if hasattr(th.backends, 'mps') and th.backends.mps.is_available():
                th.mps.manual_seed(seed)

class SRG(thbase):
    def __init__(self,  
            Exp: Union[np.ndarray, th.Tensor],
            Cor: Union[np.ndarray, th.Tensor],
            BI: Union[np.ndarray, th.Tensor],
            mode: ModeType = "poisson",

            device: str = "cuda" if th.cuda.is_available() else "cpu",
            dtype=th.float32,
            verbose: int = 1,
            eps: float = 1e-8,
            seed: int = 0,):
        super().__init__(device=device, dtype=dtype, eps=eps, seed=seed)
        self.Exp = self.to_tensor(Exp, dtype=self.dtype, device='cpu')
        self.Cor = self.to_tensor(Cor, dtype=self.dtype, device='cpu')
        self.BI  = BI
        self.BIv = label_to_vector(BI)
        self.BIv = self.to_tensor(self.BIv, dtype=self.dtype, device=self.device)
        self.nB = len(set(BI))
        self.N = self.Exp.shape[0]
        self.G = self.Exp.shape[1]

        self.Lib = self.Exp.sum(1).to(self.device) + 1.0
        assert self.nB == self.BIv.max().item() + 1, "Error: nB != max(BIv) + 1"

        self.mode = mode
        self.device = device
        self.dtype = dtype
        self.verbose = verbose
        self.seed = seed
    
    def KnnGraph(self, knn=None, radius=None, knn_method='global', BIv=None,  ks=1, 
                 keep_loops=False,  kd_method='annoy', CI=1.0):
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
            BIv = self.BIv.detach().cpu().numpy() if BIv is None else BIv
            nB = BIv.max() + 1
            knn = scalar2array(knn, (nB, 2))
            radius = scalar2array(radius, (nB, 2))
            src, dst, dist = [], [], []
            pbar = tqdm(range(nB), total=nB, desc=f'KnnGraph pair')
            for i in pbar:
                for j in range(i, min(i+ks+1, nB)):
                    bid1 = np.where(BIv == i)[0]
                    bid2 = np.where(BIv == j)[0]
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
        src = self.to_tensor(src, dtype=th.int64, device=self.device)
        dst = self.to_tensor(dst, dtype=th.int64, device=self.device)
        return dst, src

    def rbe_train(self, 
        C_gene: th.Tensor,

        use_expw : bool = False,
        sigma: float = 1.0,
        tau: float = 1.0,
        B_mode: BType = "lowrank",

        u_enhence: bool = False,
        u_gen: UGenType = "gnn",
        K: int = 32, R: int = 16, 
        h_dim: int = 64,
        gnn_layers: int = 2,

        lambda_graph: float = 1e-2,
        use_edge_gate: bool = True,
        use_node_reliability: bool = False,
        epochs: int = 200,
        lr: float = 2e-2,
        reg_l2: float = 1e-6,

    ) -> Dict[str, any]:
        th.manual_seed(self.seed)
        np.random.seed(self.seed)

        Exp = self.Exp[:,C_gene].clone().to(self.device, dtype=self.dtype)
        Cor = self.Cor
        n,g = Exp.shape
        Lib = self.Lib
        BIv = self.BIv
        nB  = self.nB
        mode = self.mode

        src, dst = self.src, self.dst
        dist = th.norm(Cor[src] - Cor[dst], dim=1).to(self.device, dtype=self.dtype)
        sigma = dist.mean().item() if sigma is None else sigma

        if u_enhence:
            use_expw = False if use_edge_gate else False

        if use_expw:
            Expn = normalize(Exp.cpu())
            diste = th.norm(Expn[src] - Expn[dst], dim=1).to(self.device, dtype=self.dtype)
            del Expn
            tau = diste.mean().item() if tau is None else tau
            base_w = th.exp(- (dist ** 2) / (sigma**2) - (diste ** 2) / (2 * tau * tau))
            print(f"[StageA] n={n} g={g} slices={self.nB} common={g} sigma={sigma:.3f} tau={tau:.3f}")
        else:
            base_w = th.exp(- (dist ** 2) / (sigma**2))
            print(f"[StageA] n={n} g={g} slices={self.nB} common={g} sigma={sigma:.4f}")

        if u_enhence:
            model = StageA_GNN(
                n=n, c=g, n_slices=nB, K=K, R=R,
                mode=mode, u_gen=u_gen, gnn_layers=gnn_layers,
                h_dim=h_dim, 
                use_edge_gate=use_edge_gate,
                use_node_reliability=use_node_reliability
            ).to(self.device)

            opt = th.optim.Adam(model.parameters(), lr=lr)
            pbar = tqdm(range(epochs), total=epochs,
                        desc=f'stageA_enhance',  colour='red', disable=(self.verbose <1))
            for ep in pbar:
                U, h = model.compute_U(Exp, src, dst, base_w, dist)

                mu, theta = model.forward_mu_common(U, Lib, BIv, batch_correct=False)
                if mode == "poisson":
                    loss_data = (mu - Exp * th.log(mu.clamp_min(self.eps))).mean()
                else:
                    loss_data = (-self.nb_log_prob(Exp, mu, theta)).mean()
                rho = model.rho() if use_node_reliability else None
                loss_graph = self.charbonnier_graph_loss(U, h, src, dst, base_w, dist,
                                                          edge_gate=model.edge_gate, rho=rho, eps=1e-3)
                loss_graph = lambda_graph * loss_graph
                
                # stable h-> exp,
                # with th.no_grad():
                #     h0 = model.enc(Exp)                    # (n,h_dim)
                #     h0 = F.normalize(h0, dim=1)
                #     cos = (h0[src] * h0[dst]).sum(dim=1)   # (E,)
                #     w_feat = th.exp((cos - 1.0) / (tau + 1e-8))   # (0,1]
                # base_w2 = base_w * w_feat

                loss_reg = 0.0
                if reg_l2 > 0:
                    for p in model.parameters():
                        loss_reg = loss_reg + p.pow(2).mean()
                    loss_reg = reg_l2 * loss_reg

                loss = loss_data + loss_graph + loss_reg

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                msg ={ 'loss_data': loss_data.item(), 'loss_graph': loss_graph.item()}
                if reg_l2 > 0:
                    msg['loss_reg'] = loss_reg.item()
                if rho is not None:
                    msg['rho_mean'] = rho.mean().item()
                pbar.set_postfix(msg)
            pbar.close()

            with th.no_grad():
                U, _ = model.compute_U(Exp, src, dst, base_w, dist)
                stagea = {
                    # "modelA": model,
                    "U": U.detach().cpu(),                     # (n,K)
                    "P": model.P.detach().cpu(),                # (S,R) 
                    "Wc": model.W_pos().detach().cpu(),         # (c,K)
                    "alphac": model.alpha.detach().cpu(),       # (c,)
                    "Qc": model.Q.detach().cpu(),               # (c,R)
                }
                if use_node_reliability:
                    stagea["rho"] = model.rho().detach().cpu()
                if mode == "nb":
                    stagea["thetac"] = model.theta_pos().detach().cpu()

        else:
            model = StageA_Simple(
                n=n, c=g, n_slices=nB, K=K, R=R,
                mode=mode, B_mode=B_mode
            ).to(self.device)
            opt = th.optim.Adam(model.parameters(), lr=lr)
            pbar = tqdm(range(epochs), total=epochs,
                        desc=f'stageA_simple',  colour='red', disable=(self.verbose <1))
            for ep in pbar:
                mu, theta = model.forward_mu(Lib, BIv)

                if mode == "poisson":
                    loss_data = (mu - Exp * th.log(mu.clamp_min(self.eps))).mean()
                else:
                    loss_data = (-self.nb_log_prob(Exp, mu, theta)).mean()
                loss_graph = self.laplacian_graph_loss(model.U_pos(), src, dst, base_w,)

                loss_reg = 0.0
                if reg_l2 > 0:
                    for p in model.parameters():
                        loss_reg = loss_reg + p.pow(2).mean()
                    loss_reg = reg_l2 * loss_reg
                loss = loss_data + lambda_graph * loss_graph + loss_reg

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                msg ={ 'loss_data': loss_data.item(), 'loss_graph': loss_graph.item()}
                if reg_l2 > 0:
                    msg['loss_reg'] = loss_reg.item()
                pbar.set_postfix(msg)
            pbar.close()

            with th.no_grad():
                stagea = {
                    # "modelA": model,
                    "U": model.U_pos().detach().cpu(),                     # (n,K)
                    "P": model.P.detach().cpu(),                # (S,R) 
                    "Wc": model.W_pos().detach().cpu(),         # (c,K)
                    "alphac": model.alpha.detach().cpu(),       # (c,)
                    "Qc": model.Q.detach().cpu(),               # (c,R)
                }
                if mode == "nb":
                    stagea["thetac"] = model.theta_pos().detach().cpu()
        stagea['C_gene'] = C_gene.detach().cpu()
        self.stagea = stagea

    def imputation_train(self,
        U_genes: List[th.Tensor],
        epochs: int = 200,
        lr: float = 2e-2,
        reg_l2: float = 1e-6,

    ) -> Dict[str, any]:
        th.manual_seed(self.seed+1)
        np.random.seed(self.seed+1)
        
        Exp = self.Exp
        Lib = self.Lib
        BIv = self.BIv
        nB  = self.nB
        mode = self.mode
        device, dtype, verbose = self.device, self.dtype, self.verbose

        U_geneC = th.unique(th.concat(U_genes)).astype(th.long)
        Gnc = U_geneC.size(0)
        if Gnc == 0:
            raise ValueError("non-common genes is empty in stageB.")

        # TODO slow for large Gnc
        g2l = {int(gg): i for i, gg in enumerate(U_geneC)} # slow for large Gnc
        U_geneL = [] #local ids in non-common
        U_geneG = U_genes #global ids in union
        for s in range(nB):
            gl = U_geneG[s]
            if len(gl) == 0:
                U_geneL.append( th.empty(0, dtype=th.long, device=device))
            else:
                ll = th.tensor([g2l[int(x)] for x in gl], device=device, dtype=th.long)
                U_geneL.append(ll)

        # torch fixed
        U = self.stagea['U'].to(dtype=dtype, device=device)    # (n,K)
        P = self.stagea['P'].to(dtype=dtype, device=device)     # (S,R)
        Pm = P.mean(dim=0, keepdim=True)


        dec = SharedDecoder(Gnc=Gnc, K=U.shape[1], R=P.shape[1], mode=mode).to(device)
        opt = th.optim.Adam(dec.parameters(), lr=lr)

        verbose and print(f"[StageB] shared decoder: noncommon={Gnc} mode={mode}")
        pbar = tqdm(range(epochs), total=epochs,
                    desc=f'stageB_impute',  colour='blue', disable=(verbose <1))
        for ep in pbar:
            loss_data = th.tensor(0.0, device=device)
            denom = 0
            for s in range(nB):
                idx = th.where(BIv == s)[0]
                if idx.numel() == 0:
                    continue
                gl = U_geneG[s]
                ll = U_geneL[s]
                if ll.numel() == 0:
                    continue

                Y = Exp[idx][:, gl].to(device=device)
                w, q, a, t = dec.decode(ll)  # w:(m,K)

                dot = (U[idx] @ w.t()).clamp_min(1e-8)      # (ns,m)
                b   = (P[s:s+1] - Pm) @ q.t()               # (1,m)
                logmu = th.log(Lib[idx].clamp_min(1e-8))[:, None] + th.log(dot) + a[None, :] + b
                mu = th.exp(logmu.clamp(-20, 20))

                if mode == "poisson":
                    loss_s = (mu - Y * th.log(mu.clamp_min(1e-8))).mean()
                else:
                    theta = t[None, :].expand_as(mu)
                    loss_s = (-self.nb_log_prob(Y, mu, theta)).mean()

                loss_data = loss_data + loss_s
                denom += 1
            loss_data = loss_data / max(denom, 1)

            loss_reg = 0.0
            if reg_l2 > 0:
                for p in dec.parameters():
                    loss_reg += p.pow(2).mean()
                loss_reg = reg_l2 * loss_reg

            loss = loss_data + loss_reg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            msg ={ 'loss_data': loss_data.item()}
            if reg_l2 > 0:
                msg['loss_reg'] = loss_reg.item()
            pbar.set_postfix(msg)
        pbar.close()
        self.stageb = {"decoder": dec, "U_geneL": U_geneL, "U_geneC": U_geneC}


    @th.no_grad()
    def prediction_common(self, block_size = None, device='cpu'):
        """
        common genes prediction with cell blocking.

        Returns
        -------
        mu_corr : (n, c)   batch-removed (unified) Poisson/NB mean
        mu_raw  : (n, c)   slice-specific mean (includes batch term)
        """
        Lib = self.Lib.to(device)            # (n,)
        BIv = self.BIv.to(device).long()     # (n,)

        U = self.stagea["U"].to(device)      # (n,K)
        Wc = self.stagea["Wc"].to(device)    # (c,K)  (>=0)
        alphac = self.stagea["alphac"].to(device)  # (c,)

        P = self.stagea["P"].to(device)      # (S,R)
        Qc = self.stagea["Qc"].to(device)    # (c,R)
        Pm = P.mean(dim=0, keepdim=True)     # (1,R)
        Bs = (P - Pm) @ Qc.t()               # (S,c)

        if block_size is None:
            logCorr  =  th.log(Lib.clamp_min(1e-8))[:, None] + alphac[None, :]
            logCorr +=  th.log((U @ Wc.t()).clamp_min(1e-8))

            mu_corr = th.exp(logCorr.clamp(-20, 20))
            mu_raw = Bs[BIv] + logCorr
            mu_raw = th.exp(mu_raw.clamp(-20, 20))
        else:
            n = U.shape[0]
            c = Wc.shape[0]
            mu_corr = th.empty((n, c), device='cpu', dtype=th.float32)
            mu_raw  = th.empty((n, c), device='cpu', dtype=th.float32)

            for st in range(0, n, block_size):
                ed = min(st + block_size, n)

                libb = Lib[st:ed].clamp_min(1e-8)          # (B,)
                Ub   = U[st:ed]                             # (B,K)
                sidb = BIv[st:ed]                           # (B,)

                dot = (Ub @ Wc.t()).clamp_min(1e-8)         # (B,c)
                logCorr = th.log(libb)[:, None] + alphac[None, :] + th.log(dot)  # (B,c)
                logRaw = logCorr + Bs[sidb]  

                mu_corr[st:ed] = th.exp(logCorr.clamp(-20, 20)).detach().cpu()
                mu_raw[st:ed] = th.exp(logRaw.clamp(-20, 20)).detach().cpu()

        return mu_corr.detach().cpu(), mu_raw.detach().cpu()
        
    @th.no_grad()
    def prediction_impute(self, block_size=None, device='cpu'):
        """
        Predict noncommon genes for all cells using StageB decoder.
        Returns:
        mu_corr: batch-removed (unified) mean, shape (n,Gnc)
        mu_raw : slice-specific mean,      shape (n,Gnc)
        Note: this is huge in real data; block_size controls cell blocking.
        """
        Lib = self.Lib.to(device)            # (n,)
        BIv = self.BIv.to(device).long()     # (n,)

        U = self.stagea["U"].to(device)      # (n,K)
        P = self.stagea["P"].to(device)      # (S,R)
        Pm = P.mean(dim=0, keepdim=True)     # (1,R)

        dec: SharedDecoder = self.stageb["decoder"].to(device)
        Gnc = dec.emb.num_embeddings
        ll = th.arange(Gnc, device=device, dtype=th.long) 
        w, q, a, t = dec.decode(ll)   # w:(m,K). q:(m,R). a:(m,). t:(m,) 
        Bs = (P - Pm) @ q.t()  # (S,gnc)

        if block_size is None:
            logCorr =  th.log(Lib.clamp_min(1e-8))[:, None] + a[None, :]
            logCorr += th.log( (U @ w.t()).clamp_min(1e-8) )
 
            mu_corr = th.exp(logCorr.clamp(-20, 20))
            mu_raw = Bs[BIv] + logCorr
            mu_raw = th.exp(mu_raw.clamp(-20, 20))

            return mu_corr, mu_raw
        else:
            n = U.shape[0]
            mu_corr = th.empty((n, Gnc), device='cpu', dtype=th.float32)
            mu_raw  = th.empty((n, Gnc), device='cpu', dtype=th.float32)
            for st in range(0, n, block_size):
                ed = min(st + block_size, n)

                Ub = U[st:ed]                                     # (B,K)
                libb = Lib[st:ed].clamp_min(1e-8)                 # (B,)
                sidb = BIv[st:ed]                                 # (B,)

                dot = (Ub @ w.t()).clamp_min(1e-8)                # (B,Gnc)
                lmc = th.log(libb)[:, None] + th.log(dot) + a[None, :]   # (B,Gnc)
                lmr = lmc + Bs[sidb]                              # (B,Gnc)

                mu_corr[st:ed] = th.exp(lmc.clamp(-20, 20)).detach().cpu()
                mu_raw[st:ed]  = th.exp(lmr.clamp(-20, 20)).detach().cpu()

        return mu_corr.detach().cpu(), mu_raw.detach().cpu()

    @th.no_grad()
    def prediction_space(self, block_size=None, device='cpu'):
        mcc, mcr = self.prediction_common(block_size = block_size, device=device)
        mnc, mnr = self.prediction_impute(block_size = block_size, device=device)
        mcc = th.cat([mcc, mnc], dim=1)
        mcr = th.cat([mcr, mnr], dim=1)
        return mcc, mcr

    def nb_log_prob(self, y: th.Tensor, mu: th.Tensor, theta: th.Tensor, eps: float = 1e-8) -> th.Tensor:
        # NB(mean=mu, inv_disp=theta)
        t1 = th.lgamma(y + theta) - th.lgamma(theta) - th.lgamma(y + 1.0)
        t2 = theta * (th.log(theta + eps) - th.log(theta + mu + eps))
        t3 = y * (th.log(mu + eps) - th.log(theta + mu + eps))
        return t1 + t2 + t3

    def charbonnier_graph_loss(self,
        U: th.Tensor, h: th.Tensor,
        src: th.Tensor, dst: th.Tensor,
        base_w: th.Tensor, dist: th.Tensor,
        edge_gate, #: EdgeGate,
        rho: Optional[th.Tensor] = None,
        eps: float = 1e-3
    ) -> th.Tensor:
        if edge_gate is None:
            w = base_w
        else:
            h_src = h[src]; h_dst = h[dst]
            gate = edge_gate(h_src, h_dst, dist)
            w = base_w * gate
            # w = base_w * gate.detach()
        if rho is not None:
            w = w * rho[src] * rho[dst]
        diff2 = (U[src] - U[dst]).pow(2).sum(dim=1)
        return (w * th.sqrt(diff2 + eps * eps)).mean()

    def laplacian_graph_loss(self, U, src, dst, base_w, eps=1e-3):
        diff2 = (U[src] - U[dst]).pow(2).sum(dim=1)
        return (base_w * diff2).mean()
        return (base_w * th.sqrt(diff2 + eps * eps)).mean()

class EdgeGate(nn.Module):
    """
    [h_src, h_dst, dist] -> sigmoid gate in (0,1)
    """
    def __init__(self, dim_h: int, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim_h + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, h_src: th.Tensor, h_dst: th.Tensor, dist: th.Tensor) -> th.Tensor:
        x = th.cat([h_src, h_dst, dist[:, None]], dim=1)
        return th.sigmoid(self.mlp(x)).squeeze(-1)  # (E,)

class StageA_Simple(nn.Module):
    """
     common genes（c 个）：
      mu_{i,g} = lib_i * (u_i^T w_g) * exp(alpha_g + b_{s(i),g})

    低秩 batch：
      b_{s,g} = (P_s - mean(P))^T Q_g     (rank=R)

    约束：
      u_i,w_g >= 0  用 softplus
    """
    def __init__(self, n: int, c: int, n_slices: int, K: int = 32, R: int = 16, 
                 mode: ModeType = "poisson", B_mode: str = "lowrank"):
        super().__init__()
        self.n, self.c = n, c
        self.n_slices = n_slices
        self.K, self.R = K, R
        self.mode = mode

        self.U_raw = nn.Parameter(th.randn(n, K) * 0.01)    # (n,K)
        self.W_raw = nn.Parameter(th.randn(c, K) * 0.01)    # (c,K)
        self.alpha = nn.Parameter(th.zeros(c))              # (c,)

        if B_mode == "full":
            self.B = nn.Parameter(th.zeros(n_slices, c))  # (S,G)
            nn.init.zeros_(self.B)
        elif B_mode == "lowrank":
            self.P = nn.Embedding(n_slices, R)  # slice emb
            self.Q = nn.Embedding(c, R)   # gene emb
            nn.init.normal_(self.P.weight, std=0.02)
            nn.init.normal_(self.Q.weight, std=0.02)
        else:
            raise ValueError(f"Unknown B_mode={B_mode}")

        if mode == "nb":
            self.theta_raw = nn.Parameter(th.zeros(c))      # (c,)

    def U_pos(self):
        return F.softplus(self.U_raw)

    def W_pos(self):
        return F.softplus(self.W_raw)

    def theta_pos(self):
        return F.softplus(self.theta_raw) + 1e-4

    def forward_mu(self, lib: th.Tensor, sid: th.Tensor, eps: float = 1e-8):
        """
        lib: (n,)
        sid: (n,) long
        return mu: (n,c)
        """
        U = self.U_pos()          # (n,K)
        W = self.W_pos()          # (c,K)
        dot = (U @ W.t()).clamp_min(eps)  # (n,c)
        alpha = self.alpha.squeeze(-1)  # (c,)

        Pm = self.P.mean(dim=0, keepdim=True)  # (1,R)
        b  = (self.P[sid] - Pm) @ self.Q.t()   # (n,c)

        logmu = th.log(lib.clamp_min(eps))[:, None] + th.log(dot) + alpha  + b
        logmu = logmu.clamp(-20, 20)
        mu = th.exp(logmu)
        if self.mode == "nb":
            theta = self.theta_pos()[None, :].expand_as(mu)
            return mu, theta
        return mu, None

class StageA_GNN(nn.Module):
    """
    common genes only:
      mu_{i,g} = lib_i * (U_i^T W_g) * exp(alpha_g + (P_{s(i)}-mean(P))^T Q_g)
    """
    def __init__(
        self,
        n: int, c: int, n_slices: int,
        K: int = 32, R: int = 16,
        mode: ModeType = "poisson",
        u_gen: UGenType = "gnn",
        gnn_layers: int = 2,
        h_dim: int = 64,
        use_edge_gate: bool = True,
        use_node_reliability: bool = True,
    ):
        super().__init__()
        self.n, self.c, self.n_slices = n, c, n_slices
        self.K, self.R = K, R
        self.mode = mode
        self.u_gen = u_gen
        self.gnn_layers = int(gnn_layers)
        self.use_edge_gate = bool(use_edge_gate)
        self.use_node_reliability = bool(use_node_reliability)

        # encoder: X_common -> h0
        self.enc = nn.Sequential(
            nn.Linear(c, h_dim), nn.ReLU(),
            nn.Linear(h_dim, h_dim), nn.ReLU(),
        )

        # upd + LN
        self.upd = nn.ModuleList([
            nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, h_dim))
            for _ in range(self.gnn_layers)
        ])
        self.ln = nn.ModuleList([nn.LayerNorm(h_dim) for _ in range(self.gnn_layers)])

        # edge attention
        if self.use_edge_gate:
            self.edge_gate = EdgeGate(dim_h=h_dim, hidden=64)
        else:
            self.edge_gate = None

        # node reliability rho_i in (0,1)
        if self.use_node_reliability:
            self.rho_raw = nn.Parameter(th.zeros(n))

        # U head
        if self.u_gen == "gnn":
            self.toU = nn.Linear(h_dim, K)
        else:
            self.U_base_raw = nn.Parameter(th.randn(n, K) * 0.01)
            self.toDelta = nn.Linear(h_dim, K)

        # gene/batch params (common genes)
        self.W_raw = nn.Parameter(th.randn(c, K) * 0.01)
        self.alpha = nn.Parameter(th.zeros(c))
        self.P = nn.Parameter(th.randn(n_slices, R) * 0.02)
        #self.P = nn.Embedding(n_slices, R) # sparse
        self.Q = nn.Parameter(th.randn(c, R) * 0.02)

        if self.mode == "nb":
            self.theta_raw = nn.Parameter(th.zeros(c))

    def rho(self) -> Optional[th.Tensor]:
        if not self.use_node_reliability:
            return None
        return th.sigmoid(self.rho_raw)

    def W_pos(self) -> th.Tensor:
        return F.softplus(self.W_raw)

    def theta_pos(self) -> th.Tensor:
        return F.softplus(self.theta_raw) + 1e-4

    def mp_aggregate(self, n: int, src: th.Tensor, dst: th.Tensor, 
                     w: th.Tensor, h: th.Tensor, eps: float = 1e-8):
        """
        m[dst] += w * h[src];  deg[dst] += w
        return neigh = m/deg
        """
        try:
            from torch_scatter import scatter_add
            m = scatter_add(w[:, None] * h[src], dst, dim=0, dim_size=n)
            deg = scatter_add(w, dst, dim=0, dim_size=n)
            return m / (deg[:, None] + eps)
        except:
            dim = h.shape[1]
            m = th.zeros((n, dim), device=h.device, dtype=h.dtype)
            deg = th.zeros((n,), device=h.device, dtype=h.dtype)
            m.index_add_(0, dst, w[:, None] * h[src])
            deg.index_add_(0, dst, w)
            return m / (deg[:, None] + eps)

    def gnn_forward(self, x_common: th.Tensor,
                    src: th.Tensor, dst: th.Tensor,
                    base_w: th.Tensor, dist: th.Tensor) -> th.Tensor:
        n = x_common.shape[0]
        h = self.enc(x_common)

        rho = self.rho()
        for l in range(self.gnn_layers):
            if self.use_edge_gate:
                h_src = h[src]
                h_dst = h[dst]
                gate = self.edge_gate(h_src, h_dst, dist)  # (E,)
                w = base_w * gate
            else:
                w = base_w
            if rho is not None:
                w = w * rho[src] * rho[dst]

            neigh = self.mp_aggregate(n, src, dst, w, h)
            h = self.ln[l](h + self.upd[l](neigh))
        return h

    def compute_U(self, x_common: th.Tensor,
                  src: th.Tensor, dst: th.Tensor,
                  base_w: th.Tensor, dist: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        h = self.gnn_forward(x_common, src, dst, base_w, dist)
        if self.u_gen == "gnn":
            U = F.softplus(self.toU(h))
        else:
            U = F.softplus(self.U_base_raw + self.toDelta(h))
        return U, h

    def forward_mu_common(self, U: th.Tensor, lib: th.Tensor, sid: th.Tensor, batch_correct: bool, eps: float = 1e-8):
        """
        返回 common genes 的 mu: (n,c)
        batch_correct=True -> b=0
        """
        W = self.W_pos()                            # (c,K)
        dot = (U @ W.t()).clamp_min(eps)            # (n,c)
        if batch_correct:
            b = 0.0
        else:
            Pm = self.P.mean(dim=0, keepdim=True)
            b = (self.P[sid] - Pm) @ self.Q.t()     # (n,c)

        logmu = th.log(lib.clamp_min(eps))[:, None] + th.log(dot) + self.alpha[None, :] + b
        mu = th.exp(logmu.clamp(-20, 20))
        if self.mode == "nb":
            theta = self.theta_pos()[None, :].expand_as(mu)
            return mu, theta
        return mu, None

class SharedDecoder(nn.Module):
    """
    gene_local_id -> (w>=0, q, alpha, theta?)
    """
    def __init__(self, Gnc: int, K: int, R: int, mode: ModeType = "poisson",
                 E: int = 32, H: int = 128):
        super().__init__()
        self.mode = mode
        self.emb = nn.Embedding(Gnc, E)
        self.trunk = nn.Sequential(nn.Linear(E, H), nn.ReLU(),
                                   nn.Linear(H, H), nn.ReLU())
        self.head_w = nn.Linear(H, K)
        self.head_q = nn.Linear(H, R)
        self.head_a = nn.Linear(H, 1)
        if mode == "nb":
            self.head_t = nn.Linear(H, 1)

    def decode(self, gid_local: th.Tensor):
        z = self.trunk(self.emb(gid_local))
        w = F.softplus(self.head_w(z))
        q = self.head_q(z)
        a = self.head_a(z).squeeze(-1)
        if self.mode == "nb":
            t = F.softplus(self.head_t(z)).squeeze(-1) + 1e-4
            return w, q, a, t
        return w, q, a, None

def scalar2array(X, size):
    if X is None:
        return np.zeros(size)
        return np.full(size, np.nan)
    else:
        X = np.array(X)
        if X.ndim == 0:
            return np.full(size, X)
        elif (X.ndim == 1) and (X.shape[0] == size[-1]):
            return np.full(size, X)
        elif (X.ndim == 1) and (X.shape[0] == size[0]):
            return np.full(size, X[:, None])
        else:
            assert X.shape == tuple(size)
            return X

def normalize(X):
    X = X.clone()
    l2x = th.norm(X, dim=1, keepdim=True)
    l2x[l2x == 0] = 1
    l2x = th.clamp(l2x, min=1e-8)
    return X/l2x

def label_to_vector(s: np.ndarray) -> np.ndarray:
    uniq = np.unique(s)
    mp = {int(v): i for i, v in enumerate(uniq)}
    return np.vectorize(mp.get)(s).astype(np.int64)

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

def common_genes(G_gene: List[np.ndarray], rate=1.0) -> np.ndarray:
    A_gene  = np.concatenate(G_gene, axis=0)
    unique, counts = np.unique(A_gene, return_counts=True)
    thresh = rate * len(G_gene) 
    common = unique[counts >= thresh]
    return common
