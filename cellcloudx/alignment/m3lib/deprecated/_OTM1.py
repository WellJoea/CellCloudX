"""Multimodal unbalanced OT with an optional single FGW gradient correction.

solver() fits transported mass; transform() predicts coordinates in either
direction with kernel, nearest, or KNN mapping. Geometry and feature costs use
squared Euclidean distances. Import within the original package: project
normalization helpers and PyKeOps are required, including for dense execution."""
import torch as th
import numpy as np
import math
import re
import scipy.sparse as ssp
from copy import copy
import warnings
from .operation_expectation import (centerlize,  center_normalize, normalize, scaling,  image_normalize,
                                    scale_normalize, spsparse_to_thsparse, thsparse_to_spsparse,
                                    sigma_square_cos, sigma_square)

try:
    import pykeops
    pykeops.set_verbose(False)
    from pykeops.torch import LazyTensor, Genred
except ImportError as exc:
    raise ImportError('pykeops is not installed, `pip install pykeops`') from exc



def scalar2vector(x, L, default = None, force=False):
    if L == 0:
        return []
    if force:
        xs = [ x for i in range(L) ]
    else:
        if ((type(x) in [str, float, int, bool])
            or isinstance(x, (str, bytes))
            or np.isscalar(x)
            or (th.is_tensor(x) and x.ndim == 0)
            or (x is None)):
            xs = [ x for i in range(L) ]
        else:
            assert len(x) == L, f'len(x)={len(x)} != L={L}'
            xs = x
    if default is not None:
        xs = [ default if x is None else x for x in xs ]
    return xs

def n2v(x, v): 
    return v if x is None else x

    
def auto_select_device( device=None, verbose=False):
    # User manually specifies device
    if device is not None:
        if isinstance(device, int):
            device = th.device(f"cuda:{device}")
        else:
            device = th.device(device)
        if verbose:
            print(f"Using specified device: {device}")
        return device
    if not th.cuda.is_available():
        if verbose:
            print("CUDA is not available. Using CPU.")
        return th.device("cpu")
    devices = []
    for i in range(th.cuda.device_count()):
        free, total = th.cuda.mem_get_info(i)
        devices.append(dict(index=i, name=th.cuda.get_device_name(i),
                            free_bytes=free, free_gb=free / 2**30,
                            total_gb=total / 2**30))
    if len(devices) == 0:
        if verbose:
            print("No CUDA device found. Using CPU.")
        return th.device("cpu")
    best = max(devices, key=lambda x: x["free_bytes"])
    selected = th.device(f"cuda:{best['index']}")
    if verbose:
        print(
            f"Using cuda:{best['index']} | {best['name']} | "
            f"free {best['free_gb']:.2f} GB / total {best['total_gb']:.2f} GB"
        )
    return selected

def use_keops_bool( use_keops=None, thresh=1e8):
    use_keops = n2v(use_keops, True)
    if type(use_keops) in [bool]: 
        return use_keops
    else:
        return use_keops >= thresh

def dist2( X, Y, use_keops):
    def kodist2(X, Y):
        x_i = LazyTensor(X[:, None, :])
        y_j = LazyTensor(Y[None, :, :])
        return ((x_i - y_j)**2).sum(dim=2)
    def thdist2(X, Y):
        D = th.cdist(X, Y, p=2)
        D.pow_(2)
        return D
    return kodist2(X, Y) if use_keops else thdist2(X, Y)

def D1_geom(X, Y, sigma2, use_keops):
    return dist2(X, Y, use_keops)/sigma2

def D1_feat(Fxs, Fys, tau2s, fscales_1st, use_keops):
    assert type(Fxs) is list
    Nf = len(Fxs)
    D = 0
    for i in range(Nf):
        Di = dist2(Fxs[i], Fys[i], use_keops) * ( fscales_1st[i]/tau2s[i] )
        if i == 0:
            D = Di
        else:
            D = D + Di
    return D

def plan_extract(P, rowi, colj, chunksize=30000):
    """Return [rowi,colj,Pij] for equal-length paired indices of a scalar plan.
    
    Dense extraction gathers P[rowi,colj]. KeOps batches chunksize candidates,
    without creating a square candidate matrix. Indices become device-local long
    tensors; repeated/empty pairs are supported. KeOps shallow-copies the expression
    wrapper before variable fixing, shares leaf storage, and retains a batched
    fallback on RuntimeError/ValueError. Values remain transported mass."""
    rowi = th.as_tensor(rowi, device=P.device, dtype=th.long).reshape(-1)
    colj = th.as_tensor(colj, device=P.device, dtype=th.long).reshape(-1)

    if th.is_tensor(P):
        Pij = P[rowi, colj]
    else:
        # KeOps 2.3: copy the expression wrapper, sharing leaf Tensor storage.
        # fixvariables replaces formula/variables on this copy; it does not write tensors.
        # This extraction assumes scalar plans without external batch dimensions.
        P = copy(P)
        pattern = r"Var\((\d+),\s*(\d+),\s*([012])\)"
        meta = re.findall(pattern, P.formula)
        # Do not fix twice: KeOps 2.3 can clear already-renumbered variables.
        if {int(i) for i, _, _ in meta} != set(range(len(P.variables))):
            P.fixvariables()
            meta = re.findall(pattern, P.formula)

        cats = {int(i): int(cat) for i, _, cat in meta}
        variables = [
            (cats[i], z.reshape(-1).contiguous() if cats[i] == 2
            else z.reshape(-1, z.shape[-1]))
            for i, z in enumerate(P.variables)
        ]
        # Both endpoints share the same candidate index.
        formula = re.sub(
            pattern,
            lambda m: f"Var({m[1]},{m[2]},{2 if m[3] == '2' else 0})",
            P.formula,
        )

        def extract(formula, batched):
            evaluate = Genred(formula, [], reduction_op="Sum", axis=1)
            Pij = variables[0][1].new_empty(rowi.numel())
            for start in range(0, rowi.numel(), chunksize):
                s = slice(start, start + chunksize)
                ids = (rowi[s], colj[s])
                args = []
                for cat, z in variables:
                    x = z if cat == 2 else z[ids[cat]]
                    if batched:
                        x = x.reshape(1, -1) if cat == 2 else x.unsqueeze(1)
                    args.append(x.contiguous())
                Pij[s] = evaluate(*args).reshape(-1)
            return Pij
        try:
            Pij = extract(formula, False)
        except (RuntimeError, ValueError):
            Pij = extract(P.formula, True)
    return [rowi, colj, Pij]
    
class OTM:
    """Match X rows to Y columns using UOT and optional fixed FGW structure.
    
    For X of shape (nx, d) and Y of shape (ny, d), the transport plan has
    logical matrix shape (nx, ny). A scalar KeOps expression has an additional
    trailing feature dimension of one. Rows correspond to X and columns to Y.
    Each feature modality supplies matching feature dimensions on both sides.
    The returned P contains transported mass, not row-normalized probabilities;
    r = P @ 1 and c = P.T @ 1 are its row and column marginals.
    
    The first-order cost is the unnormalized sum of geometric and feature costs:
        D1 = gscale_1st * dist2(X, Y) / sigma2_1st
             + sum(fscales_1st[k] * dist2(Fx[k], Fy[k]) / tau2s_1st[k]).
    
    When use_fgw=True, within-set squared distances define each structure term:
        G(P) = 0.5 * sum((Cx[i,k] - Cy[j,l])**2 * P[i,j] * P[k,l]).
    The weighted gradient at a reference plan P0 is added to D1. Sinkhorn then
    solves the resulting UOT subproblem with this effective cost held fixed.
    With no supplied reference plan, solver first solves UOT and then applies
    one FGW correction. This is not a converged outer FGW optimization loop.
    
    The kernel is exp(-D_eff / eps), multiplied by label compatibility when
    labels are present. reg='kl' additionally uses the reference measure
    a[:, None] * b[None, :]; reg='entropy' uses an all-ones reference measure.
    
    See __init__ for parameter tuning and solver for input and output shapes.
    """

    def __init__(self, *,
                 normal_g = False,
                 normal_f = None,
                 mode="normal",
                 use_fgw=False,
                 use_lowrank=True,
                 reg="entropy",
                 use_keops=8e8,
                 moment_max_dim=16,
                 sigma2_1st=None, tau2s_1st=None,
                 sigma2_2nd = None, tau2s_2nd  = None,
                 gscale_1st = None, fscales_1st = None,
                 gscale_2nd = None, fscales_2nd = None,
                 ci=1., cf=1., record = False,
                 delta_h=.00,  eps=.05, rho=1., theta = None,
                 uv_iter=1000, tol=1e-5, check_every=10, convergence="hybrid",
                 chunksize = 40000,
                 device=None, dtype=th.float32, verbose=True):
        """Configure additive costs, marginal penalties, and the numerical backend.
        
        Parameters
        ----------
        normal_g : bool or {'each', 'isoscale', 'global', 'X', 'pass'}
            True/'isoscale': separate centers, shared mean scale; 'each': separate
            centers/scales. 'global': joint statistics; 'X': X statistics for both.
            False/'pass' skips normalization. Use shared statistics or False to retain
            centroid offsets in an already registered frame.
        normal_f : str or sequence, optional
            Per-modality normalization; None uses 'ccos'. Modes: cos/cosine (row L2),
            ccos/ccosine (column centering then row L2), zc (center/global scale),
            pcc (row centering/L2), std/zscore (column standardization), image/i2p/img
            (shared robust center/scale), label (multiply by 2), pass/restc/resta.
        mode : {'log', 'normal'}
            Log updates support a wider dynamic range; normal updates are faster in
            some cases but can underflow/overflow for extreme cost/eps ratios.
        use_fgw : bool
            Add one structure-gradient correction; does not enforce one-to-one edges.
        use_lowrank : bool
            Use exact squared-distance factors of width d+2, not a low-rank plan.
        reg : {'entropy', 'kl'}
            Reference measure: ones or a[:,None]*b[None,:]. Equal for all-one budgets.
        use_keops : bool or number
            Force a backend or enable KeOps when nx*ny reaches the threshold (8e8).
            KeOps saves pair-matrix memory at the cost of compilation/reduction overhead.
        moment_max_dim : int
            Dimensions up to this cutoff (16) use factor-space moment matrices;
            larger dimensions use distance reductions. Changes computation, not the model.
        sigma2_1st, tau2s_1st : positive scalar or per-modality sequence, optional
            Geometry/feature squared-distance denominators; None estimates them.
            Larger values weaken the corresponding first-order costs linearly.
        sigma2_2nd, tau2s_2nd : positive scalar or per-modality sequence, optional
            Structural squared-distance denominators; fixed-plan gradients scale with
            their inverse square. Geometry None inherits sigma2_1st; feature None entries
            are estimated and need not inherit explicitly supplied tau2s_1st values.
        gscale_1st, fscales_1st : nonnegative scalar or per-modality sequence, optional
            Additive geometry/feature weights; defaults are 1. Larger weights strengthen
            their terms; zero disables them. Weights are not normalized to sum to one.
        gscale_2nd, fscales_2nd : nonnegative scalar or per-modality sequence, optional
            FGW geometry/feature weights; defaults are 1 and 0. Start small and compare
            weighted gradients with first-order costs. Gradients grow with reference mass.
        ci, cf : scalar
            Retained attributes; unused by the current solver.
        delta_h : float in [0, 1)
            H=(1-delta_h)*Lx@Ly.T+delta_h. Larger floors weaken label discrimination
            for normalized labels. Labels are not normalized internally; log clamping
            means zero compatibility need not produce exactly zero mass.
        eps : positive float
            Entropic strength (0.05). Smaller values sharpen plans but make convergence
            harder; larger values diffuse them. Compare with effective cost magnitudes.
            Labels are equivalent to the additional cost -eps*log(H).
        rho : positive finite scalar or pair
            Marginal KL penalties (rho_x,rho_y), default 1 each. Larger values enforce
            budgets more strongly; smaller values allow more mass change. Update exponents
            are rho/(rho+eps). Infinite penalties are not supported here.
        uv_iter, tol, check_every : int, float, int
            Sinkhorn iteration cap (1000), tolerance (1e-5), and check interval (10).
            Smaller tol usually needs more updates; frequent checks add synchronization.
        convergence : str
            Normal: pot/rel/rel_mean/step/kkt/hybrid. Log: pot/log/log_mean/step/kkt/hybrid.
            Change criteria track scalings; step checks rows; kkt checks both axes;
            hybrid screens rows first. The cap returns the last iterate without a flag.
        chunksize : positive int
            KeOps plan_extract candidate batch size (40000); larger uses more memory.
            This does not batch Sinkhorn.
        device, dtype : optional device, th.dtype
            None picks the CUDA device with most free memory, or CPU. Integers select
            CUDA indices. float64 improves precision with greater memory/runtime cost.
        verbose : bool
            Stored option; current solver loops do not print progress.
        
        Notes
        -----
        Default budgets are ones, not unit-sum histograms. First balance normalized
        geometry/features without FGW; then add small structure weights and tune eps/rho
        against actual cost and mass scales. Keep a positive first-order term. Automatic
        scales, option lists, and backend selection mutate during initialization; use
        a fresh instance for independent datasets when relying on automatic settings."""
        self.mode, self.reg, self.use_fgw, self.use_lowrank = mode, reg , use_fgw, use_lowrank
        # uot+fgw+log+lowrank+entropy/kl
        self.moment_max_dim = moment_max_dim
        self.ci, self.cf  = ci, cf
        self.init_sinkhorn(eps, rho, theta)
        self.normal_g = normal_g
        self.normal_f = normal_f
        self.sigma2_1st, self.tau2s_1st, self.sigma2_2nd, self.tau2s_2nd = sigma2_1st, tau2s_1st, sigma2_2nd, tau2s_2nd
        self.gscale_1st, self.gscale_2nd = gscale_1st, gscale_2nd
        self.fscales_1st, self.fscales_2nd = fscales_1st, fscales_2nd
        self.delta_h = delta_h
        self.use_keops = use_keops
        self.uv_iter, self.tol, self.check_every = uv_iter, tol, check_every
        self.convergence = convergence
        self.device = self.auto_select_device(device, verbose=False)
        self.dtype  = dtype
        self.verbose = verbose
        self.chunksize = chunksize
        self.record = record
        self.tiny = th.finfo(self.dtype).tiny
        self.scalar2vector = scalar2vector
        self.auto_select_device = auto_select_device
        self.n2v = n2v
        self.plan_extract = plan_extract


    def as_tensor(self, z, device=None, dtype=None):
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype
        return th.as_tensor(z, device=device, dtype=dtype).contiguous()

    def _view(self, z, axis):
        """Only the row/column representation differs between backends."""
        if self.use_keops:
            return self.LazyTensor(z.reshape(-1, 1), axis=axis)
        return z[:, None] if axis == 0 else z[None, :]

    def _dot(self, X, Y):
        """Dense GEMM versus implicit pairwise inner products."""
        if self.use_keops :
            return (self.LazyTensor(X, axis=0)*self.LazyTensor(Y, axis=1)).sum(-1)
        return X @ Y.T


    def init_model(self, X, Y, Fxs=None, Fys=None, Lx=None, Ly=None, 
                   a=None, b=None, u=None, v = None, mass = 1.0):
        """Convert inputs, prepare budgets, and initialize this call's state."""
        self.init_config()
        self.init_idata( X, Y, Fxs , Fys , Lx , Ly )
        self.init_keops(self.use_keops)
        self.X, self.Y = self.geom_normal(self.Xr, self.Yr, predict=False)
        self.XF, self.YF = self.feat_normal(self.XFr, self.YFr , predict=False)

        self.init_otparas(a, b, u, v, mass=mass)
        self.sigma2_1st = self.get_sigma2(self.X, self.Y, self.sigma2_1st)
        if self.sigma2_2nd is None:
            self.sigma2_2nd = self.sigma2_1st
        self.tau2s_1st = self.get_tau2s(self.XF, self.YF, self.tau2s_1st, self.normal_f)
        if self.tau2s_2nd is None:
            self.tau2s_2nd = self.tau2s_1st
        else:
            self.tau2s_2nd = self.get_tau2s(self.XF, self.YF, self.tau2s_2nd, self.normal_f)

    def init_sinkhorn(self, eps, rho, theta=None):
        if theta is None:
            rho_x, rho_y = self.scalar2vector(rho, 2, default=1.0)
            tx, ty = rho_x/(rho_x+eps), rho_y/(rho_y+eps)
        else:
            tx, ty = self.scalar2vector(theta, 2, default=1.0)
            for i in [tx, ty]:
                assert 0 <= i < 1.0, f'0 <= theta <= 1.0, but got {i}'
            rho_y = eps * ty / (1.0 - ty)
            rho_x = eps * tx / (1.0 - tx)
        self.tx, self.ty = tx, ty
        self.rho_x, self.rho_y = rho_x, rho_y
        self.eps = eps

    def init_config(self, ):
        self.normal_f = self.n2v( self.normal_f, 'ccos')

        self.sigma2_1st = self.n2v( self.sigma2_1st, None)
        self.sigma2_2nd = self.n2v( self.sigma2_2nd, None)
        self.tau2s_1st = self.n2v( self.tau2s_1st, None)
        self.tau2s_2nd = self.n2v( self.tau2s_2nd, None)
        self.fscales_1st = self.n2v( self.fscales_1st, 1.0)
        self.fscales_2nd = self.n2v( self.fscales_2nd, 0.0)
        self.gscale_1st = self.n2v( self.gscale_1st, 1.0)
        self.gscale_2nd = self.n2v( self.gscale_2nd, 1.0)
        self.use_keops = self.n2v( self.use_keops, 8e8)
        assert 0 <= self.delta_h < 1.0

    def init_otparas(self, a, b, u=None, v=None, mass = 1):
        # a = self.X.new_ones(self.nx) if a is None else self.as_tensor(a)
        # b = self.Y.new_ones(self.ny) if b is None else self.as_tensor(b)
        totals = self.as_tensor(mass).expand(2)
        if a is None:
            a = th.full((self.nx,), float(totals[0]) / self.nx, device=self.device, dtype=self.dtype)
        else:
            a = self.as_tensor(a)
        if b is None:
            b = th.full((self.ny,), float(totals[1]) / self.ny, device=self.device, dtype=self.dtype)
        else:
            b = self.as_tensor(b)

        u = self.X.new_ones(self.nx) if u is None else self.as_tensor(u)
        v = self.Y.new_ones(self.ny) if v is None else self.as_tensor(v)
        self.a, self.b = a, b
        self.u, self.v = u, v

        self.logr = self.logc = None
        self.loga, self.logb = a.clamp_min(self.tiny).log(), b.clamp_min(self.tiny).log()
        if self.mode == 'log':
            self.logu, self.logv = u.clamp_min(self.tiny).log(), v.clamp_min(self.tiny).log()
        else:
            self.logu = self.logv = None
        self.P = self.logK = self.K = None
        self.r = self.c = None
        self.D1st = self.D2nd = self.D = None
        self.H = self.logH = None
        self.K = self.logK =  None
        self._fgw_prediction = {}

    def init_keops(self, use_keops ):
        self.use_keops = use_keops_bool(use_keops, self.nx * self.ny)
        self.LazyTensor = None
        if self.use_keops:
            self.LazyTensor = LazyTensor

    def init_idata(self, X, Y, Fxs , Fys , Lx , Ly ):
        self.Xr, self.XFr, self.XL = self.check_input(X, Fxs, Lx)
        self.Yr, self.YFr, self.YL = self.check_input(Y, Fys, Ly)
        self.nx, self.ny, self.D = self.Xr.shape[0], self.Yr.shape[0], self.Xr.shape[1]

        self.FL = len(self.XFr)
        assert self.FL == len(self.YFr)

        self.LL = int(self.XL is not None and self.YL is not None)

        for i in ['normal_f', 'tau2s_1st', 'tau2s_2nd', 'fscales_1st', 'fscales_2nd']:
            ivalue = getattr(self, i)
            setattr(self, i, self.scalar2vector(ivalue, self.FL))

    def check_input(self, X, Fxs, Lx):
        if X is None:
            nx = 0
        else:
            X = self.as_tensor(X)
            nx = X.shape[0]

        if Fxs is None:
            Fxs = []
        XF = list(Fxs) if isinstance(Fxs, (list, tuple)) else [Fxs]
        XF = [self.as_tensor(F) for F in XF]
        for i in range(len(XF)):
            assert nx == XF[i].shape[0], f"Features {i} must have the same points"
        if Lx is None:
            XL = None
        else:
            XL = self.as_tensor(Lx)
            assert nx == XL.shape[0], f"Labels must have the same points"
        return X, XF, XL


    def geom_normal(self, *Xa, predict_ids=None, predict=False):
        """Normalize coordinates and return one matrix per input domain.
        
        Training stores g_shift/g_scale. With predict=True, predict_ids selects fitted
        domain statistics without updating them. 'each' separates centers/scales;
        True/'isoscale' shares the mean scale; 'global' shares joint statistics; 'X'
        shares the first domain's statistics. False/'pass' preserves coordinates."""
        Xa = [self.as_tensor(iX) for iX in Xa]
        L = len(Xa)
        if predict:
            assert (predict_ids is not None) and (len(predict_ids) == L)
        else:
            M, S = Xa[0].new_zeros((L, self.D)), Xa[0].new_ones(L)
            predict_ids = range(L)
            if self.normal_g in ['each', 'isoscale', True]:
                for l in range(L):
                    M[l], S[l] = centerlize(Xa[l])[1][:2]
                if self.normal_g in ['isoscale', True]:
                    S = th.mean(S).expand(L).clone()
            elif self.normal_g == 'global':
                iXm, iXs = centerlize(th.cat(Xa, 0))[1][:2]
                M = iXm.expand(L, -1).clone()
                S = iXs.expand(L).clone()
            elif self.normal_g == 'X':
                iXm, iXs = centerlize(Xa[0])[1][:2]
                M = iXm.expand(L, -1).clone()
                S = iXs.expand(L).clone()
            elif self.normal_g in [False, 'pass']:
                pass
            else:
                raise ValueError("Unknown normalization method: {}".format(self.normal_g))
            self.g_shift, self.g_scale = M, S

        Xn = []
        for l in range(L):
            pid = predict_ids[l]
            iX = centerlize(Xa[l], Xm=self.g_shift[pid], Xs=self.g_scale[pid])[0]
            Xn.append(iX.to(dtype=self.dtype))
        return Xn

    def feat_normal(self, *Fas, predict_ids=None, predict=False):
        """Normalize each modality, returning one feature list per input domain.
        
        Training stores reusable statistics in feat_scales[domain][modality]. Prediction
        uses predict_ids to reuse fitted centers/scales; sample-wise statistics must
        be recomputed by the imported normalization helpers for each new sample."""
        La = len(Fas)
        if self.FL == 0:
            return [[] for _ in range(La)]

        if predict:
            assert (predict_ids is not None) and (len(predict_ids) == La)
            feat_scales = self.feat_scales
        else:
            feat_scales = {n: {} for n in range(La)}
            self.feat_scales = feat_scales
            predict_ids = range(La)

        Fs = [[] for _ in range(La)]
        for iL in range(self.FL):
            inorm = self.normal_f[iL]
            if inorm in ['image', 'i2p', 'img'] and not predict:
                outs = image_normalize(*[self.as_tensor(F[iL]) for F in Fas])
                for n in range(La):
                    Fs[n].append(outs[n])
                    feat_scales[n][iL] = outs[-1]
                continue
            for n in range(La):
                pid = predict_ids[n]
                stats = feat_scales.get(pid, {}).get(iL, [])
                iF = self.as_tensor(Fas[n][iL])
                if inorm in ['image', 'i2p', 'img']:
                    outs = image_normalize(iF, Xm=stats[0], Xs=stats[1])
                elif inorm in ['cosine', 'cos']:
                    outs =  normalize(iF, *stats)
                elif inorm == 'zc':
                    outs = centerlize(iF, *stats)
                elif inorm in ['ccosine', 'ccos']:
                    outs = center_normalize(iF, *stats)
                elif inorm == 'pcc':
                    outs = scale_normalize(iF)
                elif inorm in ['zscore', 'std']:
                    outs = scaling(iF, *stats, anis_var=True,)
                elif inorm == 'label':
                    outs = [ iF * 2 , stats]
                elif inorm in ['pass', 'restc', 'resta']:
                    outs = [ iF  , stats]
                else:
                    warnings.warn(f"Unknown feature normalization method: {inorm}")
                    outs = [iF, stats]
                iF, stats = outs
                Fs[n].append(iF)
                if not predict:
                    # if inorm in ['cosine', 'cos', 'pcc']:
                    #     stats = []
                    # elif inorm in ['ccosine', 'ccos']:
                    #     stats = stats[:1]
                    # elif inorm == 'zc':
                    #     stats = stats[:2]
                    feat_scales[pid][iL] = stats
        return Fs

    def get_sigma2(self, X, Y, sigma2):
        if sigma2 is None:
            sigma2 = self.as_tensor(sigma_square(X, Y)).clamp_min(self.tiny)
        return self.as_tensor(sigma2)

    def get_tau2s(self,  Fxs, Fys, tau2s, normal_f):
        assert type(Fxs) is list
        Nf = len(Fxs)
        tau2s = list(self.scalar2vector(tau2s, Nf))
        for i in range(Nf):
            if tau2s[i] is None:
                if normal_f[i] in ['cos', 'cosine', 'ccosine', 'ccos', 'restc']:
                    tau2s[i] = self.as_tensor(sigma_square_cos(Fxs[i], Fys[i])).clamp_min(self.tiny)
                elif normal_f[i] in ['label']:
                    tau2s[i] = 1.0
                else:
                    tau2s[i] = self.as_tensor(sigma_square(Fxs[i], Fys[i])).clamp_min(self.tiny)
        return tau2s

    def _factors(self, Z):
        """Return exact squared-distance factors phi=[q,1,Z], psi=[1,q,-2Z].
        
        For Z shaped (n,d), q=sum(Z**2,dim=1) and phi@psi.T equals dist2(Z,Z).
        The width d+2 is algebraically exact, not a sampled approximation."""
        q = Z.square().sum(1, keepdim=True)
        one = th.ones_like(q)
        return th.cat((q, one, Z), 1), th.cat((one, q, -2*Z), 1)

    def _moment(self, Z, phi, psi, mass):
        """Return sum_k C[i,k]**2*mass[k] for squared-distance matrix C.
        
        Use a (d+2)-square factor moment when d<=moment_max_dim, otherwise direct
        distance reductions. These are fourth powers of Euclidean distances."""
        if Z.shape[1] <= self.moment_max_dim:
            return ((phi @ (psi.T @ (mass[:, None]*psi)))*phi).sum(1)
        return (dist2(Z, Z, self.use_keops).square() @ mass).reshape(-1)

    def D2_fgw_gradient(self, X, Y, P, r, c, sigma2_2nd, centerlize=True):
        """Return the FGW gradient at P with its actual marginals r=P@1, c=P.T@1.
        
        Cx/Cy are within-set squared distances divided by sigma2_2nd. The gradient is
        Cx.square()@r + (Cy.square()@c).T - 2*Cx@P@Cy.T for the half-sum FGW energy.
        Optional centering preserves exact distances. Factors are exact; KeOps also
        uses factors for the cross term when direct moments are selected. A larger
        sigma2_2nd weakens the gradient quadratically. _fgw_last stores reference-plan
        coefficients (cross,dx,dy) for prediction and is overwritten on each call."""
        mx = X.mean(0) if centerlize else X.new_zeros(X.shape[1])
        my = Y.mean(0) if centerlize else Y.new_zeros(Y.shape[1])
        X = (X - mx) / (sigma2_2nd**0.5)
        Y = (Y - my) / (sigma2_2nd**0.5)
        phix, psix = self._factors(X)
        phiy, psiy = self._factors(Y)
        cross = psix.T @ (P @ psiy)
        if self.use_lowrank:
            left, right = phix @ cross, phiy
            dx, dy = self._moment(X, phix, psix, r), self._moment(Y, phiy, psiy, c)
        else:
            Cx, Cy = dist2(X, X,  self.use_keops), dist2(Y, Y,  self.use_keops)
            dx = (Cx.square() @ r).reshape(-1)
            dy = (Cy.square() @ c).reshape(-1)
            if self.use_keops:
                # Direct moments remain; exact factors avoid Lazy @ Lazy.
                left, right = phix @ cross, phiy
            else:
                left, right = Cx @ P, Cy
        # Only the reference-plan coefficients are needed by kernel prediction.
        self._fgw_last = (cross, dx, dy)
        return self._view(dx, 0) + self._view(dy, 1) - 2 * self._dot(left, right)

    def D1_cost_agg(self, X, Y, sigma2_1st, gscale_1st, Fxs, Fys, tau2s_1st, fscales_1st):
        D1st = 0
        if gscale_1st >0:
            D1st = D1st + (gscale_1st/sigma2_1st) * dist2(X, Y,  self.use_keops)
        assert type(Fxs) is list
        Nf = len(Fxs)
        for i in range(Nf):
            if fscales_1st[i] > 0:
                D1st = D1st + (fscales_1st[i]/tau2s_1st[i]) * dist2(Fxs[i], Fys[i],  self.use_keops)
        return D1st

    def D2_cost_agg(self,  P, r, c, X, Y, sigma2_2nd, gscale_2nd,
                    Fxs, Fys, tau2s_2nd, fscales_2nd):
        """Sum positive geometry/feature FGW weights at the same reference (P,r,c).
        
        sigma2_2nd/tau2s_2nd scale structural distances; their effects are quadratic.
        Nonpositive weights omit terms; no active terms returns scalar zero.
        Store each modality's reference coefficients in _fgw_prediction for prediction."""
        D2nd = 0
        self._fgw_prediction = {}
        if gscale_2nd > 0:
            D2nd = D2nd + (gscale_2nd) * self.D2_fgw_gradient(X, Y, P, r, c, sigma2_2nd, centerlize=True)
            self._fgw_prediction[-1] = self._fgw_last
        assert type(Fxs) is list
        Nf = len(Fxs)
        for i in range(Nf):
            if fscales_2nd[i] > 0:
                D2nd = D2nd + fscales_2nd[i] * self.D2_fgw_gradient(Fxs[i], Fys[i], P, r, c, tau2s_2nd[i], centerlize=True)
                self._fgw_prediction[i] = self._fgw_last
        return D2nd

    def D_effective(self, 
                    X, Y,  sigma2_1st, sigma2_2nd, gscale_1st, gscale_2nd,
                    XF, YF, tau2s_1st, tau2s_2nd, fscales_1st, fscales_2nd,
                    Prc = None, use_cache=True):
        if not use_cache:
            self.D1st, self.D2nd = None, None
            self._fgw_prediction = {}

        if (self.D1st is None):
            self.D1st = self.D1_cost_agg(X, Y, sigma2_1st, gscale_1st,
                                         XF, YF, tau2s_1st, fscales_1st)
        Deff = self.D1st
        if (self.D2nd is None) and (self.use_fgw) and (Prc is not None):
            P, r, c = Prc
            self.D2nd = self.D2_cost_agg( P, r, c,
                                          X,  Y,  sigma2_2nd, gscale_2nd,
                                          XF, YF, tau2s_2nd, fscales_2nd)
        if self.use_fgw and self.D2nd is not None:
            Deff = Deff + self.D2nd
        # if self.record:
        #     self.D = Deff
        return Deff

    def K_effective(self, D, eps, XL=None, YL=None, delta_h=0, use_cache=True):
        """Return exp(-D/eps) with label and regularization-reference factors.
        
        Both XL/YL enable H=(1-delta_h)*XL@YL.T+delta_h; reg='kl' also multiplies
        by a[:,None]*b[None,:]. Log mode returns additive log terms and requires
        initialized log budgets. eps must match the Sinkhorn relaxation exponents;
        small eps increases cost discrimination and normal-domain numerical difficulty."""
        use_label = (XL is not None) and (YL is not None)
        # if not use_cache:
        #     self.H  = None
    
        if self.mode=="log":
            logK = -D/ eps
            if use_label:
                logH = self.H_label_log(XL, YL, delta_h)
                logK = logK + logH
            if self.reg == "kl":
                logK = logK + self._view(self.loga, 0)+self._view(self.logb, 1)
            if self.record:
                self.logK = logK
            return logK
        else:
            K = (-D/ eps).exp()
            if use_label:
                H = self.H_label(XL, YL, delta_h)
                K = K * H
            if self.reg == "kl":
                K = K * self._view(self.a, 0) * self._view(self.b, 1)
            if self.record:
                self.K = K
            return K

    def H_label(self, Lx, Ly, delta_h, S_lable=None):
        """Return H=(1-delta_h)*Lx@(Ly@S_lable).T+delta_h.
        
        Lx/Ly are nonnegative labels; without S_lable they share a label dimension.
        S_lable has shape (Ly.shape[1],Lx.shape[1]). Inputs are not normalized here.
        For row-normalized labels, larger delta_h weakens discrimination. At zero
        floor, zero label overlap gives exact zero compatibility in normal mode."""
        dh = float(delta_h)
        lh = 1.0 - dh
        use_dh = bool(dh >0)
        A, B = Lx, Ly
        if not S_lable is None:
            B = B @ S_lable
        if self.use_keops :
            H = (self.LazyTensor(A, axis=0) * self.LazyTensor(B, axis=1)).sum(-1)
            # A_i = LazyTensor(A.contiguous()[:, None, :])
            # B_j = LazyTensor(B.contiguous()[None, :, :])
            # H = (A_i * B_j).sum(-1)
            if use_dh:
                H = H * A.new_tensor(lh) + dh
        else:
            if use_dh:
                H = (lh* A) @ B.T
                H.add_(dh)
                # H = lh * H + dh
            else:
                H = A @ B.T
        return H

    def H_label_log(self, Lx, Ly, delta_h, S_lable=None, faster=False):
        """Return log((1-dh) * Lx @ (Ly @ S).T + dh), A,B >= 0."""
        dh = float(delta_h)
        assert 0 <= dh < 1.0
        use_dh = bool(dh >0)
        lh = 1.0 - dh
        A, B = Lx, Ly
        if S_lable is not None:
            B = B @ S_lable
        tiny = th.finfo(A.dtype).tiny
        logA = A.clamp_min(tiny).log()
        logB = B.clamp_min(tiny).log()
        if use_dh:
            half_lh = 0.5 * math.log(lh)
            half_dh = 0.5 * math.log(dh)
            logA = th.cat([logA + half_lh, logA.new_full((A.shape[0], 1), half_dh)], dim=1)
            logB = th.cat([logB + half_lh, logB.new_full((B.shape[0], 1), half_dh)], dim=1)
        if self.use_keops :
            A_i = self.LazyTensor(logA.contiguous()[:, None, :])   # (n_x, 1, K+1)
            B_j = self.LazyTensor(logB.contiguous()[None, :, :])   # (1, n_y, K+1)
            
            if faster:
                return (A_i + B_j).logsumexp(dim=-1) 
            else:
                Z = A_i + B_j
                shift = Z.max(dim=-1)
                return shift + (Z - shift).exp().sum(dim=-1).log()
        return th.logsumexp(logA[:, None, :] + logB[None, :, :], dim=-1)

    def H_label_log0(self, Lx, Ly, delta_h, S_lable=None, faster=False):
        """Alternative log label compatibility with an explicit mixture-floor update.
        
        S_lable maps Ly into Lx's label space; zero components are clamped to dtype tiny."""
        dh = float(delta_h)
        lh = 1.0 - dh
        use_dh = bool(dh >0)
        A, B = Lx, Ly
        if S_lable is not None:
            B = B @ S_lable
        tiny = th.finfo(A.dtype).tiny
        logA = A.clamp_min(tiny).log()
        logB = B.clamp_min(tiny).log()
        if self.use_keops :
            A_i = self.LazyTensor(logA.contiguous()[:, None, :])   # (n_x, 1, K)
            B_j = self.LazyTensor(logB.contiguous()[None, :, :])   # (1, n_y, K)

            if faster:
                log_H = (A_i + B_j).logsumexp(dim=-1) #Faster
            else:
                Z = A_i + B_j
                shift = Z.max(dim=-1)
                log_H = shift + (Z - shift).exp().sum(dim=-1).log()

            if use_dh:
                logdh = math.log(dh)
                d = log_H + math.log(lh) - logdh
                log_H = logdh + d.relu() + (1.0 + (-d.abs()).exp()).log()
        else:
            log_H = th.logsumexp( logA[:, None, :] + logB[None, :, :], dim=-1 )
            if use_dh:
                log_H = th.logaddexp(log_H + math.log(lh),
                                    log_H.new_tensor(math.log(dh)))
        return log_H

    def sinkhorn_normal(self, K, u, v, a, b, tx, ty,
                        convergence="hybrid", check_every=10):
        """Solve fixed-kernel UOT with multiplicative Sinkhorn updates.
        
        K includes labels/reference factors; a,b are positive budgets; u,v are initial
        scalings (None uses ones). tx/ty=rho_x/y/(rho_x/y+eps). Updates use
        u=(a/(K@v))**tx, v=(b/(K.T@u))**ty with denominator floors.
        Return (u,v,r,c), where r/c are the plan marginals. Uses self.uv_iter/tol and
        checks every check_every updates. Criteria: pot/rel/rel_mean/step/kkt/hybrid.
        The iteration cap returns the last iterate; extreme kernels can under/overflow."""
        if u is None:
            u = th.ones_like(a)
        if v is None:
            v = th.ones_like(b)
        KT, tiny = K.T, self.tiny
        Kv = (K @ v).reshape(-1)
        for k in range(1, self.uv_iter + 1):
            u0, v0 = u, v
            u = (a / Kv.clamp_min(tiny)).pow(tx)
            # u = th.where(Kv > 0, (a / Kv.clamp_min(tiny)).pow(tx), 1.)
            Ktu = (KT @ u).reshape(-1)
            v = (b / Ktu.clamp_min(tiny)).pow(ty)
            # v = th.where(Ktu > 0, (b / Ktu.clamp_min(tiny)).pow(ty), 1.)
            Kv = (K @ v).reshape(-1)
            if k % check_every == 0 or k == self.uv_iter:
                if self._uv_conver_normal(
                    u, v, u0, v0, Kv, Ktu,
                    a, b, tx, ty, convergence,
                ):
                    break
        r = u*Kv
        c = v*Ktu
        return u, v, r, c

    def _uv_conver_normal(self, u, v, u0, v0, Kv, Ktu,
                        a, b, tx, ty, convergence="hybrid"):
        """Check scaling changes or normal-domain fixed-point residuals against tol.
        
        pot uses scaled changes; rel/rel_mean use maximum/mean relative changes.
        step checks rows; kkt checks both axes; hybrid checks rows before columns.
        Kv/Ktu are current reductions. Raw fixed-point products require nonzero support."""
        tiny = self.tiny
        if convergence == "pot":
            errors = []
            for z, z0 in ((u, u0), (v, v0)):
                denom = th.maximum(z.amax(), z0.amax()).clamp_min(1.)
                errors.append((z - z0).abs().amax() / denom)
            return (0.5 * (errors[0] + errors[1])).item() <= self.tol
        if convergence in ("rel", "rel_mean"):
            du = (u - u0).abs() / th.maximum(u, u0).clamp_min(tiny)
            dv = (v - v0).abs() / th.maximum(v, v0).clamp_min(tiny)
            error = ( th.maximum(du.amax(), dv.amax())
                     if convergence == "rel"
                     else 0.5 * (du.mean() + dv.mean()) )
            return error.item() <= self.tol
        if convergence not in ("step", "kkt", "hybrid"):
            raise ValueError(f"Unknown convergence method: {convergence}")
        # Check the original scaling equations using raw products.
        # nextu = (a / Kv).pow(tx)
        nextu = th.where(Kv > 0, (a / Kv.clamp_min(tiny)).pow(tx), 1.)
        err_u = ( (u - nextu).abs() / th.maximum(u, nextu).clamp_min(tiny) ).amax()
        if convergence == "step":
            return err_u.item() <= self.tol
        if convergence == "hybrid":
            if not (err_u.item() <= self.tol):
                return False
        # nextv = (b / Ktu).pow(ty)
        nextv = th.where(Ktu > 0, (b / Ktu.clamp_min(tiny)).pow(ty), 1.)
        err_v = ( (v - nextv).abs() / th.maximum(v, nextv).clamp_min(tiny) ).amax()
        if convergence == "hybrid":
            return err_v.item() <= self.tol
        return th.maximum(err_u, err_v).item() <= self.tol

    def sinkhorn_log(self, logK, logu, logv, loga, logb, tx, ty,
                    convergence="hybrid", check_every=10):
        """Solve fixed-log-kernel UOT with stable log-sum-exp Sinkhorn updates.
        
        logK includes label/reference terms; loga/logb are log budgets. Initial
        logu/logv=None uses zeros. tx/ty=rho_x/y/(rho_x/y+eps).
        Return (logu,logv,logr,logc). Uses self.uv_iter/tol and check_every;
        criteria are pot/log/log_mean/step/kkt/hybrid. Residual checks use instance
        eps/rho, which must agree with tx/ty. The cap returns the last iterate;
        later exponentiation can still exceed the ordinary domain's numerical range."""
        if logu is None:
            logu = th.zeros_like(loga)
        if logv is None:
            logv = th.zeros_like(logb)
        Kv = (logK + self._view(logv, 1)).logsumexp(dim=1).reshape(-1)
        for k in range(1, self.uv_iter + 1):
            # Keep references: subsequent updates create new tensors.
            logu0, logv0 = logu, logv
            logu = tx * (loga - Kv)
            Ktu = (logK + self._view(logu, 0)).logsumexp(dim=0).reshape(-1)
            logv = ty * (logb - Ktu)
            Kv = (logK + self._view(logv, 1)).logsumexp(dim=1).reshape(-1)
            if k % check_every == 0 or k == self.uv_iter:
                if self._uv_conver_log(
                    logu, logv, logu0, logv0,
                    Kv, Ktu, loga, logb, convergence,
                ):
                    break
        logr = logu+Kv
        logc = logv+Ktu
        return logu, logv, logr, logc

    def _uv_conver_log(self, logu, logv, logu0, logv0,
                    Kv, Ktu, loga, logb, convergence="hybrid"):
        """Check log changes or scaled UOT residuals against self.tol.
        
        pot checks rescaled ordinary changes; log/log_mean use maximum/mean log changes.
        step checks rows; kkt checks the row-plus-column residual extrema; hybrid first
        screens rows. Residuals use instance eps/rho and divide by max(1,eps,rho_x,rho_y)."""
        if convergence == "pot":
            errors = []
            for z, z0 in ((logu, logu0), (logv, logv0)):
                shift = th.maximum(z.amax(), z0.amax()).clamp_min(0.)
                errors.append( ((z - shift).exp() - (z0 - shift).exp()).abs().amax() )
            error = 0.5 * (errors[0] + errors[1])
        elif convergence in ("log", "log_mean"):
            du = (logu - logu0).abs()
            dv = (logv - logv0).abs()
            error = (
                th.maximum(du.amax(), dv.amax())
                if convergence == "log"
                else 0.5 * (du.mean() + dv.mean())
            )
        elif convergence in ("step", "kkt", "hybrid"):
            scale = max(1., self.eps, self.rho_x, self.rho_y)
            ex = self.eps * logu + self.rho_x * (logu + Kv - loga)
            # Step: check the row fixed-point residual directly.
            if convergence == "step":
                return (ex.abs().amax() / scale).item() <= self.tol
            # Hybrid: reject first; confirm candidates with full KKT.
            if convergence == "hybrid":
                if (ex.abs().amax() / scale).item() > self.tol:
                    return False
            # Both kkt and accepted hybrid candidates reach this block.
            ey = self.eps * logv + self.rho_y * (logv + Ktu - logb)
            # The primal gradient is ex_i + ey_j; its extrema give the sup norm.
            error = th.maximum(
                (ex.amax() + ey.amax()).abs(),
                (ex.amin() + ey.amin()).abs(),
            ) / scale
        else:
            raise ValueError(f"Unknown convergence method: {convergence}")
        return error.item() <= self.tol

    def plan(self, K, u, v, rowi=None, colj=None):
        """Return the plan, or [rowi, colj, Pij] for paired indices."""
        if self.mode == "log":
            P = (K + self._view(u, 0) + self._view(v, 1)).exp()
        else:
            P = self._view(u, 0) * K * self._view(v, 1)

        if (rowi is not None) and (colj is not None):
            P = self.plan_extract( P, rowi, colj, chunksize=getattr(self, "chunksize", 30000))
        return P

    def uv_sinkhorn(self, K,):
        if self.mode == "log":
            result = self.sinkhorn_log(K, self.logu, self.logv, self.loga, self.logb, self.tx, self.ty,
                                       self.convergence, self.check_every)
            logu, logv, logr, logc = result
            u, v, r, c = (z.exp() for z in result)
            P = self.plan(K, logu, logv)
        else:
            result = self.sinkhorn_normal(K, self.u, self.v, self.a, self.b,
                                          self.tx, self.ty, self.convergence, self.check_every)
            u, v, r, c = result
            logu, logv, logr, logc = (z.log() for z in result)
            P = self.plan(K, u, v)
        self.u, self.v, self.r, self.c = u, v, r, c
        self.logu, self.logv, self.logr, self.logc = logu, logv, logr, logc
        self.P = P
        return P, r, c

    def solve(self, Prc=None):
        """Solve one UOT subproblem using initialized inputs and cached effective costs.
        
        Prc=(P0,r0,c0) supplies an FGW reference when the structural cache is empty;
        r0/c0 must be its marginals. Return (P,r,c). Call init_model first. Changing
        Prc alone does not refresh an existing D2nd; invalidate effective-cost caches
        explicitly when changing the reference within the same initialized state."""
        D = self.D_effective(
                self.X, self.Y,  self.sigma2_1st, self.sigma2_2nd, self.gscale_1st, self.gscale_2nd,
                self.XF, self.YF, self.tau2s_1st, self.tau2s_2nd, self.fscales_1st, self.fscales_2nd,
                Prc = Prc, use_cache=True)
        K = self.K_effective(D, eps=self.eps, XL=self.XL, YL=self.YL, delta_h=self.delta_h)
        Prc = self.uv_sinkhorn(K)
        return Prc

    @th.no_grad()
    def solver(self, X, Y, Fxs=None, Fys=None, Lx=None, Ly=None, *, 
               a=None, b=None, u=None, v=None, Prc=None, mass=1.0):
        """Fit (P,r,c) from X(nx,d), Y(ny,d), optional features and soft labels.
        
        Fxs/Fys are matrices or paired modality lists with matching feature dimensions;
        Lx/Ly share a nonnegative label space. Coordinates use normal_g, features normal_f.
        a/b are positive marginal budgets; None uses ones, not unit-sum histograms.
        u/v are ordinary initial scalings even in log mode. Prc=(P0,r0,c0) optionally
        supplies a compatible FGW reference and skips the preliminary UOT solve.
        With use_fgw and no reference, solve UOT then one corrected UOT subproblem;
        the FGW gradient stays fixed within each solve. Return transported mass P
        (dense or implicit), r=P@1, c=P.T@1. Runs without gradients; no edge decoding
        or converged outer FGW optimum is returned. Automatic settings mutate in place."""
        self.init_model( X, Y, Fxs=Fxs, Fys=Fys, Lx=Lx, Ly=Ly, a=a, b=b, u=u, v = v, mass=mass)
        PrcNew = self.solve(Prc)
        if (Prc is None) and (self.use_fgw):
            PrcNew = self.solve(PrcNew)
        return PrcNew

    @th.no_grad()
    def transform(self, X=None, Fxs=None, Lx=None, *, inverse=False,
                  method="kernel", k=8, w_eps=None, batch_size=1024,):
        """Predict opposite-domain coordinates using fitted costs, without refitting OT.
        
        X/Fxs/Lx are query coordinates/features/labels; provide fitted modalities and
        labels when enabled. X=None returns fitted barycenters. inverse=True maps Y->X;
        the reverse is not guaranteed to be a mathematical inverse.
        method='kernel' uses normalized conditionals with opposite-side Sinkhorn scales,
        multimodal costs, labels and fixed FGW structure. 'nearest' copies the most
        similar same-side training displacement; 'knn' softmax-interpolates k displacements.
        Neighbors use geometry/features/labels; FGW affects their fitted displacements.
        w_eps sets KNN temperature (None uses eps); kernel uses fitted eps. batch_size
        limits query batches. Return coordinates in original units, not a newly optimized
        global plan; query normalization reuses fitted domain statistics."""
        if getattr(self, "P", None) is None:
            raise RuntimeError("Call solver before prediction.")

        if method not in ("kernel", "nearest", "knn"):
            raise ValueError("method must be 'kernel', 'nearest', or 'knn'.")

        if batch_size < 1 or k < 1:
            raise ValueError("batch_size and k must be positive.")

        temp = self.eps if w_eps is None else w_eps
        assert temp > 0, "temperature must be positive."

        X, Fxs, Lx = self.check_input(X, Fxs, Lx)
        side = int(inverse)
        anchor_side = 1 - side if method == "kernel" else side

        if inverse:
            source, target = (self.Yr, self.Xr)
            P, mass = (self.P.T, self.c) 
        else:
            source, target = (self.Xr, self.Yr)
            P, mass = (self.P, self.r)

        if X is None:
            mapped = (P @ target) / mass[:, None].clamp_min(self.tiny)
            return th.where(mass[:, None] > self.tiny, mapped, source)

        if len(Fxs) == 0 and Lx is None and th.equal(X, source):
            return self.transform(inverse=inverse)

        assert len(Fxs) == self.FL, "Query features must match fitted modalities."
        Gq = self.geom_normal(X, predict_ids=[side], predict=True)[0]
        Fq = self.feat_normal(Fxs, predict_ids=[side], predict=True)[0]
        Lq = Lx

        if self.LL and Lq is None:
            raise ValueError("Lx is required because fitted label costs are enabled.")

        Gr = (self.X, self.Y)[anchor_side]
        Fr = (self.XF, self.YF)[anchor_side]
        Lr = (self.XL, self.YL)[anchor_side] if self.LL else None
        if method == "kernel":
            offset = (self.logu, self.logv)[anchor_side]
            if self.reg == "kl":
                offset = offset + (self.loga, self.logb)[anchor_side]
        Recenter = target if method == "kernel" else (self.transform(inverse=inverse) - source)

        # Kernel costs compare opposite domains. Neighbors compare the query
        # to its own domain; their displacements already contain the fitted FGW effect.
        use_fgw = self.use_fgw and method == "kernel"
        Tpre = th.empty_like(X)
        for start in range(0, X.shape[0], batch_size):
            sl = slice(start, start + batch_size)
            igq, ifq = Gq[sl], [F[sl] for F in Fq]
            ilq = None if Lq is None else Lq[sl]

            D = self._prediction_cost(igq, Gr, ifq, Fr, ilq, Lr, side, use_fgw=use_fgw)
            if method == "kernel":
                logw = -D / self.eps + self._view(offset, 1)
                if self.use_keops:
                    Tpre[sl] = logw.sumsoftmaxweight( self.LazyTensor(Recenter[None, :, :]), dim=1 )
                else:
                    Tpre[sl] = th.softmax(logw, dim=1) @ Recenter
            else:
                nk = 1 if method == "nearest" else min(k, Gr.shape[0])
                if self.use_keops:
                    d, idx = D.Kmin_argKmin(nk, dim=1)
                    d, idx = d.reshape(-1, nk), idx.reshape(-1, nk).long()
                else:
                    d, idx = th.topk(D, nk, dim=1, largest=False, sorted=True)
                w = th.softmax(-(d - d[:, :1]) / temp, dim=1) if nk > 1 else th.ones_like(d)
                Tpre[sl] = X[sl] + (w[:, :, None] * Recenter[idx]).sum(1)
        return Tpre

    def _prediction_cost(self, Q, R, Fq, Fr, Lq, Lr, side, use_fgw=False):
        """Query costs; FGW row constants are omitted for kernel softmax only."""
        D = self.D1_cost_agg(Q, R, self.sigma2_1st, self.gscale_1st,
                            Fq, Fr, self.tau2s_1st, self.fscales_1st)
        if use_fgw:
            if self.gscale_2nd > 0:
                G = self._prediction_fgw(Q, R, self.X, self.Y,
                                         self.sigma2_2nd, self._fgw_prediction[-1], side)
                D = D + self.gscale_2nd * G
            for i in range(self.FL):
                if self.fscales_2nd[i] > 0:
                    G = self._prediction_fgw(Fq[i], Fr[i], self.XF[i], self.YF[i],
                                             self.tau2s_2nd[i], self._fgw_prediction[i], side)
                    D = D + self.fscales_2nd[i] * G
        if Lq is not None and Lr is not None:
            D = D - self.eps * self.H_label_log(Lq, Lr, self.delta_h)
        return D

    def _prediction_fgw(self, Q, R, X, Y, scale, state, side):
        """Extend fixed FGW reference coefficients, omitting query-row constants.
        
        state=(B,dx,dy), B=psiX.T@P0@psiY. Reuse fitted centering and distance scale;
        inverse prediction transposes B. Omitted query moments cancel in row softmax;
        fitting retains both moments in the full structural gradient."""
        B, dx, dy = state
        if side == 0:
            other = dy
        else:
            X, Y = Y, X
            B, other = B.T, dx
        Q = (Q - X.mean(0)) / scale**0.5
        R = (R - Y.mean(0)) / scale**0.5
        phiq, _ = self._factors(Q)
        phir, _ = self._factors(R)
        return self._view(other, 1) - 2 * self._dot(phiq @ B, phir)


@th.no_grad()
def matching(X, Y=None, Fxs=None, Fys=None, Lx=None, Ly=None, *, 
             otm=None, ot_options=None, a=None, b=None, mass=1.0,
             model=None, use_keops=int(1e7), method="greedy", sx=1, sy=1,
             topk=1, assignment_min_score=0.0, k=None, candidate_space="cost",
             score_threshold=0.05, pe_ratio=0.05, score_rule="either", device=None, 
             dtype=th.float32):
    """Run the three stages; return [S, Psp, Qxy, Qyx, eligible, r, c, model].

    S/Psp/Qxy/Qyx contain selected edges; eligible is the sparse pre-decode
    boundary mask. cost candidates use D1; D candidates include FGW.
    """
    
    if candidate_space in ("cost", "D"):
        record = True
    else:
        record = False

    model = OTM_solver(X, Y, Fxs, Fys, Lx, Ly, 
             ot_options=ot_options, a=a, b=b, mass=mass,
             use_keops=use_keops, model=model, record=record,
               device=device, dtype=dtype)

    if candidate_space in ("cost", "D"):
        cost = -(model.logK if model.mode == 'log' else model.K)
    else:
        cost = None

    Inds, psp, qxy, qyx, score = filter_edges(
        model.P, model.r, model.c, model.a, model.b, D=cost, 
        X=model.X, Y=model.Y, use_keops=model.use_keops,
        k=None, candidate_space=candidate_space, 
        score_threshold=score_threshold,
        pe_ratio=pe_ratio, score_rule=score_rule,
        )
    keep = decode(Inds, psp, qxy, qyx, score, model.nx, model.ny, 
            method=method, sx=sx, sy=sx,
            topk=topk, assignment_min_score=assignment_min_score)

    return { "Inds": Inds[keep], "psp": psp[keep], 
            "qxy": qxy[keep], "qyx": qyx[keep],
            "score": score[keep], "model": model }



# def matching_spaces(Xs, Fs=None, R=None, Ls=None, *, otm=None, sx=1, sy=1,
#                     method="greedy", mode=None, use_keops=None,
#                     ot_options=None, match_options=None):
#     """Return [space_pairs, results] for R's directed pairs or adjacent spaces."""
#     if R is None:
#         pairs = [[s, s + 1] for s in range(len(Xs) - 1)]
#     else:
#         pairs = th.as_tensor(R, dtype=th.bool).nonzero().cpu().tolist()
#         pairs = [[s, t] for s, t in pairs if s != t]
#     options = dict(otm=otm, sx=sx, sy=sy, method=method, mode=mode,
#                    use_keops=use_keops, ot_options=ot_options)
#     options.update(match_options or {})
#     results = [matching(Xs[s], Xs[t],
#                         None if Fs is None else Fs[s], None if Fs is None else Fs[t],
#                         None if Ls is None else Ls[s], None if Ls is None else Ls[t],
#                         **options) for s, t in pairs]
#     return [pairs, results]


@th.no_grad()
def OTM_solver(X=None, Y=None, Fxs=None, Fys=None, Lx=None, Ly=None, *,
             ot_options=None, a=None, b=None, mass=1.0,
             use_keops=None, model=None, record=False,
             device=None, dtype=th.float32):
    """Return [P, logP, D1, D, cost, r, c, a, b, model].

    P/D1/D/cost keep their dense or KeOps form. D includes the FGW correction;
    cost also includes labels. logP=P.log(); no transport/kernel is rebuilt.
    Missing budgets have unit total; explicit budgets are unchanged.
    """
    if  model is None:
        options = dict(reg="kl", mode="log", use_keops=False, use_lowrank=False,
                       normal_g=False, normal_f="pass", use_fgw=False,
                       eps=0.05, rho=1.0, delta_h=0.1)
        options.update(ot_options or {})
        options["use_keops"] = use_keops
        options['device'] = device
        options['dtype'] = dtype
        model = OTM(**options, record=record)
        model.solver(X, Y, Fxs=Fxs, Fys=Fys, Lx=Lx, Ly=Ly, a=a, b=b, mass=mass)
    return model


def filter_edges(P, r, c, a, b, *, D=None,  X=None, Y=None,
                 k=None, candidate_space="cost", 
                 score_threshold=0.05,
                 pe_ratio=0.05, score_rule="either",
                use_keops=True):
    nx, ny = r.numel(), c.numel()
    if k is None:
        ids = th.arange(nx * ny, device=r.device)
    else:
        if candidate_space == "geometry":
            assert X is not None and Y is not None
            rank = dist2(X, Y, use_keops)
        elif candidate_space in ("cost", "D"):
            assert D is not None
            rank = D
        elif candidate_space == "P":
            rank = -P
        else:
            raise ValueError("candidate_space must be geometry, cost, D, P.")

        kx, ky = min(k, ny), min(k, nx)
        if th.is_tensor(rank):
            jx = rank.topk(kx, dim=1, largest=False).indices
            iy = rank.T.topk(ky, dim=1, largest=False).indices
        else:
            jx = rank.argKmin(kx, dim=1).reshape(nx, kx)
            iy = rank.argKmin(ky, dim=0).reshape(ny, ky)
        forward = th.arange(nx, device=r.device)[:, None] * ny + jx
        reverse = iy * ny + th.arange(ny, device=r.device)[:, None]
        ids = th.cat((forward.flatten(), reverse.flatten())).unique(sorted=True)

    i, j = ids // ny, ids % ny
    if th.is_tensor(P):
        p = P[i, j]
    else:
        p = plan_extract(P, i, j)[2]
    qxy = th.where(r[i] > 0, p / r[i], 0)
    qyx = th.where(c[j] > 0, p / c[j], 0)

    score = qxy.sqrt() * qyx.sqrt()
    if score_rule == "either":
        confidence = th.maximum(qxy, qyx)
    elif score_rule == "both":
        confidence = th.minimum(qxy, qyx)
    else:
        raise ValueError("probability_rule must be either or both.")
    keep = ((p > 0) 
            & (confidence >= score_threshold)
            & (r[i] / a[i] >= pe_ratio) 
            & (c[j] / b[j] >= pe_ratio))
    indices = th.stack((i[keep], j[keep]))

    return [ indices, p[keep], qxy[keep], qyx[keep], score[keep] ]


def decode(indices, p, Qxy, Qyx, score, nx, ny, *, method="greedy", sx=1, sy=1,
           topk=None, assignment_min_score=0.0):
    knn = n2v(topk, max(sx, sy))
    i, j = indices
    count = p.numel()
    keep = th.zeros(count, device=p.device, dtype=th.bool)

    if count == 0:
        keep = th.zeros_like(keep)
    elif method == "soft":
        keep = th.ones_like(keep)
    elif method == "mutual":
        base = p.argsort(descending=True, stable=True)
        positions = th.arange(count, device=p.device)
        keep = th.ones_like(keep)
        for group in (i, j):
            order = base[group[base].argsort(stable=True)]
            groups = group[order]
            first = th.cat((groups.new_ones(1, dtype=th.bool), groups[1:] != groups[:-1]))
            starts = th.where(first, positions, 0).cummax(0).values
            selected = th.zeros_like(keep)
            selected[order] = positions - starts < knn
            keep &= selected
    elif method == "greedy":
        capacities = []
        for value, size, limit in ((sx, nx, ny), (sy, ny, nx)):
            if value is None:
                capacity = th.full((size,), limit)
            else:
                capacity = th.as_tensor(value).detach().cpu().expand(size)
            if (capacity < 0).any() or (capacity != capacity.trunc()).any():
                raise ValueError("sx/sy must be nonnegative integer capacities or None.")
            capacities.append(capacity.long().tolist())
        cx, cy = capacities
        order = score.argsort(descending=True, stable=True)
        ordered = th.stack((order, i[order], j[order]), 1).cpu().tolist()
        chosen = []
        for edge, x, y in ordered:
            if cx[x] > 0 and cy[y] > 0:
                chosen.append(edge)
                cx[x] -= 1
                cy[y] -= 1
        keep[th.as_tensor(chosen, device=p.device, dtype=th.long)] = True
    elif method == "assignment":
        gain = score.double() - assignment_min_score
        positive = gain > 0
        # Nonzero dummy weights are required by the sparse graph solver.
        offset = th.finfo(th.float64).tiny
        rows = th.cat((i[positive].cpu(), th.arange(nx)))
        cols = th.cat((j[positive].cpu(), ny + th.arange(nx)))
        weights = th.cat((gain[positive].cpu() + offset, th.full((nx,), offset, dtype=th.float64)))
        graph = ssp.coo_matrix((weights.numpy(), (rows.numpy(), cols.numpy())), shape=(nx, ny + nx)).tocsr()
        rows, cols = [th.as_tensor(v, device=p.device) for v in
                      ssp.csgraph.min_weight_full_bipartite_matching(graph, maximize=True)]
        real = cols < ny
        keep = th.isin(i * ny + j, rows[real] * ny + cols[real]) & positive
    else:
        raise ValueError("method must be soft, mutual, greedy or assignment.")

    return keep